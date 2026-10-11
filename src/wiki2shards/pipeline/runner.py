"""The build: six file-producing stages in a fixed order (verify runs after them, in the CLI).

Each stage is skipped when its stage key matches DONE.json and every journaled file verifies;
otherwise committed units that verify are kept and the rest are recomputed. Output never depends on
the worker count or on the order in which pool results complete (G1).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from wiki2shards.acquire.download import (
    dump_names,
    fetch,
    parse_sha1sums,
    source_url,
    transport_for,
)
from wiki2shards.clean.wikitext import clean_part
from wiki2shards.config import BuildConfig
from wiki2shards.extract.multistream import plan_units, stream_offsets, write_siteinfo
from wiki2shards.extract.xmlpages import extract_unit
from wiki2shards.io.atomic import atomic_write_bytes
from wiki2shards.io.hashing import canonical_json_pretty, sha1_file, sha256_file
from wiki2shards.io.manifest import FORMAT, FORMAT_VERSION, Manifest, write_manifest
from wiki2shards.pack import writer
from wiki2shards.pipeline import faults
from wiki2shards.pipeline.provenance import build_info
from wiki2shards.pipeline.stage import STAGES, StageError, StageState, stage_key
from wiki2shards.select import read_selection, select, write_selection
from wiki2shards.tokenize.base import Tokenizer, effective_pad_id, load_tokenizer
from wiki2shards.tokenize.cache import tokenize_part

# Bump a stage's version whenever a code change alters that stage's output bytes.
STAGE_VERSIONS = {"download": 1, "extract": 1, "clean": 1, "tokenize": 1, "select": 1, "pack": 1}
PART = r"part-\d{5}"


@dataclass
class Job:
    unit: str
    fn: Callable[..., list[str]]  # must be a top-level function when run in the pool
    args: tuple[Any, ...] = ()


def _call(fn: Callable[..., list[str]], args: tuple[Any, ...]) -> list[str]:
    return fn(*args)


def _exit_when_orphaned(parent: int) -> None:
    """Pool-worker initializer. If the coordinator dies (SIGKILL, or os._exit at a fault point),
    idle workers would otherwise wait on their queue forever: macOS has no parent-death signal.
    A daemon thread notices the re-parenting and exits the worker."""

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(0.5)
        os._exit(1)

    threading.Thread(target=watch, daemon=True, name="w2s-orphan-watch").start()


def run_jobs(state: StageState, jobs: list[Job], workers: int) -> None:
    """Run jobs inline or in a spawn pool; journal each unit once its files are in place."""

    def commit(job: Job, files: list[str]) -> None:
        faults.hit(state.stage, "before_journal_append")
        state.commit(job.unit, files)
        faults.hit(state.stage, "after_unit_commit")

    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            commit(job, _call(job.fn, job.args))
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_exit_when_orphaned,
        initargs=(os.getpid(),),
    ) as pool:
        futures = {pool.submit(_call, job.fn, job.args): job for job in jobs}
        results = [(futures[f], f.result()) for f in as_completed(futures)]
        if os.environ.get("W2S_TEST_COMPLETION_ORDER") == "reverse":  # test hook (PLAN C15)
            results.sort(key=lambda r: r[0].unit, reverse=True)
        for job, files in results:  # completion order: harmless, since files are named by unit
            commit(job, files)


class Build:
    def __init__(self, cfg: BuildConfig, workers: int | None = None) -> None:
        self.cfg = cfg
        self.workers = workers or cfg.workers
        self.work = Path(cfg.workdir)
        self.out = Path(cfg.output)
        self.raw = self.work / "raw"
        self.extracted = self.work / "extracted"
        self.cleaned = self.work / "cleaned"
        self.tokens = self.work / "tokens"
        self.selection = self.work / "selection" / cfg.select.policy
        self.names = dump_names(cfg.source.wiki, cfg.source.date)
        self.fps: dict[str, str] = {}  # stage -> output fingerprint

    def run(self, from_stage: str | None = None, to_stage: str | None = None) -> None:
        """Run stages up to to_stage; from_stage forces that stage and everything after it."""
        stop = STAGES.index(to_stage) if to_stage else len(STAGES) - 1
        force_from = STAGES.index(from_stage) if from_stage else len(STAGES)
        for i, stage in enumerate(STAGES[: stop + 1]):
            getattr(self, f"_{stage}")(force=i >= force_from)

    def _execute(
        self,
        stage: str,
        out_dir: Path,
        patterns: list[str],
        config: Any,
        inputs: dict[str, str],
        plan: Callable[[], list[Job]],
        force: bool,
        workers: int = 1,
    ) -> None:
        state = StageState(self.work, stage, out_dir, patterns)
        key = stage_key(stage, STAGE_VERSIONS[stage], config, inputs)
        if not force and (fp := state.is_done(key)) is not None:
            self.fps[stage] = fp
            return
        committed = state.begin(key, force=force)
        run_jobs(state, [job for job in plan() if job.unit not in committed], workers)
        faults.hit(stage, "before_done")
        self.fps[stage] = state.finish(key)

    # ---- stages -------------------------------------------------------------------------
    def _download(self, force: bool) -> None:
        src, n, raw = self.cfg.source, self.names, self.raw
        transport = transport_for(src.mirror)

        def get(key: str) -> list[str]:
            sha1 = None
            if key != "sha1sums":
                sums = parse_sha1sums((raw / n["sha1sums"]).read_text())
                if n[key] not in sums:
                    raise StageError(f"{n['sha1sums']} does not list {n[key]}")
                sha1 = sums[n[key]]
            fetch(source_url(src.mirror, src.wiki, src.date, n[key]), raw / n[key], transport, sha1)
            return [n[key]]

        def record() -> list[str]:
            files = sorted(
                (
                    {
                        "name": name,
                        "bytes": (raw / name).stat().st_size,
                        "sha1": sha1_file(raw / name),
                    }
                    for name in n.values()
                ),
                key=lambda f: str(f["name"]),
            )
            body = {"dump_date": src.date, "files": files, "wiki": src.wiki}
            atomic_write_bytes(raw / "source.json", canonical_json_pretty(body))
            return ["source.json"]

        jobs = [
            Job("0-sha1sums", get, ("sha1sums",)),
            Job("1-index", get, ("index",)),
            Job("2-xml", get, ("xml",)),
            Job("3-source", record),
        ]
        patterns = [re.escape(name) for name in n.values()] + [r"source\.json"]
        self._execute(
            "download", raw, patterns, {"wiki": src.wiki, "date": src.date}, {}, lambda: jobs, force
        )

    def _extract(self, force: bool) -> None:
        xml = self.raw / self.names["xml"]
        target = self.cfg.extract.target_bytes

        def plan() -> list[Job]:
            offsets = stream_offsets(self.raw / self.names["index"])
            units = plan_units(offsets, xml.stat().st_size, target)
            jobs = [
                Job(
                    "siteinfo",
                    write_siteinfo,
                    (xml, offsets[0], self.extracted, self.cfg.source.wiki),
                )
            ]
            return jobs + [Job(u.name, extract_unit, (xml, u, self.extracted)) for u in units]

        self._execute(
            "extract",
            self.extracted,
            [rf"{PART}\.(jsonl|stats\.json)", r"siteinfo\.json"],
            {"unit_bytes": target, "wiki": self.cfg.source.wiki},
            {"download": self.fps["download"]},
            plan,
            force,
            self.workers,
        )

    def _clean(self, force: bool) -> None:
        wiki, clean_cfg = self.cfg.source.wiki, self.cfg.clean

        def plan() -> list[Job]:
            return [
                Job(p.name.removesuffix(".jsonl"), clean_part, (p, self.cleaned, wiki, clean_cfg))
                for p in sorted(self.extracted.glob("part-*.jsonl"))
            ]

        self._execute(
            "clean",
            self.cleaned,
            [rf"{PART}\.(jsonl|stats\.json)"],
            {"clean": self.cfg.hashed_view()["clean"], "wiki": wiki},
            {"extract": self.fps["extract"]},
            plan,
            force,
            self.workers,
        )

    def _tokenize(self, force: bool) -> None:
        spec = self.cfg.tokenizer
        tok = load_tokenizer(spec)

        def plan() -> list[Job]:
            return [
                Job(p.name.removesuffix(".jsonl"), tokenize_part, (p, self.tokens, spec))
                for p in sorted(self.cleaned.glob("part-*.jsonl"))
            ]

        self._execute(
            "tokenize",
            self.tokens,
            [rf"{PART}\.tok\.(bin|idx)"],
            {"tokenizer_fingerprint": tok.fingerprint},
            {"clean": self.fps["clean"]},
            plan,
            force,
            self.workers,
        )

    def _select(self, force: bool) -> None:
        sel = self.cfg.select

        def run() -> list[str]:
            splits, stats = select(
                self.cleaned, self.tokens, sorted(sel.apply), sel.val_per_10k, self.extracted
            )
            return write_selection(self.selection, splits, stats)

        self._execute(
            "select",
            self.selection,
            [r"(train|val)\.npy", r"stats\.json"],
            sel.model_dump(mode="python"),
            {k: self.fps[k] for k in ("extract", "clean", "tokenize")},
            lambda: [Job("selection", run)],
            force,
        )

    def _pack(self, force: bool) -> None:
        cfg = self.cfg
        tok = load_tokenizer(cfg.tokenizer)
        config_sha256 = cfg.config_sha256()

        def plan() -> list[Job]:
            store = writer.TokenStore(self.tokens)
            sel = read_selection(self.selection)
            plans = {s: writer.plan_split(s, sel[s], store, cfg.pack) for s in writer.SPLIT_NAMES}
            shard = ShardJob(effective_pad_id(tok), tok.fingerprint, cfg.pack.mode)
            jobs = [
                Job(writer.shard_stem(s, i), shard, (p, store, i, self.out))
                for s, p in plans.items()
                for i in range(p.n_shards)
            ]
            jobs.append(Job("~side-files", self._side_files, (plans,)))
            jobs.append(Job("~~manifest", self._manifest, (plans, store, tok, config_sha256)))
            return jobs

        patterns = [
            r"(train|val)-\d{5}\.(bin|idx)",
            r"docs\.jsonl",
            r"stats\.json",
            r"dataset\.manifest",
        ]
        self._execute(
            "pack",
            self.out,
            patterns,
            {"name": cfg.name, "config_sha256": config_sha256},
            {k: self.fps[k] for k in ("download", "clean", "tokenize", "select")},
            plan,
            force,
        )

    def _side_files(self, plans: dict[str, writer.SplitPlan]) -> list[str]:
        writer.write_docs_jsonl(
            self.out, plans, writer.doc_meta(self.cleaned), self.cfg.source.wiki
        )
        sel_stats = json.loads((self.selection / "stats.json").read_bytes())
        writer.write_stats(self.out, writer.build_stats(plans, sel_stats))
        return ["docs.jsonl", "stats.json"]

    def _manifest(
        self,
        plans: dict[str, writer.SplitPlan],
        store: writer.TokenStore,
        tok: Tokenizer,
        config_sha256: str,
    ) -> list[str]:
        cfg = self.cfg
        splits = {
            name: {
                "n_rows": p.layout.n_rows,
                "n_tokens": int(p.layout.length.sum()),
                "n_pad_tokens": p.layout.n_pad_tokens,
                "n_docs": int(p.uids.shape[0]),
                "n_tokens_dropped_tail": p.layout.n_tokens_dropped_tail,
                "shards": [
                    writer.shard_entry(self.out, p, i).model_dump() for i in range(p.n_shards)
                ],
            }
            for name, p in plans.items()
        }
        manifest = Manifest.model_validate(
            {
                "format": FORMAT,
                "format_version": list(FORMAT_VERSION),
                "name": cfg.name,
                "build": build_info(config_sha256),
                "source": json.loads((self.raw / "source.json").read_bytes()),
                "tokenizer": {
                    "spec": tok.name,
                    "fingerprint": tok.fingerprint,
                    "vocab_size": tok.vocab_size,
                    "eos_id": tok.eos_id,
                    "pad_id": effective_pad_id(tok),
                    "dtype": writer.dtype_str(store),
                },
                "packing": {
                    "mode": cfg.pack.mode,
                    "seq_len": cfg.pack.seq_len,
                    "row_len": cfg.pack.row_len,
                    "doc_order": "doc_uid_asc",
                    "last_row": cfg.pack.effective_last_row,
                },
                "selection": {
                    "policy": cfg.select.policy,
                    "soft_filters": sorted(cfg.select.apply),
                    "dedup": "exact_text_sha256",
                    "val_per_10k": cfg.select.val_per_10k,
                },
                "splits": splits,
                "side_files": [
                    {
                        "name": n,
                        "bytes": (self.out / n).stat().st_size,
                        "sha256": sha256_file(self.out / n),
                    }
                    for n in ("docs.jsonl", "stats.json")
                ],
            }
        )
        faults.hit("pack", "before_manifest")
        write_manifest(self.out, manifest)
        faults.hit("pack", "after_manifest")
        return ["dataset.manifest"]


class ShardJob:
    """writer.write_shard with the dataset-wide arguments bound."""

    def __init__(self, pad_id: int, fingerprint: str, mode: str) -> None:
        self.pad_id, self.fingerprint, self.mode = pad_id, fingerprint, mode

    def __call__(
        self, plan: writer.SplitPlan, store: writer.TokenStore, index: int, out: Path
    ) -> list[str]:
        return writer.write_shard(
            plan,
            store,
            index,
            out,
            pad_id=self.pad_id,
            tokenizer_fp=self.fingerprint,
            pack_mode=self.mode,
        )
