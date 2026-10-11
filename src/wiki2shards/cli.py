"""wiki2shards CLI. Exit codes: 0 ok, 1 verification failed, 2 usage or config error,
3 integrity or IO error, 137 injected fault (tests only)."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from wiki2shards import __version__
from wiki2shards.acquire.download import DownloadError
from wiki2shards.config import BuildConfig, ConfigError, load_build_config
from wiki2shards.io.binfmt import FormatError
from wiki2shards.io.manifest import ManifestError
from wiki2shards.loader.api import LoaderStateError
from wiki2shards.loader.dataset import DatasetError
from wiki2shards.pipeline.stage import STAGES, StageError
from wiki2shards.select import SelectionError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Wikipedia dump -> checksummed token shards, plus a deterministic loader.",
)

ConfigOpt = Annotated[Path, typer.Option("--config", help="build config YAML")]
WorkersOpt = Annotated[int | None, typer.Option("--workers", help="override config workers")]


def _override(cfg: BuildConfig, **changes: Any) -> BuildConfig:
    """Return cfg with dotted-path fields replaced (None values are ignored)."""
    data = cfg.model_dump(mode="python")
    for dotted, value in changes.items():
        if value is None:
            continue
        node = data
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    try:
        return BuildConfig.model_validate(data)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _run_build(
    cfg: BuildConfig, from_stage: str | None, to_stage: str | None, workers: int | None = None
) -> None:
    from wiki2shards.pipeline.runner import Build

    Build(cfg, workers=workers).run(from_stage, to_stage)


def _stage_command(stage: str, cfg: BuildConfig, workers: int | None) -> None:
    """Run one stage of the configured build (journaled and resumable, like `build`)."""
    _run_build(cfg, stage, stage, workers)


@app.callback(invoke_without_command=True)
def _root(
    version: Annotated[bool, typer.Option("--version", help="print version")] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit(0)


@app.command()
def download(config: ConfigOpt, mirror: Annotated[str | None, typer.Option()] = None) -> None:
    """Fetch the dump, its index and sha1sums into <workdir>/raw and verify SHA-1."""
    _stage_command(
        "download", _override(load_build_config(config), **{"source.mirror": mirror}), None
    )


@app.command()
def extract(
    config: ConfigOpt,
    workers: WorkersOpt = None,
    unit_mb: Annotated[int | None, typer.Option("--unit-mb")] = None,
) -> None:
    """Multistream dump -> extracted/part-NNNNN.jsonl."""
    _stage_command(
        "extract", _override(load_build_config(config), **{"extract.unit_mb": unit_mb}), workers
    )


@app.command()
def clean(config: ConfigOpt, workers: WorkersOpt = None) -> None:
    """Wikitext -> cleaned text and filter verdicts."""
    _stage_command("clean", load_build_config(config), workers)


@app.command()
def tokenize(config: ConfigOpt, workers: WorkersOpt = None) -> None:
    """Cleaned text -> per-part token cache."""
    _stage_command("tokenize", load_build_config(config), workers)


@app.command("select")
def select_cmd(config: ConfigOpt) -> None:
    """Exact dedup, hash split, and the policy's filters."""
    _stage_command("select", load_build_config(config), None)


@app.command()
def pack(
    config: ConfigOpt,
    seq_len: Annotated[int | None, typer.Option("--seq-len")] = None,
    mode: Annotated[str | None, typer.Option("--mode")] = None,
    shard_mb: Annotated[int | None, typer.Option("--shard-mb")] = None,
    out: Annotated[str | None, typer.Option("--out")] = None,
) -> None:
    """Rows, shards, docs.jsonl and the manifest."""
    cfg = _override(
        load_build_config(config),
        **{"pack.seq_len": seq_len, "pack.mode": mode, "pack.shard_mb": shard_mb, "output": out},
    )
    _stage_command("pack", cfg, None)


@app.command()
def build(
    config: ConfigOpt,
    from_stage: Annotated[str | None, typer.Option("--from-stage")] = None,
    to_stage: Annotated[str | None, typer.Option("--to-stage")] = None,
    workers: WorkersOpt = None,
) -> None:
    """Every stage, resumable; then verify --full."""
    for s in (from_stage, to_stage):
        if s is not None and s not in STAGES:
            raise ConfigError(f"unknown stage {s!r}; choose from {', '.join(STAGES)}")
    cfg = load_build_config(config)
    _run_build(cfg, from_stage, to_stage, workers)
    if to_stage in (None, "pack"):
        _report(Path(cfg.output), full=True, strict=True)


@app.command("train-tokenizer")
def train_tokenizer(
    in_dir: Annotated[Path, typer.Option("--in", help="directory of cleaned part-*.jsonl")],
    out: Annotated[Path, typer.Option("--out")],
    vocab_size: Annotated[int, typer.Option("--vocab-size")] = 32_000,
    sample_docs: Annotated[int, typer.Option("--sample-docs")] = 100_000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Train a byte-level BPE tokenizer.json with HF tokenizers."""
    from wiki2shards.tokenize.train_bpe import train_to_file

    train_to_file(sorted(in_dir.glob("part-*.jsonl")), out, vocab_size, sample_docs, seed)
    typer.echo(out)


def _report(path: Path, full: bool, strict: bool) -> None:
    from wiki2shards.verify import verify_dataset

    rep = verify_dataset(path, full=full, strict=strict)
    for w in rep.warnings:
        typer.echo(f"warning: {w}", err=True)
    for e in rep.errors:
        typer.echo(f"error: {e}", err=True)
    if rep.fatal:
        typer.echo(f"fatal: {rep.fatal}", err=True)
    if rep.exit_code == 0:
        typer.echo(f"ok {path} dataset_id={rep.dataset_id}")
    raise typer.Exit(rep.exit_code)


@app.command()
def verify(dataset: Path, full: bool = False, strict: bool = False) -> None:
    """Integrity check: header mode by default, every SHA-256 with --full."""
    _report(dataset, full, strict)


@app.command()
def inspect(dataset: Path, split: str = "train", row: int = 0, decode: bool = False) -> None:
    """One row: token ids, segments and (optionally) decoded text."""
    from wiki2shards.inspect import inspect_row

    for line in inspect_row(dataset, split, row, decode):
        typer.echo(json.dumps(line, ensure_ascii=False))


@app.command()
def stats(dataset: Path, json_out: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Dataset statistics from stats.json and the manifest."""
    from wiki2shards.stats import dataset_stats

    s = dataset_stats(dataset)
    typer.echo(json.dumps(s, indent=None if json_out else 2, sort_keys=True))


@app.command("mix-plan")
def mix_plan(
    mix: Annotated[Path, typer.Option("--mix")],
    steps: Annotated[int, typer.Option("--steps")],
    global_batch: Annotated[int, typer.Option("--global-batch")],
) -> None:
    """Period, worst lag, and epochs per source at a horizon."""
    from wiki2shards.stats import mixture_plan

    typer.echo(json.dumps(mixture_plan(mix, steps, global_batch), indent=2, sort_keys=True))


@app.command()
def replay(
    mix: Annotated[Path, typer.Option("--mix")],
    global_batch: Annotated[int, typer.Option("--global-batch")],
    step: Annotated[int, typer.Option("--step")],
    decode: Annotated[int, typer.Option("--decode", help="decode the first N samples")] = 0,
    start_sample: Annotated[int | None, typer.Option("--start-sample")] = None,
) -> None:
    """Exact contents of global batch `step` (one JSON line per sample)."""
    from wiki2shards.inspect import tokenizer_for
    from wiki2shards.loader import Loader, Mixture

    m = Mixture.from_yaml(mix)
    t0 = start_sample if start_sample is not None else step * global_batch
    batch = Loader(m, global_batch, prefetch=0).make_batch(t0)
    tok = tokenizer_for(m.datasets[0]) if decode else None
    for j in range(batch.tokens.shape[0]):
        rec: dict[str, Any] = {
            "step": batch.step,
            "t": int(batch.sample_index[j]),
            "source": m.names[int(batch.source_index[j])],
            "row": int(batch.row_id[j]),
            "tokens_sha256": hashlib.sha256(batch.tokens[j].tobytes()).hexdigest(),
        }
        if tok is not None and j < decode:
            rec["text"] = tok.decode(batch.tokens[j])
        typer.echo(json.dumps(rec, ensure_ascii=False))


@app.command("bench-loader")
def bench_loader(
    mix: Annotated[Path, typer.Option("--mix")],
    global_batch: Annotated[int, typer.Option("--global-batch")],
    steps: Annotated[int, typer.Option("--steps")] = 200,
    prefetch: Annotated[int, typer.Option("--prefetch")] = 4,
) -> None:
    """Samples/s and p50/p99 data-wait as seen by a trainer."""
    from wiki2shards.loader import Loader, Mixture

    loader = Loader(Mixture.from_yaml(mix), global_batch, prefetch=prefetch)
    waits = []
    start = time.perf_counter()
    with loader:
        for _ in range(steps):
            t = time.perf_counter()
            next(loader)
            waits.append(time.perf_counter() - t)
    elapsed = time.perf_counter() - start
    w = np.array(waits) * 1000
    typer.echo(
        json.dumps(
            {
                "samples_per_s": round(steps * global_batch / elapsed, 1),
                "data_wait_ms_p50": round(float(np.percentile(w, 50)), 3),
                "data_wait_ms_p99": round(float(np.percentile(w, 99)), 3),
                "steps": steps,
                "global_batch": global_batch,
                "prefetch": prefetch,
            }
        )
    )


def main() -> None:
    """Console entry point. Typer runs in standalone mode, so usage errors exit 2 and
    typer.Exit codes pass through; domain errors are mapped to the SPEC exit codes here.
    (Typer >= 0.20 vendors its own click, so never catch click's exception classes.)"""
    try:
        app()
    except (ConfigError, StageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    except (
        DownloadError,
        FormatError,
        ManifestError,
        DatasetError,
        SelectionError,
        LoaderStateError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(3)
