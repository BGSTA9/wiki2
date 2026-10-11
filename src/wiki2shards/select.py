"""Selection: soft filters named by the policy, exact dedup, hash split (G9).

Order (PLAN C8): policy filters -> exact dedup (smallest page_id per text_sha256 among the
survivors) -> split. Dedup runs before the split, so identical text cannot land in both splits.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wiki2shards.io.atomic import atomic_write_bytes
from wiki2shards.io.binfmt import U8
from wiki2shards.io.hashing import canonical_json_pretty, parse_uid_hex
from wiki2shards.io.records import iter_jsonl
from wiki2shards.tokenize.cache import read_token_part


class SelectionError(Exception):
    pass


def is_val(uid: int, val_per_10k: int) -> bool:
    """Train/val assignment depends on document identity only."""
    return uid % 10_000 < val_per_10k


def _sum_stats(paths: list[Path]) -> dict[str, int]:
    total: dict[str, int] = {}
    for p in paths:
        for k, v in json.loads(p.read_bytes()).items():
            total[k] = total.get(k, 0) + int(v)
    return dict(sorted(total.items()))


def select(
    cleaned_dir: Path,
    tokens_dir: Path,
    apply: list[str],
    val_per_10k: int,
    extracted_dir: Path | None = None,
) -> tuple[dict[str, NDArray[np.uint64]], dict[str, Any]]:
    """Return ({"train": uids, "val": uids}, stats). Both uid arrays ascending."""
    uids: list[int] = []
    page_ids: list[int] = []
    shas: list[str] = []
    keep: list[bool] = []
    failed = dict.fromkeys(sorted(apply), 0)
    n_tokens: list[int] = []
    for part in sorted(cleaned_dir.glob("part-*.jsonl")):
        stem = part.name.removesuffix(".jsonl")
        tp = read_token_part(tokens_dir / f"{stem}.tok.bin", tokens_dir / f"{stem}.tok.idx")
        lengths = tp.n_tokens.tolist()
        for i, doc in enumerate(iter_jsonl(part)):
            uid = parse_uid_hex(doc["doc_uid"])
            if int(tp.doc_uids[i]) != uid:
                raise SelectionError(f"{stem}: token cache and cleaned part disagree at doc {i}")
            ok = True
            for name in apply:
                if not doc["verdicts"][name]:
                    failed[name] += 1
                    ok = False
            uids.append(uid)
            page_ids.append(doc["page_id"])
            shas.append(doc["text_sha256"])
            keep.append(ok)
            n_tokens.append(lengths[i])

    uid_arr = np.array(uids, dtype=U8)
    if np.unique(uid_arr).shape[0] != uid_arr.shape[0]:
        raise SelectionError("doc_uid collision: two documents share a 64-bit id")

    # Exact dedup among survivors: keep the smallest page_id for each text_sha256.
    survivors = [i for i in range(len(uids)) if keep[i]]
    survivors.sort(key=lambda i: (shas[i], page_ids[i]))
    chosen, seen = [], set()
    for i in survivors:
        if shas[i] not in seen:
            seen.add(shas[i])
            chosen.append(i)
    dedup_removed = len(survivors) - len(chosen)

    chosen_uids = np.sort(uid_arr[np.array(chosen, dtype=np.int64)]) if chosen else uid_arr[:0]
    val_mask = (chosen_uids % np.uint64(10_000)) < np.uint64(val_per_10k)
    splits = {"train": chosen_uids[~val_mask], "val": chosen_uids[val_mask]}

    stats: dict[str, Any] = {
        "docs_in": len(uids),
        "failed_filter": failed,
        "after_filters": len(survivors),
        "dedup_removed": dedup_removed,
        "selected": len(chosen),
        "train_docs": int(splits["train"].shape[0]),
        "val_docs": int(splits["val"].shape[0]),
        "clean": _sum_stats(sorted(cleaned_dir.glob("part-*.stats.json"))),
    }
    if extracted_dir is not None:
        stats["extract"] = _sum_stats(sorted(extracted_dir.glob("part-*.stats.json")))
    return splits, stats


def write_selection(
    out_dir: Path, splits: dict[str, NDArray[np.uint64]], stats: dict[str, Any]
) -> list[str]:
    for name, arr in splits.items():
        buf = io.BytesIO()
        np.save(buf, arr.astype(U8), allow_pickle=False)
        atomic_write_bytes(out_dir / f"{name}.npy", buf.getvalue())
    atomic_write_bytes(out_dir / "stats.json", canonical_json_pretty(stats))
    return ["stats.json", "train.npy", "val.npy"]


def read_selection(sel_dir: Path) -> dict[str, NDArray[np.uint64]]:
    return {s: np.load(sel_dir / f"{s}.npy", allow_pickle=False) for s in ("train", "val")}
