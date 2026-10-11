"""Turn token caches + a selection into a dataset directory.

Pack order is ascending doc_uid. Each split gets its own shard series {split}-{index:05d}.{bin,idx}.
Shards are written atomically (.bin first: the .idx header carries the .bin's SHA-256), then
docs.jsonl, stats.json and, last, dataset.manifest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wiki2shards.config import PackConfig
from wiki2shards.io.atomic import atomic_open, atomic_write_bytes
from wiki2shards.io.binfmt import PACK_MODES, SPLITS, U2, U4, IdxHeader, Segments, encode_idx
from wiki2shards.io.hashing import canonical_json, canonical_json_pretty, sha256_file, uid_hex
from wiki2shards.io.manifest import ShardEntry
from wiki2shards.io.records import iter_jsonl
from wiki2shards.pack.bfd import bfd_layout
from wiki2shards.pack.concat import Layout, concat_layout
from wiki2shards.pipeline import faults
from wiki2shards.tokenize.cache import TokenPart, read_token_part

SPLIT_NAMES = ("train", "val")


class TokenStore:
    """Random access to every cached document's tokens, by doc_uid."""

    def __init__(self, tokens_dir: Path) -> None:
        self.parts: list[TokenPart] = []
        uids, part_of, starts, ends = [], [], [], []
        for i, idx in enumerate(sorted(tokens_dir.glob("part-*.tok.idx"))):
            tp = read_token_part(idx.with_suffix(".bin"), idx)
            self.parts.append(tp)
            uids.append(tp.doc_uids)
            part_of.append(np.full(tp.doc_uids.shape[0], i, dtype=np.int64))
            starts.append(tp.offsets[:-1].astype(np.int64))
            ends.append(tp.offsets[1:].astype(np.int64))
        all_uids = np.concatenate(uids) if uids else np.zeros(0, dtype=np.uint64)
        order = np.argsort(all_uids, kind="stable")
        self.uids = all_uids[order]
        self.part_of = np.concatenate(part_of)[order] if uids else np.zeros(0, np.int64)
        self.start = np.concatenate(starts)[order] if uids else np.zeros(0, np.int64)
        self.end = np.concatenate(ends)[order] if uids else np.zeros(0, np.int64)
        self.dtype = self.parts[0].tokens.dtype if self.parts else U2

    def locate(self, uids: NDArray[np.uint64]) -> NDArray[np.int64]:
        pos = np.searchsorted(self.uids, uids)
        if pos.size and (pos.max() >= self.uids.size or np.any(self.uids[pos] != uids)):
            raise KeyError("selection names a doc_uid missing from the token cache")
        return pos.astype(np.int64)

    def tokens(self, pos: int) -> NDArray[np.unsignedinteger]:
        part = self.parts[int(self.part_of[pos])]
        return part.tokens[int(self.start[pos]) : int(self.end[pos])]


@dataclass(frozen=True)
class SplitPlan:
    name: str
    uids: NDArray[np.uint64]  # ascending = pack order
    positions: NDArray[np.int64]  # into TokenStore
    lengths: NDArray[np.int64]
    layout: Layout
    rows_per_shard: int

    @property
    def n_shards(self) -> int:
        return -(-self.layout.n_rows // self.rows_per_shard)

    def shard_rows(self, index: int) -> tuple[int, int]:
        r0 = index * self.rows_per_shard
        return r0, min(r0 + self.rows_per_shard, self.layout.n_rows)


def plan_split(
    name: str, uids: NDArray[np.uint64], store: TokenStore, cfg: PackConfig
) -> SplitPlan:
    uids = np.sort(uids)
    pos = store.locate(uids)
    lengths = (store.end[pos] - store.start[pos]).astype(np.int64)
    if cfg.mode == "concat":
        layout = concat_layout(lengths, cfg.row_len, cfg.effective_last_row)
    else:
        layout = bfd_layout(lengths, cfg.row_len, cfg.bfd_window)
    itemsize = store.dtype.itemsize
    rows_per_shard = cfg.target_shard_bytes // (cfg.row_len * itemsize)
    if rows_per_shard < 1:
        raise ValueError("shard size is smaller than one row")
    return SplitPlan(name, uids, pos, lengths, layout, rows_per_shard)


def shard_stem(split: str, index: int) -> str:
    return f"{split}-{index:05d}"


def fill_rows(
    plan: SplitPlan, store: TokenStore, r0: int, r1: int, pad_id: int
) -> NDArray[np.unsignedinteger]:
    lay = plan.layout
    buf = np.full((r1 - r0, lay.row_len), pad_id, dtype=store.dtype)
    s = lay.rows_slice(r0, r1)
    for row, start, length, doc, off in zip(
        lay.row[s].tolist(),
        lay.start[s].tolist(),
        lay.length[s].tolist(),
        lay.doc[s].tolist(),
        lay.doc_offset[s].tolist(),
        strict=True,
    ):
        toks = store.tokens(int(plan.positions[doc]))
        buf[row - r0, start : start + length] = toks[off : off + length]
    return buf


def write_shard(
    plan: SplitPlan,
    store: TokenStore,
    index: int,
    out_dir: Path,
    *,
    pad_id: int,
    tokenizer_fp: str,
    pack_mode: str,
) -> list[str]:
    r0, r1 = plan.shard_rows(index)
    lay = plan.layout
    s = lay.rows_slice(r0, r1)
    rows = fill_rows(plan, store, r0, r1, pad_id)
    stem = shard_stem(plan.name, index)

    h = hashlib.sha256()
    with atomic_open(out_dir / f"{stem}.bin") as f:
        half = (r1 - r0) // 2
        for i, block in enumerate((rows[:half], rows[half:])):
            data = np.ascontiguousarray(block).tobytes()
            f.write(data)
            h.update(data)
            if i == 0:
                faults.hit("pack", "mid_shard")  # crash with a half-written tmp file

    counts = np.bincount(lay.row[s] - r0, minlength=r1 - r0)
    seg = Segments(
        row_seg_ptr=np.concatenate([[0], np.cumsum(counts)]).astype(np.uint64),
        doc_uid=plan.uids[lay.doc[s]].astype(np.uint64),
        start=lay.start[s].astype(np.uint32),
        length=lay.length[s].astype(np.uint32),
        piece=lay.piece[s].astype(np.uint32),
    )
    header = IdxHeader(
        itemsize=store.dtype.itemsize,
        pack_mode=PACK_MODES[pack_mode],
        split=SPLITS[plan.name],
        row_len=lay.row_len,
        shard_index=index,
        n_rows=r1 - r0,
        n_segments=seg.n_segments,
        n_pad_tokens=(r1 - r0) * lay.row_len - int(lay.length[s].sum()),
        first_global_row=r0,
        bin_sha256=h.digest(),
        tokenizer_fp=bytes.fromhex(tokenizer_fp),
    )
    atomic_write_bytes(out_dir / f"{stem}.idx", encode_idx(header, seg))
    return [f"{stem}.bin", f"{stem}.idx"]


def shard_entry(out_dir: Path, plan: SplitPlan, index: int) -> ShardEntry:
    stem = shard_stem(plan.name, index)
    b, i = out_dir / f"{stem}.bin", out_dir / f"{stem}.idx"
    r0, r1 = plan.shard_rows(index)
    return ShardEntry(
        index=index,
        n_rows=r1 - r0,
        bin=b.name,
        bin_bytes=b.stat().st_size,
        bin_sha256=sha256_file(b),
        idx=i.name,
        idx_bytes=i.stat().st_size,
        idx_sha256=sha256_file(i),
    )


def doc_meta(cleaned_dir: Path) -> dict[str, dict[str, Any]]:
    meta = {}
    for part in sorted(cleaned_dir.glob("part-*.jsonl")):
        for doc in iter_jsonl(part):
            meta[doc["doc_uid"]] = {k: doc[k] for k in ("page_id", "rev_id", "title")}
    return meta


def write_docs_jsonl(
    out_dir: Path, plans: dict[str, SplitPlan], meta: dict[str, dict[str, Any]], wiki: str
) -> None:
    rows: list[tuple[int, bytes]] = []
    for plan in plans.values():
        n_pieces = np.bincount(plan.layout.doc, minlength=plan.uids.shape[0])
        for d, uid in enumerate(plan.uids.tolist()):
            h = uid_hex(uid)
            rec = {
                "doc_uid": h,
                "n_pieces": int(n_pieces[d]),
                "n_tokens": int(plan.lengths[d]),
                "split": plan.name,
                "wiki": wiki,
                **meta[h],
            }
            rows.append((uid, canonical_json(rec) + b"\n"))
    rows.sort()
    with atomic_open(out_dir / "docs.jsonl") as f:
        for _, line in rows:
            f.write(line)


def _hist(lengths: NDArray[np.int64]) -> dict[str, int]:
    """Doc length histogram over power-of-two bins: key "2^k" counts lengths in [2^k, 2^(k+1))."""
    if lengths.size == 0:
        return {}
    exps = np.floor(np.log2(np.maximum(lengths, 1))).astype(np.int64)
    values, counts = np.unique(exps, return_counts=True)
    return {f"2^{int(v):02d}": int(c) for v, c in zip(values, counts, strict=True)}


def build_stats(plans: dict[str, SplitPlan], selection_stats: dict[str, Any]) -> dict[str, Any]:
    splits = {}
    for name, plan in plans.items():
        lay = plan.layout
        stored = int(lay.length.sum())
        splits[name] = {
            "n_docs": int(plan.uids.shape[0]),
            "n_rows": lay.n_rows,
            "n_tokens": stored,
            "n_pad_tokens": lay.n_pad_tokens,
            "n_tokens_dropped_tail": lay.n_tokens_dropped_tail,
            "packing_efficiency": [stored, lay.n_rows * lay.row_len],
            "doc_length_histogram": _hist(plan.lengths),
        }
    return {"selection": selection_stats, "splits": splits}


def write_stats(out_dir: Path, stats: dict[str, Any]) -> None:
    atomic_write_bytes(out_dir / "stats.json", canonical_json_pretty(stats))


def dtype_str(store: TokenStore) -> str:
    return "<u2" if store.dtype == U2 else ("<u4" if store.dtype == U4 else str(store.dtype))
