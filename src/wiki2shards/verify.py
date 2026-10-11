"""Dataset verification (G2). Exit codes (PLAN B9):
    0  everything checks out
    1  verification ran and found a defect
    3  cannot verify: no manifest (uncommitted dataset), unreadable manifest, IO error

header mode: manifest, file presence and sizes, every .idx header against the manifest.
full mode: additionally every SHA-256, segment invariants, token range, pad and EOS rules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wiki2shards.io.binfmt import (
    PACK_MODES,
    SPLITS,
    FormatError,
    _counts,
    check_segments,
    check_tokens,
    decode_idx,
    idx_size,
    open_bin,
    read_idx_header,
)
from wiki2shards.io.hashing import parse_uid_hex, sha256_file
from wiki2shards.io.manifest import MANIFEST_NAME, Manifest, ManifestError, read_manifest
from wiki2shards.io.records import iter_jsonl


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fatal: str | None = None
    dataset_id: str | None = None

    @property
    def exit_code(self) -> int:
        return 3 if self.fatal else (1 if self.errors else 0)


def verify_dataset(path: Path | str, *, full: bool = False, strict: bool = False) -> Report:
    root = Path(path)
    rep = Report()
    if not (root / MANIFEST_NAME).is_file():
        rep.fatal = f"{root}: no {MANIFEST_NAME} (not a committed dataset)"
        return rep
    try:
        manifest, rep.dataset_id = read_manifest(root)
    except ManifestError as exc:
        if "canonical" in str(exc) or "schema" in str(exc):
            rep.errors.append(str(exc))
        else:
            rep.fatal = str(exc)
        return rep
    except OSError as exc:
        rep.fatal = str(exc)
        return rep
    try:
        _check(root, manifest, rep, full=full, strict=strict)
    except OSError as exc:
        rep.fatal = f"IO error: {exc}"
    return rep


def _check(root: Path, m: Manifest, rep: Report, *, full: bool, strict: bool) -> None:
    itemsize = 2 if m.tokenizer.dtype == "<u2" else 4
    row_len = m.packing.row_len
    expected = {MANIFEST_NAME} | {s.name for s in m.side_files}
    for split in m.splits.values():
        for sh in split.shards:
            expected |= {sh.bin, sh.idx}
    for name in sorted(os.listdir(root)):
        if name not in expected:
            (rep.errors if strict else rep.warnings).append(f"unknown file {name}")

    for side in m.side_files:
        p = root / side.name
        if not p.is_file():
            rep.errors.append(f"missing {side.name}")
        elif p.stat().st_size != side.bytes:
            rep.errors.append(f"{side.name}: size {p.stat().st_size} != {side.bytes}")
        elif full and sha256_file(p) != side.sha256:
            rep.errors.append(f"{side.name}: sha256 mismatch")

    if m.packing.row_len != m.packing.seq_len + 1:
        rep.errors.append("packing.row_len != seq_len + 1")
    for split_name, split in m.splits.items():
        first_row = 0
        rows = segs_pad = 0
        for i, sh in enumerate(split.shards):
            where = f"{split_name} shard {sh.index}"
            b, x = root / sh.bin, root / sh.idx
            if sh.index != i:
                rep.errors.append(f"{where}: shard indices are not 0..n-1")
            if not b.is_file() or not x.is_file():
                rep.errors.append(f"{where}: missing .bin or .idx")
                continue
            if b.stat().st_size != sh.bin_bytes or sh.bin_bytes != sh.n_rows * row_len * itemsize:
                rep.errors.append(f"{where}: .bin size {b.stat().st_size} does not match")
                continue
            try:
                hdr = read_idx_header(x)
            except FormatError as exc:
                rep.errors.append(f"{where}: {exc}")
                continue
            if x.stat().st_size != sh.idx_bytes or sh.idx_bytes != idx_size(
                hdr.n_rows, hdr.n_segments
            ):
                rep.errors.append(f"{where}: .idx size does not match")
                continue
            want = (
                itemsize,
                PACK_MODES[m.packing.mode],
                SPLITS[split_name],
                row_len,
                sh.index,
                sh.n_rows,
                first_row,
                bytes.fromhex(sh.bin_sha256),
                bytes.fromhex(m.tokenizer.fingerprint),
            )
            got = (
                hdr.itemsize,
                hdr.pack_mode,
                hdr.split,
                hdr.row_len,
                hdr.shard_index,
                hdr.n_rows,
                hdr.first_global_row,
                hdr.bin_sha256,
                hdr.tokenizer_fp,
            )
            if want != got:
                rep.errors.append(f"{where}: .idx header disagrees with the manifest")
            first_row += sh.n_rows
            rows += sh.n_rows
            segs_pad += hdr.n_pad_tokens
            if full:
                if sha256_file(b) != sh.bin_sha256:
                    rep.errors.append(f"{where}: .bin sha256 mismatch")
                if sha256_file(x) != sh.idx_sha256:
                    rep.errors.append(f"{where}: .idx sha256 mismatch")
        if rows != split.n_rows:
            rep.errors.append(
                f"{split_name}: shards hold {rows} rows, manifest says {split.n_rows}"
            )
        if (
            segs_pad != split.n_pad_tokens
            or split.n_rows * row_len != split.n_tokens + split.n_pad_tokens
        ):
            rep.errors.append(f"{split_name}: token/pad totals do not add up")
    if full and not rep.errors:
        _check_contents(root, m, rep, itemsize)


def _check_contents(root: Path, m: Manifest, rep: Report, itemsize: int) -> None:
    """Segment invariants, token range, pads, piece bookkeeping and the EOS rule."""
    docs: dict[str, dict[int, tuple[int, int]]] = {"train": {}, "val": {}}
    for d in iter_jsonl(root / "docs.jsonl"):
        docs[d["split"]][parse_uid_hex(d["doc_uid"])] = (d["n_pieces"], d["n_tokens"])
    row_len, eos = m.packing.row_len, m.tokenizer.eos_id
    partial_last = m.packing.mode == "concat" and m.packing.last_row == "pad"
    for split_name, split in m.splits.items():
        if split.n_docs != len(docs[split_name]):
            rep.errors.append(f"{split_name}: n_docs {split.n_docs} != docs.jsonl count")
        uids, pieces, lens, last_tok = [], [], [], []
        for k, sh in enumerate(split.shards):
            hdr, seg = decode_idx((root / sh.idx).read_bytes())
            last_shard = k == len(split.shards) - 1
            errs = check_segments(
                seg,
                row_len=row_len,
                pack_mode=hdr.pack_mode,
                allow_partial_last_row=partial_last and last_shard,
            )
            toks = open_bin(root / sh.bin, itemsize, sh.n_rows, row_len)
            errs += check_tokens(
                toks, seg, vocab_size=m.tokenizer.vocab_size, pad_id=m.tokenizer.pad_id
            )
            rep.errors += [f"{split_name} shard {sh.index}: {e}" for e in errs]
            if errs:
                return
            row_of = np.repeat(np.arange(seg.n_rows), _counts(seg.row_seg_ptr))
            end = seg.start.astype(np.int64) + seg.length.astype(np.int64) - 1
            uids.append(seg.doc_uid)
            pieces.append(seg.piece.astype(np.int64))
            lens.append(seg.length.astype(np.int64))
            last_tok.append(np.asarray(toks[row_of, end]).astype(np.int64))
        _check_docs(
            rep,
            split_name,
            docs[split_name],
            uids,
            pieces,
            lens,
            last_tok,
            eos,
            split.n_tokens_dropped_tail,
            m.packing.mode == "concat",
        )


def _check_docs(
    rep: Report,
    split: str,
    docs: dict[int, tuple[int, int]],
    uids_l: list[np.ndarray],
    pieces_l: list[np.ndarray],
    lens_l: list[np.ndarray],
    last_l: list[np.ndarray],
    eos: int,
    dropped_tail: int,
    concat: bool,
) -> None:
    order = np.array(sorted(docs), dtype=np.uint64)
    n_pieces = np.array([docs[int(u)][0] for u in order], dtype=np.int64)
    n_tokens = np.array([docs[int(u)][1] for u in order], dtype=np.int64)
    if not uids_l:
        stored = np.zeros(order.shape[0], dtype=np.int64)
    else:
        uids, pieces = np.concatenate(uids_l), np.concatenate(pieces_l)
        lens, last = np.concatenate(lens_l), np.concatenate(last_l)
        di = np.searchsorted(order, uids)
        if np.any(di >= order.shape[0]) or np.any(
            order[np.minimum(di, order.shape[0] - 1)] != uids
        ):
            rep.errors.append(f"{split}: a segment names a doc missing from docs.jsonl")
            return
        count = np.bincount(di, minlength=order.shape[0])
        stored = np.bincount(di, weights=lens, minlength=order.shape[0]).astype(np.int64)
        if np.any(count != n_pieces):
            rep.errors.append(f"{split}: segment counts disagree with docs.jsonl n_pieces")
        if np.unique(di * (1 << 32) + pieces).shape[0] != di.shape[0] or np.any(
            pieces >= n_pieces[di]
        ):
            rep.errors.append(f"{split}: a doc's pieces are not exactly 0..n_pieces-1")
        if concat:  # pieces appear in increasing order in global row order
            o = np.lexsort((np.arange(di.shape[0]), di))
            same = di[o][1:] == di[o][:-1]
            if np.any((pieces[o][1:] - pieces[o][:-1])[same] != 1):
                rep.errors.append(f"{split}: pieces out of order")
        full_doc = stored[di] == n_tokens[di]
        is_last = pieces == n_pieces[di] - 1
        if np.any(last[full_doc & is_last] != eos):
            rep.errors.append(f"{split}: a document's last piece does not end with eos_id")
    short = n_tokens - stored
    if np.any(short < 0) or int(short.sum()) != dropped_tail:
        rep.errors.append(f"{split}: stored + dropped tail tokens != docs.jsonl n_tokens")
    elif dropped_tail:  # only the final docs in pack order may be cut by the dropped tail
        cut = np.nonzero(short)[0]
        if cut.size and (np.any(short[cut[0] + 1 :] != n_tokens[cut[0] + 1 :])):
            rep.errors.append(f"{split}: a doc before the tail is missing tokens")
