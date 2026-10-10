"""The .bin/.idx shard format (SPEC 1.4). Frozen at contracts-v1.

.bin: n_rows x row_len tokens, row-major, <u2 or <u4, no header, no trailing bytes.
.idx: 128-byte little-endian header, then row_seg_ptr <u8[n_rows+1], seg_doc_uid <u8[n_seg],
      seg_start <u4[n_seg], seg_len <u4[n_seg], seg_piece <u4[n_seg].
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

IDX_MAGIC = b"W2S-IDX\0"
IDX_VERSION_MAJOR = 1
IDX_VERSION_MINOR = 0
IDX_STRUCT = struct.Struct("<8sHHBBBBIIQQQQ32s32s8s")
HEADER_SIZE = 128
assert IDX_STRUCT.size == HEADER_SIZE

PACK_MODES = {"concat": 0, "bfd": 1}
SPLITS = {"train": 0, "val": 1}

U2 = np.dtype("<u2")
U4 = np.dtype("<u4")
U8 = np.dtype("<u8")


class FormatError(Exception):
    """Base class: the bytes on disk do not match the format."""


class BadMagic(FormatError):
    pass


class UnsupportedVersion(FormatError):
    pass


class SizeMismatch(FormatError):
    pass


class DtypeRangeError(ValueError):
    """A token id does not fit the target dtype (astype would wrap silently)."""


def dtype_for_vocab(vocab_size: int) -> np.dtype[np.unsignedinteger]:
    return U2 if vocab_size <= 65_536 else U4


def checked_cast(
    ids: NDArray[np.integer], dtype: np.dtype[np.unsignedinteger]
) -> NDArray[np.unsignedinteger]:
    """Cast token ids, refusing values astype would wrap (70000 -> 4464 for <u2)."""
    if ids.size:
        lo, hi = int(ids.min()), int(ids.max())
        info = np.iinfo(dtype)
        if lo < 0 or hi > info.max:
            raise DtypeRangeError(f"token ids span [{lo}, {hi}], outside {dtype.str} range")
    return ids.astype(dtype)


@dataclass(frozen=True)
class IdxHeader:
    itemsize: int
    pack_mode: int
    split: int
    row_len: int
    shard_index: int
    n_rows: int
    n_segments: int
    n_pad_tokens: int
    first_global_row: int
    bin_sha256: bytes
    tokenizer_fp: bytes
    version_major: int = IDX_VERSION_MAJOR
    version_minor: int = IDX_VERSION_MINOR

    def pack(self) -> bytes:
        return IDX_STRUCT.pack(
            IDX_MAGIC,
            self.version_major,
            self.version_minor,
            self.itemsize,
            self.pack_mode,
            self.split,
            0,
            self.row_len,
            self.shard_index,
            self.n_rows,
            self.n_segments,
            self.n_pad_tokens,
            self.first_global_row,
            self.bin_sha256,
            self.tokenizer_fp,
            b"\0" * 8,
        )

    @classmethod
    def unpack(cls, raw: bytes) -> IdxHeader:
        if len(raw) < HEADER_SIZE:
            raise SizeMismatch(f".idx shorter than its {HEADER_SIZE}-byte header")
        (
            magic,
            vmaj,
            vmin,
            itemsize,
            pack_mode,
            split,
            _r,
            row_len,
            shard_index,
            n_rows,
            n_seg,
            n_pad,
            first_row,
            bin_sha,
            tok_fp,
            _r2,
        ) = IDX_STRUCT.unpack(raw[:HEADER_SIZE])
        if magic != IDX_MAGIC:
            raise BadMagic(f"bad .idx magic {magic!r}")
        if vmaj != IDX_VERSION_MAJOR:
            raise UnsupportedVersion(
                f".idx major version {vmaj}, reader supports {IDX_VERSION_MAJOR}"
            )
        if itemsize not in (2, 4) or pack_mode not in (0, 1) or split not in (0, 1):
            raise FormatError(
                f"bad header enum: itemsize={itemsize} mode={pack_mode} split={split}"
            )
        return cls(
            itemsize,
            pack_mode,
            split,
            row_len,
            shard_index,
            n_rows,
            n_seg,
            n_pad,
            first_row,
            bin_sha,
            tok_fp,
            vmaj,
            vmin,
        )


@dataclass(frozen=True)
class Segments:
    row_seg_ptr: NDArray[np.uint64]
    doc_uid: NDArray[np.uint64]
    start: NDArray[np.uint32]
    length: NDArray[np.uint32]
    piece: NDArray[np.uint32]

    @property
    def n_rows(self) -> int:
        return int(self.row_seg_ptr.shape[0]) - 1

    @property
    def n_segments(self) -> int:
        return int(self.doc_uid.shape[0])


def _counts(row_seg_ptr: NDArray[np.uint64]) -> NDArray[np.int64]:
    """Segments per row, from the CSR pointer."""
    ptr = row_seg_ptr.astype(np.int64)
    return ptr[1:] - ptr[:-1]


def idx_size(n_rows: int, n_segments: int) -> int:
    return HEADER_SIZE + 8 * (n_rows + 1) + 20 * n_segments


def encode_idx(header: IdxHeader, seg: Segments) -> bytes:
    if seg.n_rows != header.n_rows or seg.n_segments != header.n_segments:
        raise ValueError("header counts disagree with the segment arrays")
    return b"".join(
        [
            header.pack(),
            seg.row_seg_ptr.astype(U8).tobytes(),
            seg.doc_uid.astype(U8).tobytes(),
            seg.start.astype(U4).tobytes(),
            seg.length.astype(U4).tobytes(),
            seg.piece.astype(U4).tobytes(),
        ]
    )


def decode_idx(raw: bytes) -> tuple[IdxHeader, Segments]:
    header = IdxHeader.unpack(raw)
    want = idx_size(header.n_rows, header.n_segments)
    if len(raw) != want:
        raise SizeMismatch(f".idx is {len(raw)} bytes, header implies {want}")
    r, s = header.n_rows, header.n_segments
    off = HEADER_SIZE

    def take(dtype: np.dtype[np.unsignedinteger], count: int) -> NDArray[np.unsignedinteger]:
        nonlocal off
        arr = np.frombuffer(raw, dtype=dtype, count=count, offset=off)
        off += dtype.itemsize * count
        return arr

    seg = Segments(take(U8, r + 1), take(U8, s), take(U4, s), take(U4, s), take(U4, s))
    return header, seg


def read_idx(path: Path | str) -> tuple[IdxHeader, Segments]:
    return decode_idx(Path(path).read_bytes())


def read_idx_header(path: Path | str) -> IdxHeader:
    with open(path, "rb") as f:
        return IdxHeader.unpack(f.read(HEADER_SIZE))


def open_bin(
    path: Path | str, itemsize: int, n_rows: int, row_len: int
) -> NDArray[np.unsignedinteger]:
    dtype = U2 if itemsize == 2 else U4
    if n_rows == 0:
        return np.zeros((0, row_len), dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=(n_rows, row_len))


def check_segments(
    seg: Segments,
    *,
    row_len: int,
    pack_mode: int,
    allow_partial_last_row: bool = False,
) -> list[str]:
    """Structural invariants of SPEC 1.4 that need only the .idx arrays."""
    errors: list[str] = []
    ptr = seg.row_seg_ptr.astype(np.int64)
    n_rows, n_seg = seg.n_rows, seg.n_segments
    if ptr[0] != 0 or ptr[-1] != n_seg:
        return [f"row_seg_ptr must run from 0 to {n_seg}, got {ptr[0]}..{ptr[-1]}"]
    counts = ptr[1:] - ptr[:-1]
    if n_rows and counts.min() < 1:
        return ["row_seg_ptr is not strictly increasing (a row without segments)"]
    if n_seg == 0:
        return errors
    start = seg.start.astype(np.int64)
    length = seg.length.astype(np.int64)
    if length.min() < 1:
        errors.append("segment with zero length")
    row_of = np.repeat(np.arange(n_rows), counts)
    first = ptr[:-1]
    if np.any(start[first] != 0):
        errors.append("a row's first segment does not start at position 0")
    contiguous = np.ones(n_seg, dtype=bool)
    contiguous[first] = False
    prev_end = np.concatenate([[0], start[:-1] + length[:-1]])
    if np.any(start[contiguous] != prev_end[contiguous]):
        errors.append("segments within a row are not sorted and contiguous")
    used = np.bincount(row_of, weights=length, minlength=n_rows).astype(np.int64)
    if np.any(used > row_len):
        errors.append("a row holds more than row_len tokens")
    if pack_mode == PACK_MODES["concat"]:
        full = used == row_len
        if allow_partial_last_row and n_rows:
            full[-1] = True
        if not np.all(full):
            errors.append("concat row whose segments do not sum to row_len")
    return errors


def check_tokens(
    tokens: NDArray[np.unsignedinteger],
    seg: Segments,
    *,
    vocab_size: int,
    pad_id: int,
) -> list[str]:
    """Token-level invariants: range, and pad_id after each row's last segment."""
    errors: list[str] = []
    if tokens.size and int(tokens.max()) >= vocab_size:
        errors.append(f"token id {int(tokens.max())} >= vocab_size {vocab_size}")
    counts = _counts(seg.row_seg_ptr)
    row_of = np.repeat(np.arange(seg.n_rows), counts)
    used = np.bincount(row_of, weights=seg.length.astype(np.int64), minlength=seg.n_rows)
    row_len = tokens.shape[1] if tokens.ndim == 2 else 0
    pad_mask = np.arange(row_len)[None, :] >= used.astype(np.int64)[:, None]
    if np.any(tokens[pad_mask] != pad_id):
        errors.append("a padding position holds something other than pad_id")
    return errors
