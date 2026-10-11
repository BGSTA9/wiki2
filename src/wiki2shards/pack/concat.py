"""Row layouts, and concat packing.

A layout is a pure function of (doc lengths in pack order, row_len, mode). That is what lets pack
resume by replay: recompute the layout, keep shards that verify, rewrite the rest.

A layout is one segment per (doc, piece), as parallel int64 arrays sorted by (row, start):
    row, start (position in the row), length, doc (index in pack order), piece, doc_offset
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

I64 = np.int64
Ints = NDArray[np.int64]


@dataclass(frozen=True)
class Layout:
    row_len: int
    n_rows: int
    row: Ints
    start: Ints
    length: Ints
    doc: Ints
    piece: Ints
    doc_offset: Ints
    n_tokens_dropped_tail: int = 0

    @property
    def n_pad_tokens(self) -> int:
        return self.n_rows * self.row_len - int(self.length.sum())

    def rows_slice(self, r0: int, r1: int) -> slice:
        """Segments of rows [r0, r1), as a slice (segments are sorted by row)."""
        a, b = np.searchsorted(self.row, [r0, r1], side="left")
        return slice(int(a), int(b))


def empty_layout(row_len: int, dropped: int = 0) -> Layout:
    z = np.zeros(0, dtype=I64)
    return Layout(row_len, 0, z, z, z, z, z, z, dropped)


def concat_layout(lengths: Ints, row_len: int, last_row: str) -> Layout:
    """Stream docs (each already ending in EOS) into rows of row_len tokens.

    last_row="drop": the final partial row is dropped and counted as n_tokens_dropped_tail.
    last_row="pad": the final partial row is kept and padded (PLAN A4).
    """
    lengths = np.asarray(lengths, dtype=I64)
    if lengths.size and int(lengths.min()) < 1:
        raise ValueError("every document has at least one token (its EOS)")
    total = int(lengths.sum())
    n_full, tail = divmod(total, row_len)
    n_rows = n_full + (1 if tail and last_row == "pad" else 0)
    stored = min(total, n_rows * row_len)
    if stored == 0:
        return empty_layout(row_len, total)
    doc_starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(I64)
    # Segment boundaries: every doc start and every row start inside the stored region.
    bounds = np.union1d(doc_starts, np.arange(0, stored, row_len, dtype=I64))
    bounds = bounds[bounds < stored]
    ends = np.append(bounds[1:], stored)
    doc = np.searchsorted(doc_starts, bounds, side="right") - 1
    first = np.ones(bounds.shape[0], dtype=bool)
    first[1:] = doc[1:] != doc[:-1]
    idx = np.arange(bounds.shape[0])
    piece = idx - np.maximum.accumulate(np.where(first, idx, 0))
    row = bounds // row_len
    return Layout(
        row_len=row_len,
        n_rows=n_rows,
        row=row.astype(I64),
        start=(bounds - row * row_len).astype(I64),
        length=(ends - bounds).astype(I64),
        doc=doc.astype(I64),
        piece=piece.astype(I64),
        doc_offset=(bounds - doc_starts[doc]).astype(I64),
        n_tokens_dropped_tail=total - stored,
    )
