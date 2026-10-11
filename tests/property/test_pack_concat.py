from typing import Any, Literal, list

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from wiki2shards.io.binfmt import Segments, check_segments
from wiki2shards.pack.concat import concat_layout

DOCS = st.lists(st.integers(1, 5000), min_size=1, max_size=25)


def materialize(lengths: Any) -> list[np.ndarray]:
    """Doc d's tokens: distinct values per (doc, position) so any misplacement is visible."""
    return [np.arange(n, dtype=np.int64) + 10_000 * d for d, n in enumerate(lengths)]


def fill(layout: Any, docs: Any) -> np.ndarray:
    rows = np.full((layout.n_rows, layout.row_len), -1, dtype=np.int64)
    for r, s, n, d, off in zip(
        layout.row, layout.start, layout.length, layout.doc, layout.doc_offset, strict=True
    ):
        rows[r, s : s + n] = docs[d][off : off + n]
    return rows


@given(lengths=DOCS, row_len=st.integers(2, 4097), last_row=st.sampled_from(["drop", "pad"]))
def test_pack_concat_conservation(
    lengths: list[int], row_len: int, last_row: Literal["drop", "pad"]
) -> None:
    lay = concat_layout(np.array(lengths), row_len, last_row)
    docs = materialize(lengths)
    stream = np.concatenate(docs)
    rows = fill(lay, docs)
    stored = int(lay.length.sum())
    # rows + recorded tail == the docs in pack order
    assert np.array_equal(rows.reshape(-1)[:stored], stream[:stored])
    assert stored + lay.n_tokens_dropped_tail == stream.size
    if last_row == "drop":
        assert lay.n_pad_tokens == 0 and lay.n_tokens_dropped_tail == stream.size % row_len
    else:
        assert lay.n_tokens_dropped_tail == 0 and lay.n_pad_tokens == (-stream.size) % row_len
    # segments tile every row; pieces count up within each doc
    counts = np.bincount(lay.row, minlength=lay.n_rows)
    seg = Segments(
        np.concatenate([[0], np.cumsum(counts)]).astype(np.uint64),
        lay.doc.astype(np.uint64),
        lay.start.astype(np.uint32),
        lay.length.astype(np.uint32),
        lay.piece.astype(np.uint32),
    )
    assert (
        check_segments(seg, row_len=row_len, pack_mode=0, allow_partial_last_row=last_row == "pad")
        == []
    )
    for d in np.unique(lay.doc):
        pieces = lay.piece[lay.doc == d]
        assert pieces.tolist() == list(range(pieces.size))
