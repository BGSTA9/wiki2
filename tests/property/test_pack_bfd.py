from collections import Counter

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from wiki2shards.pack.bfd import bfd_layout


@given(
    lengths=st.lists(st.integers(1, 300), min_size=1, max_size=60),
    row_len=st.integers(2, 129),
    window=st.integers(1, 64),
)
def test_pack_bfd_invariants(lengths: list[int], row_len: int, window: int) -> None:
    lay = bfd_layout(np.array(lengths), row_len, window)
    # every piece placed exactly once; a doc is split only if longer than a row
    placed = Counter(zip(lay.doc.tolist(), lay.piece.tolist(), strict=True))
    expected = Counter((d, p) for d, n in enumerate(lengths) for p in range(-(-n // row_len)))
    assert placed == expected
    for d, n in enumerate(lengths):
        m = lay.doc == d
        assert int(lay.length[m].sum()) == n
        assert sorted(lay.doc_offset[m].tolist()) == list(range(0, n, row_len))
    # no row over capacity; segments start at 0 and are contiguous, so pads sit at row ends
    used = np.bincount(lay.row, weights=lay.length, minlength=lay.n_rows)
    assert used.max() <= row_len and np.all(used > 0)
    for r in range(lay.n_rows):
        m = lay.row == r
        starts, lens = lay.start[m], lay.length[m]
        assert starts[0] == 0 and np.array_equal(starts[1:], (starts + lens)[:-1])
    # rows never span windows
    for r in range(lay.n_rows):
        assert len({int(d) // window for d in lay.doc[lay.row == r]}) == 1
    # deterministic
    again = bfd_layout(np.array(lengths), row_len, window)
    assert all(
        np.array_equal(getattr(lay, f), getattr(again, f))
        for f in ("row", "start", "length", "doc", "piece", "doc_offset")
    )


def test_bfd_best_fit_example() -> None:
    # pieces sorted by length desc: 6, 5, 4, 3, 2 into rows of 8 (ties -> lowest row index)
    lay = bfd_layout(np.array([5, 3, 6, 2, 4]), 8, 100)
    rows = {r: sorted(lay.doc[lay.row == r].tolist()) for r in range(lay.n_rows)}
    assert rows == {0: [2, 3], 1: [0, 1], 2: [4]}
