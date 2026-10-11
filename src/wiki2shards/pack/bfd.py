"""Best-fit-decreasing packing (SPEC 1.3).

Docs are split into pieces of at most row_len, sorted by (length desc, doc, piece). Each piece goes
into the open row with the least remaining room that fits (ties: lowest row index); otherwise it
opens a new row. Rows are padded with pad_id. Work proceeds in windows of `window` docs to bound
memory, and rows never span windows.
"""

from __future__ import annotations

import bisect
import heapq

import numpy as np

from wiki2shards.pack.concat import I64, Ints, Layout, empty_layout


def bfd_layout(lengths: Ints, row_len: int, window: int) -> Layout:
    lengths = np.asarray(lengths, dtype=I64)
    placed: list[tuple[int, int, int, int, int, int]] = []  # row, start, len, doc, piece, offset
    n_rows = 0
    for w0 in range(0, lengths.shape[0], window):
        pieces: list[tuple[int, int, int, int]] = []  # (-len, doc, piece, doc_offset)
        for d in range(w0, min(w0 + window, lengths.shape[0])):
            n = int(lengths[d])
            for p, off in enumerate(range(0, n, row_len)):
                pieces.append((-min(row_len, n - off), d, p, off))
        pieces.sort()
        rooms: list[int] = []  # sorted distinct room values that have an open row
        buckets: dict[int, list[int]] = {}  # room -> min-heap of row ids with that room
        used: dict[int, int] = {}  # row id -> tokens already placed
        for neg_len, d, p, off in pieces:
            size = -neg_len
            i = bisect.bisect_left(rooms, size)  # least room that still fits
            if i < len(rooms):
                room = rooms[i]
                r = heapq.heappop(buckets[room])  # lowest row index among ties
                if not buckets[room]:
                    del buckets[room]
                    rooms.pop(i)
            else:
                r, room = n_rows, row_len
                n_rows += 1
                used[r] = 0
            placed.append((r, used[r], size, d, p, off))
            used[r] += size
            left = room - size
            if left > 0:
                if left not in buckets:
                    buckets[left] = []
                    bisect.insort(rooms, left)
                heapq.heappush(buckets[left], r)
    if not placed:
        return empty_layout(row_len)
    a = np.array(sorted(placed), dtype=I64)
    return Layout(row_len, n_rows, a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5])
