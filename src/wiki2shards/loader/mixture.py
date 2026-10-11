"""Mixture schedule ("edf-pfair-v1") and the pure sampler sample_at(t) -> (source, row).

The n-th draw of source s (n = 1, 2, ...) is released at floor((n-1)·D/w_s) and due at
ceil(n·D/w_s). At each slot, among released draws pick the smallest (due, source_index). With
total weight 1 a Pfair schedule exists (Baruah et al., 1996) and EDF meets every window on one
processor, so every source's lag stays strictly inside (-1, +1). After D slots each source has
exactly w_s draws and the windows repeat, so one precomputed period answers every t in O(1).
"""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wiki2shards.loader.permutation import FeistelPermutation, round_keys

SCHEDULER_ID = "edf-pfair-v1"
MAX_PERIOD = 1_000_000


def reduce_weights(weights: Sequence[int]) -> list[int]:
    if not weights or any(int(w) < 1 for w in weights):
        raise ValueError("weights must be positive integers")
    g = math.gcd(*weights)
    return [int(w) // g for w in weights]


def simulate_edf(weights: Sequence[int], n_slots: int) -> list[int]:
    """Sequential reference: the source chosen at each of n_slots slots (weights pre-reduced)."""
    D = sum(weights)
    next_n = [1] * len(weights)  # 1-based index of each source's next draw
    pending = [(0, s) for s in range(len(weights))]  # (release time, source)
    heapq.heapify(pending)
    ready: list[tuple[int, int]] = []  # (due, source)
    out = []
    for t in range(n_slots):
        while pending and pending[0][0] <= t:
            _, s = heapq.heappop(pending)
            n = next_n[s]
            heapq.heappush(ready, (-(-n * D // weights[s]), s))  # ceil(n·D / w)
        if not ready:
            raise AssertionError(f"no released draw at slot {t}: schedule infeasible")
        _, s = heapq.heappop(ready)
        out.append(s)
        next_n[s] += 1
        heapq.heappush(pending, ((next_n[s] - 1) * D // weights[s], s))
    return out


@dataclass(frozen=True)
class Schedule:
    weights: tuple[int, ...]  # reduced
    period: NDArray[np.uint16]  # source index per slot
    rank_in_period: NDArray[np.uint32]  # earlier draws of the same source within the period

    @property
    def D(self) -> int:
        return int(self.period.shape[0])

    @classmethod
    def build(cls, weights: Sequence[int]) -> Schedule:
        w = reduce_weights(weights)
        D = sum(w)
        if D > MAX_PERIOD:
            raise ValueError(f"period D={D} exceeds {MAX_PERIOD}; use smaller integer weights")
        if len(w) > 65_535:
            raise ValueError("too many sources")
        period = np.array(simulate_edf(w, D), dtype=np.uint16)
        rank = np.zeros(D, dtype=np.uint32)
        for s in range(len(w)):
            mask = period == s
            rank[mask] = np.arange(int(mask.sum()), dtype=np.uint32)
        sched = cls(tuple(w), period, rank)
        sched.check_lag()
        return sched

    def check_lag(self) -> None:
        """Exhaustive over one period, O(k·D): fail on any lag of one sample or more."""
        D = self.D
        t = np.arange(D + 1, dtype=np.int64)
        for s, w in enumerate(self.weights):
            c = np.concatenate([[0], np.cumsum(self.period == s)]).astype(np.int64)
            if c[-1] != w or np.any(np.abs(D * c - t * w) >= D):
                raise AssertionError(f"mixture lag >= 1 sample for source {s}")

    def draw(self, t: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """(source, k): k is the 0-based draw index within that source."""
        t = np.asarray(t, dtype=np.int64)
        slot = t % self.D
        s = self.period[slot].astype(np.int64)
        w = np.asarray(self.weights, dtype=np.int64)[s]
        k = (t // self.D) * w + self.rank_in_period[slot].astype(np.int64)
        return s, k

    def draws_before(self, t: int) -> list[int]:
        """Per-source draw counts among slots [0, t)."""
        q, r = divmod(t, self.D)
        part = np.bincount(self.period[:r], minlength=len(self.weights))
        return [q * w + int(part[s]) for s, w in enumerate(self.weights)]


class Sampler:
    """sample_at(t) = f(source names, sizes, weights, seed, t). Nothing else."""

    def __init__(
        self,
        names: Sequence[str],
        sizes: Sequence[int],
        weights: Sequence[int],
        seed: int,
        *,
        cache: int = 64,
    ) -> None:
        if not 0 <= seed < 1 << 64:
            raise ValueError("seed must be in [0, 2^64)")
        if any(n < 1 for n in sizes):
            raise ValueError("every source needs at least one row")
        self.names = list(names)
        self.sizes = [int(n) for n in sizes]
        self.seed = int(seed)
        self.schedule = Schedule.build(weights)
        self._perms: OrderedDict[tuple[int, int], FeistelPermutation] = OrderedDict()
        self._cache = cache

    def _perm(self, s: int, epoch: int) -> FeistelPermutation:
        key = (s, epoch)
        p = self._perms.get(key)
        if p is None:
            p = FeistelPermutation(self.sizes[s], round_keys(self.seed, self.names[s], epoch))
            self._perms[key] = p
            if len(self._perms) > self._cache:
                self._perms.popitem(last=False)
        else:
            self._perms.move_to_end(key)
        return p

    def sample_at(self, t: int) -> tuple[int, int]:
        s, k = self.schedule.draw(np.array([t]))
        si, ki = int(s[0]), int(k[0])
        epoch, pos = divmod(ki, self.sizes[si])
        return si, self._perm(si, epoch)(pos)

    def sample_many(self, t: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        s, k = self.schedule.draw(t)
        rows = np.empty(s.shape[0], dtype=np.int64)
        sizes = np.asarray(self.sizes, dtype=np.int64)
        epoch, pos = k // sizes[s], k % sizes[s]
        for si, ep in sorted(set(zip(s.tolist(), epoch.tolist(), strict=True))):
            m = (s == si) & (epoch == ep)
            rows[m] = self._perm(si, ep).many(pos[m]).astype(np.int64)
        return s, rows
