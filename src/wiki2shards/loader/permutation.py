"""Keyed Feistel permutation of [0, N) with cycle-walking ("feistel6-splitmix64-v1", SPEC 1.5).

O(1) memory per epoch, and the order is part of the data contract: it is defined here, not by
NumPy's Generator.permutation, whose output is tied to NumPy's implementation.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

PERM_ID = "feistel6-splitmix64-v1"
ROUNDS = 6
MASK64 = (1 << 64) - 1
_C1, _C2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def round_keys(seed: int, source: str, epoch: int) -> tuple[int, ...]:
    return tuple(
        int.from_bytes(
            hashlib.sha256(f"w2s-perm-v1|{seed}|{source}|{epoch}|{r}".encode()).digest()[:8],
            "little",
        )
        for r in range(ROUNDS)
    )


def mix64(z: int) -> int:
    """SplitMix64 finalizer, mod 2^64."""
    z ^= z >> 30
    z = (z * _C1) & MASK64
    z ^= z >> 27
    z = (z * _C2) & MASK64
    z ^= z >> 31
    return z


def _mix64_vec(z: NDArray[np.uint64]) -> NDArray[np.uint64]:
    z = z ^ (z >> np.uint64(30))
    z = z * np.uint64(_C1)  # uint64 arrays wrap mod 2^64 without warnings
    z = z ^ (z >> np.uint64(27))
    z = z * np.uint64(_C2)
    return z ^ (z >> np.uint64(31))


class FeistelPermutation:
    def __init__(self, n: int, keys: tuple[int, ...]) -> None:
        if n < 1:
            raise ValueError("permutation domain must be non-empty")
        bits = max(2, (n - 1).bit_length())
        bits += bits & 1  # even, so the two halves are equal
        self.n = n
        self.half = bits // 2
        self.mask = (1 << self.half) - 1
        self.keys = keys

    def _encrypt(self, x: int) -> int:
        left, right = x >> self.half, x & self.mask
        for k in self.keys:
            left, right = right, left ^ (mix64(right ^ k) & self.mask)
        return (left << self.half) | right

    def __call__(self, i: int) -> int:
        """Scalar reference: cycle-walk until the value lands in [0, n)."""
        if not 0 <= i < self.n:
            raise IndexError(i)
        x = self._encrypt(i)
        while x >= self.n:
            x = self._encrypt(x)
        return x

    def _encrypt_vec(self, x: NDArray[np.uint64]) -> NDArray[np.uint64]:
        half, mask = np.uint64(self.half), np.uint64(self.mask)
        left, right = x >> half, x & mask
        for k in self.keys:
            left, right = right, left ^ (_mix64_vec(right ^ np.uint64(k)) & mask)
        return (left << half) | right

    def many(self, idx: NDArray[np.integer]) -> NDArray[np.uint64]:
        """Vectorized; identical to calling the scalar version element by element."""
        x = self._encrypt_vec(np.asarray(idx, dtype=np.uint64))
        n = np.uint64(self.n)
        out = x >= n
        while np.any(out):
            x[out] = self._encrypt_vec(x[out])
            out = x >= n
        return x


def permutation(n: int, seed: int, source: str, epoch: int) -> FeistelPermutation:
    return FeistelPermutation(n, round_keys(seed, source, epoch))
