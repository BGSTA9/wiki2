import json
from pathlib import Path

import numpy as np

from wiki2shards.loader.permutation import permutation

KEYS = [(0, "a", 0), (1234, "enwiki", 1), (2**64 - 1, "x", 99)]


def test_feistel_bijection() -> None:
    for n in range(1, 2049):
        for seed, source, epoch in KEYS:
            p = permutation(n, seed, source, epoch)
            v = p.many(np.arange(n))
            assert np.array_equal(np.sort(v), np.arange(n, dtype=np.uint64)), (n, seed)
            # vectorized == scalar: every index for small n, a spread sample for larger n
            check = range(n) if n <= 64 else range(0, n, max(1, n // 31))
            assert all(p(i) == int(v[i]) for i in check), (n, seed)


def test_feistel_golden() -> None:
    cases = json.loads((Path(__file__).parents[1] / "golden/perm_v1.json").read_text())
    for c in cases:
        p = permutation(c["n"], c["seed"], c["source"], c["epoch"])
        assert [p(i) for i in c["index"]] == c["perm"]
        assert p.many(np.array(c["index"])).tolist() == c["perm"]


def test_perm_quality() -> None:
    n = 10**6
    v = permutation(n, 1234, "enwiki", 0).many(np.arange(n)).astype(np.int64)
    i = np.arange(n)
    assert abs(np.corrcoef(i, v)[0, 1]) < 0.01
    assert np.mean(np.abs(np.diff(v)) == 1) < 1e-4
    v1 = permutation(n, 1234, "enwiki", 1).many(np.arange(n)).astype(np.int64)
    assert np.sum(v == v1) < 20  # epochs are independent orders (about 1 match expected)
