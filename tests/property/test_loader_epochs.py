import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from wiki2shards.loader.mixture import Sampler


@given(
    sizes=st.lists(st.integers(1, 60), min_size=1, max_size=4),
    weights=st.lists(st.integers(1, 9), min_size=4, max_size=4),
    seed=st.integers(0, 2**64 - 1),
)
def test_epoch_coverage(sizes, weights, seed):
    names = [f"s{i}" for i in range(len(sizes))]
    sampler = Sampler(names, sizes, weights[: len(sizes)], seed)
    horizon = 3 * max(sizes) * sum(sampler.schedule.weights)
    src, rows = sampler.sample_many(np.arange(horizon, dtype=np.int64))
    for s, n in enumerate(sizes):
        drawn = rows[src == s]
        for e in range(drawn.size // n):  # every complete epoch is a permutation of the rows
            assert sorted(drawn[e * n : (e + 1) * n].tolist()) == list(range(n))
    # scalar sample_at agrees with the vectorized path
    assert all(sampler.sample_at(t) == (int(src[t]), int(rows[t])) for t in range(0, horizon, 7))
