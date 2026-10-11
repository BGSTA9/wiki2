import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from wiki2shards.loader.mixture import Schedule, reduce_weights, simulate_edf

WEIGHTS = st.lists(st.integers(1, 100), min_size=1, max_size=8)
EXPLICIT = [1, 3, 1, 1, 22, 22, 5, 1, 3, 3]  # largest-deficit-first lags 1.19 here


def max_lag(seq, weights):
    D, t = sum(weights), np.arange(len(seq) + 1)
    seq = np.asarray(seq)
    worst = 0
    for s, w in enumerate(weights):
        c = np.concatenate([[0], np.cumsum(seq == s)])
        worst = max(worst, int(np.max(np.abs(D * c - t * w))))
    return worst / D  # in samples


def _check(weights):
    w = reduce_weights(weights)
    D = sum(w)
    seq = simulate_edf(w, 2 * D)
    assert max_lag(seq, w) < 1  # lag below one sample at every prefix
    assert [seq[:D].count(s) for s in range(len(w))] == w  # exact at D
    assert seq[:D] == seq[D:]  # periodic over 2D
    assert Schedule.build(weights).period.tolist() == seq[:D]


@given(WEIGHTS)
def test_mixture_lag_exhaustive(weights):
    _check(weights)
    _check(EXPLICIT)


@given(WEIGHTS)
def test_mixture_random_access(weights):
    sched = Schedule.build(weights)
    D = sched.D
    seq = simulate_edf(list(sched.weights), 3 * D)
    s, k = sched.draw(np.arange(3 * D))
    assert s.tolist() == seq
    for src in range(len(sched.weights)):  # k counts this source's earlier draws
        assert k[s == src].tolist() == list(range(int(np.sum(s == src))))


def test_iid_sampler_drifts():
    """Why EDF exists: i.i.d. sampling at w=0.3 drifts by tens of samples; EDF never by one."""
    rng = np.random.Generator(np.random.PCG64(1234))
    draws = rng.random(100_000) < 0.3
    c = np.cumsum(draws)
    t = np.arange(1, draws.size + 1)
    assert np.max(np.abs(c - 0.3 * t)) > 20
    assert max_lag(simulate_edf([3, 7], 100_000), [3, 7]) < 1
