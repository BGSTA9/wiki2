import numpy as np

from wiki2shards.loader import Dataset, Loader, Mixture


def test_mixture_end_to_end(fixture_build, fixture_build_b):
    mix = Mixture(
        {
            "a": (Dataset.open(fixture_build.out, "train"), 3),
            "b": (Dataset.open(fixture_build_b.out, "train"), 1),
        },
        seed=1234,
    )
    src = np.empty(10_000, dtype=np.int64)
    with Loader(mix, global_batch_size=100, prefetch=4) as loader:
        for _ in range(100):
            batch = next(loader)
            src[batch.sample_index] = batch.source_index  # from batch metadata only
    D, t = 4, np.arange(10_001)
    for s, w in enumerate((3, 1)):
        c = np.concatenate([[0], np.cumsum(src == s)])
        assert np.max(np.abs(D * c - t * w)) < D  # lag below one sample at every prefix
        assert c[-1] == 10_000 * w // D
