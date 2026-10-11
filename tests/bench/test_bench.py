"""Benchmarks (marker `bench`, never gating). Run: uv run pytest -m bench -s"""

import platform
import time

import numpy as np
import pytest
from helpers.tok import fixture_hf

from wiki2shards.loader import Dataset, Loader, Mixture
from wiki2shards.pack.concat import concat_layout


def _report(name: str, value: float, unit: str) -> None:
    print(
        f"\nBENCH {name}: {value:,.0f} {unit} ({platform.machine()}, {platform.system()},"
        f" Python {platform.python_version()})"
    )


@pytest.mark.bench
def test_bench_tokenize():
    tok = fixture_hf()
    texts = ["The river valley was home to many early settlements and markets. " * 40] * 200
    t = time.perf_counter()
    n = sum(len(ids) for ids in tok.encode_batch(texts))
    _report("tokenize", n / (time.perf_counter() - t), "tokens/s")


@pytest.mark.bench
def test_bench_pack_layout():
    lengths = np.random.default_rng(0).integers(1, 5000, size=200_000)
    t = time.perf_counter()
    lay = concat_layout(lengths, 2049, "drop")
    _report("pack layout", lay.n_rows / (time.perf_counter() - t), "rows/s")


@pytest.mark.bench
def test_bench_loader(fixture_build):
    ds = Dataset.open(fixture_build.out, "train")
    with Loader(Mixture({"a": (ds, 1)}, seed=0), 64, prefetch=4) as loader:
        next(loader)
        waits, t0 = [], time.perf_counter()
        for _ in range(200):
            t = time.perf_counter()
            next(loader)
            waits.append(time.perf_counter() - t)
        _report("loader", 200 * 64 / (time.perf_counter() - t0), "samples/s")
        print(
            f"BENCH loader data-wait p50 {np.percentile(waits, 50) * 1e3:.3f} ms,"
            f" p99 {np.percentile(waits, 99) * 1e3:.3f} ms"
        )
