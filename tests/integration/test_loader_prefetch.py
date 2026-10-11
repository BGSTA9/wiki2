from pathlib import Path

import numpy as np
from helpers.synth import write_synthetic_dataset

from wiki2shards.loader import Dataset, Loader, Mixture


def _mix(tmp_path: Path) -> Mixture:
    a = Dataset.open(write_synthetic_dataset(tmp_path / "a", "a", n_rows=40), "train")
    b = Dataset.open(write_synthetic_dataset(tmp_path / "b", "b", n_rows=13), "train")
    return Mixture({"a": (a, 3), "b": (b, 1)}, seed=1234)


def test_prefetch_state_is_consumed_not_produced(tmp_path: Path) -> None:
    G = 8
    ref = Loader(_mix(tmp_path), G, prefetch=0, with_segments=True)
    expected = [ref.make_batch(b * G) for b in range(8)]
    with Loader(_mix(tmp_path), G, prefetch=8) as loader:
        for _ in range(5):
            next(loader)
        assert loader.prefetch_full.wait(timeout=30)  # the thread has produced well ahead
        state = loader.state_dict()
        assert state["samples_consumed"] == 5 * G
        assert sum(state["per_source_draws"].values()) == 5 * G
    with Loader(_mix(tmp_path), G, prefetch=3, with_segments=True) as resumed:
        resumed.load_state_dict(state)
        for b in range(5, 8):
            got = next(resumed)
            assert got.step == b
            for field in ("tokens", "loss_mask", "segment_ids", "sample_index", "row_id"):
                assert np.array_equal(getattr(got, field), getattr(expected[b], field)), field


def test_loss_mask_and_segments(tmp_path: Path) -> None:
    b = Loader(_mix(tmp_path), 8, prefetch=0, with_segments=True).make_batch(0)
    assert b.segment_ids is not None
    assert b.loss_mask is not None

    for j in range(8):
        filled = int((b.segment_ids[j] >= 0).sum())
        assert b.tokens[j, filled - 1] == 0  # the piece ends with EOS
        assert b.loss_mask[j].tolist() == [1] * (filled - 1) + [0] * (b.tokens.shape[1] - filled)
