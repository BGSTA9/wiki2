import hashlib
import json
import signal
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from helpers.build import REPO, write_mix

from wiki2shards.loader import Dataset, Loader, LoaderStateError, Mixture


def _mix(a, b, wa=3, wb=1, seed=1234):
    return Mixture(
        {"a": (Dataset.open(a.out, "train"), wa), "b": (Dataset.open(b.out, "train"), wb)},
        seed=seed,
    )


def _trainer(*args):
    with tempfile.TemporaryFile("w+") as err:  # a file, not a pipe (see helpers.run_cli)
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "trainer_sim.py"), *args],
            stdout=subprocess.DEVNULL,
            stderr=err,
            timeout=300,
        )
        err.seek(0)
        return subprocess.CompletedProcess(proc.args, proc.returncode, "", err.read())


def _log(path):
    return {r["step"]: r["sha256"] for r in map(json.loads, path.read_text().splitlines())}


@pytest.mark.slow
def test_sigkill_resume_bit_identical(fixture_build, fixture_build_b, tmp_path):
    mix = write_mix(
        tmp_path / "mix.yaml", [("a", fixture_build.out, 3), ("b", fixture_build_b.out, 1)]
    )
    common = ["--mix", str(mix), "--global-batch", "16", "--steps", "300", "--prefetch", "8"]
    ref = _trainer(*common, "--log", str(tmp_path / "ref.jsonl"))
    assert ref.returncode == 0, ref.stderr
    reference = _log(tmp_path / "ref.jsonl")
    assert sorted(reference) == list(range(300))

    kill_at = 30 + int(hashlib.sha256(b"w2s-kill").hexdigest(), 16) % 240  # seeded, not random
    ckpt = ["--ckpt-dir", str(tmp_path / "ckpt"), "--ckpt-every", "25", "--sync-prefetch"]
    killed = _trainer(
        *common, *ckpt, "--log", str(tmp_path / "killed.jsonl"), "--kill-at-step", str(kill_at)
    )
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    resumed = _trainer(*common, *ckpt, "--log", str(tmp_path / "resumed.jsonl"), "--resume")
    assert resumed.returncode == 0, resumed.stderr

    after = _log(tmp_path / "resumed.jsonl")
    first = min(after)
    assert first == (kill_at + 1) // 25 * 25  # restarted at the last checkpoint
    assert sorted(after) == list(range(first, 300))
    assert all(after[s] == reference[s] for s in after)  # every later batch is bit-identical
    before = _log(tmp_path / "killed.jsonl")
    assert all(before[s] == reference[s] for s in before)


def _sig(batch):
    return {
        (int(t), int(s), int(r), hashlib.sha256(tok.tobytes()).hexdigest())
        for t, s, r, tok in zip(
            batch.sample_index, batch.source_index, batch.row_id, batch.tokens, strict=True
        )
    }


def test_world_size_invariance(fixture_build, fixture_build_b):
    G = 16
    ref = Loader(_mix(fixture_build, fixture_build_b), G, prefetch=0)
    reference = [_sig(ref.make_batch(b * G)) for b in range(100)]

    ranks4 = [
        Loader(_mix(fixture_build, fixture_build_b), G, rank=r, world_size=4, prefetch=2)
        for r in range(4)
    ]
    for step in range(50):
        got = set().union(*(_sig(next(ld)) for ld in ranks4))
        assert got == reference[step]
    states = [ld.state_dict() for ld in ranks4]
    assert all(s == states[0] for s in states)  # the state is rank-independent
    for ld in ranks4:
        ld.close()

    ranks2 = [
        Loader(_mix(fixture_build, fixture_build_b), G, rank=r, world_size=2, prefetch=2)
        for r in range(2)
    ]
    for ld in ranks2:
        ld.load_state_dict(states[0])
    for step in range(50, 100):
        batches = [next(ld) for ld in ranks2]
        assert {b.step for b in batches} == {step}
        assert set().union(*(_sig(b) for b in batches)) == reference[step]
    for ld in ranks2:
        ld.close()


def test_resume_refuses_mismatch(fixture_build, fixture_build_b):
    with Loader(_mix(fixture_build, fixture_build_b), 8, prefetch=0) as ld:
        for _ in range(3):
            next(ld)
        state = ld.state_dict()
    other_weight = Loader(_mix(fixture_build, fixture_build_b, wa=2), 8, prefetch=0)
    with pytest.raises(LoaderStateError):
        other_weight.load_state_dict(state)
    other_seed = Loader(_mix(fixture_build, fixture_build_b, seed=99), 8, prefetch=0)
    with pytest.raises(LoaderStateError):
        other_seed.load_state_dict(state)
    swapped = Mixture(
        {
            "a": (Dataset.open(fixture_build.out, "train"), 3),
            "b": (Dataset.open(fixture_build.out, "val"), 1),
        },
        seed=1234,
    )
    other_data = Loader(swapped, 8, prefetch=0)
    with pytest.raises(LoaderStateError):
        other_data.load_state_dict(state)
    tampered = {**state, "per_source_draws": {"a": 0, "b": 24}}
    with pytest.raises(LoaderStateError):
        Loader(_mix(fixture_build, fixture_build_b), 8, prefetch=0).load_state_dict(tampered)
    other_data.load_state_dict(state, allow_mismatch=True)  # explicit override works
    assert other_data.samples_consumed == 24
    with pytest.raises(LoaderStateError):
        other_data.load_state_dict({**state, "version": 2}, allow_mismatch=True)


def test_global_batch_change_continues_at_samples_consumed(fixture_build, fixture_build_b):
    ref = Loader(_mix(fixture_build, fixture_build_b), 8, prefetch=0)
    with Loader(_mix(fixture_build, fixture_build_b), 8, prefetch=0) as ld:
        for _ in range(5):
            next(ld)
        state = ld.state_dict()
    bigger = Loader(_mix(fixture_build, fixture_build_b), 12, prefetch=0)
    bigger.load_state_dict(state)
    b = next(bigger)
    assert b.sample_index.tolist() == list(range(40, 52))
    assert np.array_equal(b.tokens[:8], ref.make_batch(40).tokens)
