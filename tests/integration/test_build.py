import json
import os
from pathlib import Path

import pytest
from helpers.build import build_digests, build_fixture, fixture_config, run_cli, write_config

from wiki2shards.verify import verify_dataset

CLOCKSHIFT = Path(__file__).parents[1] / "helpers" / "clockshift"


def test_end_to_end_build(fixture_build):
    rep = verify_dataset(fixture_build.out, full=True, strict=True)
    assert rep.exit_code == 0, rep.errors
    exp = fixture_build.expected()
    stats = json.loads((fixture_build.out / "stats.json").read_text())["selection"]
    assert stats["extract"] == exp["extract"]
    assert stats["clean"]["docs"] == exp["clean"]["docs"]
    assert stats["clean"]["dropped_empty"] == exp["clean"]["dropped_empty"]
    assert stats["failed_filter"] == {
        k: exp["verdicts_false"][k] for k in ("disambiguation", "min_words")
    }
    assert stats["after_filters"] == exp["select"]["after_filters"]
    assert stats["dedup_removed"] == exp["select"]["dedup_removed"]
    assert stats["selected"] == exp["select"]["selected"]
    docs = (fixture_build.out / "docs.jsonl").read_text().splitlines()
    assert len(docs) == exp["select"]["selected"]
    longest = max(json.loads(d)["n_tokens"] for d in docs)
    assert longest > 3 * fixture_build.cfg.pack.row_len  # "a page longer than three rows"


@pytest.mark.slow
def test_build_twice_identical(fixture_build, dumps, tmp_path):
    cfg = fixture_config(tmp_path, dumps)
    pythonpath = os.pathsep.join(filter(None, [str(CLOCKSHIFT), os.environ.get("PYTHONPATH")]))
    env = {"PYTHONPATH": pythonpath, "W2S_TEST_CLOCK_OFFSET_S": str(86_400 + 3_600 + 7)}
    proc = run_cli(["build", "--config", str(write_config(cfg, tmp_path / "c.yaml"))], env=env)
    assert proc.returncode == 0, proc.stderr
    assert build_digests(tmp_path) == build_digests(fixture_build.root)


@pytest.mark.slow
def test_worker_invariance(dumps, tmp_path, monkeypatch):
    one = build_fixture(tmp_path / "w1", dumps, workers=1)
    monkeypatch.setenv("W2S_TEST_COMPLETION_ORDER", "reverse")  # deterministic reordering
    four = build_fixture(tmp_path / "w4", dumps, workers=4)
    assert build_digests(one.root) == build_digests(four.root)


@pytest.mark.slow
def test_hashseed_invariance(dumps, tmp_path):
    roots = []
    for seed in ("1", "2"):
        cfg = fixture_config(tmp_path / seed, dumps)
        proc = run_cli(
            ["build", "--config", str(write_config(cfg, tmp_path / f"{seed}.yaml"))],
            env={"PYTHONHASHSEED": seed},
        )
        assert proc.returncode == 0, proc.stderr
        roots.append(tmp_path / seed)
    assert build_digests(roots[0]) == build_digests(roots[1])
