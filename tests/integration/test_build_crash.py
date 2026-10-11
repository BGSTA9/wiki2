import contextlib
import os
import signal

import pytest
from helpers.build import build_digests, fixture_config, run_cli, write_config

FAULT_POINTS = [
    "download:mid_file:1",
    "extract:after_unit_commit:2",
    "clean:before_journal_append:1",
    "tokenize:after_unit_commit:3",
    "select:before_done:1",
    "pack:mid_shard:2",
    "pack:before_manifest:1",
    "pack:after_manifest:1",
]


@pytest.mark.slow
@pytest.mark.parametrize("fault", FAULT_POINTS)
def test_build_crash_resume(fault, fixture_build, dumps, tmp_path):
    reference = build_digests(fixture_build.root)
    cfg_path = write_config(fixture_config(tmp_path, dumps), tmp_path / "c.yaml")

    crashed = run_cli(
        ["build", "--config", str(cfg_path)], env={"W2S_FAULT": fault}, new_session=True
    )
    assert crashed.returncode == 137, crashed.stderr
    with contextlib.suppress(ProcessLookupError):  # reap orphaned pool workers; no sleeps
        os.killpg(crashed.pid, signal.SIGKILL)

    # No partial file under a final name: whatever exists is byte-identical to a clean build.
    for rel, digest in build_digests(tmp_path).items():
        assert reference.get(rel) == digest, f"{rel} is partial or wrong after {fault}"

    rerun = run_cli(["build", "--config", str(cfg_path)])
    assert rerun.returncode == 0, rerun.stderr
    assert build_digests(tmp_path) == reference
