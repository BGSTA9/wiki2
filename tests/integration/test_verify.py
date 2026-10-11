import json
import shutil

import pytest
from typer.testing import CliRunner

from wiki2shards.cli import app
from wiki2shards.io.manifest import read_manifest


def _flip(path, offset):
    data = bytearray(path.read_bytes())
    data[offset] ^= 0xFF
    path.write_bytes(bytes(data))


def _edit_manifest(d):
    m, _ = read_manifest(d)
    data = json.loads(m.to_bytes())
    data["splits"]["train"]["shards"][0]["bin_sha256"] = "0" * 64
    from wiki2shards.io.hashing import canonical_json_pretty

    (d / "dataset.manifest").write_bytes(canonical_json_pretty(data))  # still canonical


CASES = {
    "flip_bin_byte": (lambda d: _flip(d / "train-00000.bin", 777), ["--full"], 1),
    "flip_idx_array_byte": (lambda d: _flip(d / "train-00000.idx", 200), ["--full"], 1),
    "truncate_bin": (
        lambda d: (d / "train-00001.bin").write_bytes((d / "train-00001.bin").read_bytes()[:-2]),
        [],
        1,
    ),
    "delete_shard": (lambda d: (d / "val-00000.bin").unlink(), [], 1),
    "extra_file_strict": (lambda d: (d / "notes.txt").write_text("x"), ["--strict"], 1),
    "extra_file_lenient": (lambda d: (d / "notes.txt").write_text("x"), [], 0),
    "edited_manifest": (_edit_manifest, [], 1),
    "noncanonical_manifest": (
        lambda d: (d / "dataset.manifest").write_text(
            json.dumps(json.loads((d / "dataset.manifest").read_text()))
        ),
        [],
        1,
    ),
    "garbage_manifest": (lambda d: (d / "dataset.manifest").write_bytes(b"\xff{not json"), [], 3),
    "missing_manifest": (lambda d: (d / "dataset.manifest").unlink(), [], 3),
    "untouched": (lambda d: None, ["--full", "--strict"], 0),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_verify_detects_corruption(case, fixture_build, tmp_path):
    corrupt, flags, want = CASES[case]
    d = tmp_path / "ds"
    shutil.copytree(fixture_build.out, d)
    corrupt(d)
    result = CliRunner().invoke(app, ["verify", str(d), *flags])
    assert result.exit_code == want, result.output
