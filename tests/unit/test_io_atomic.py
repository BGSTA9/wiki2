import subprocess
import sys
from pathlib import Path
import pytest

from wiki2shards.io.atomic import atomic_open, atomic_write_bytes


def test_atomic_write(tmp_path) -> None:
    final = tmp_path / "out.bin"
    with pytest.raises(RuntimeError), atomic_open(final) as f:
        f.write(b"partial")
        raise RuntimeError("boom")
    assert not final.exists()
    assert list(tmp_path.iterdir()) == []  # the tmp file is gone too

    atomic_write_bytes(final, b"v1")
    with pytest.raises(RuntimeError), atomic_open(final) as f:
        f.write(b"v2, half written")
        raise RuntimeError("boom")
    assert final.read_bytes() == b"v1"  # the old content survives a failed rewrite

    # A hard kill mid-write runs no cleanup at all; still nothing appears under the final name.
    killed = tmp_path / "killed.bin"
    code = (
        "import os, sys\nfrom wiki2shards.io.atomic import atomic_open\n"
        "with atomic_open(sys.argv[1]) as f:\n    f.write(b'x' * 4096)\n    f.flush()\n"
        "    os._exit(137)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code, str(killed)])
    assert proc.returncode == 137
    assert not killed.exists()
