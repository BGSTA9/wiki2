"""Crash-safe writes: tmp -> fsync -> os.replace -> fsync(dir).

A reader can never observe a partially written file under its final name: the bytes land in a
hidden tmp file next to it, and only a complete, fsynced file is renamed into place. os.replace
is atomic on POSIX within one filesystem, which is why the tmp file lives in the same directory.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

TMP_SUFFIX = ".tmp"


def tmp_name(final: Path) -> Path:
    """Hidden sibling used while `final` is being written."""
    return final.with_name(f".{final.name}{TMP_SUFFIX}")


def fsync_dir(path: Path) -> None:
    """Persist a directory entry (the rename) itself."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def atomic_open(final: Path | str) -> Iterator[BinaryIO]:
    """Open a binary writer whose content appears at `final` only if the block succeeds."""
    final = Path(final)
    tmp = tmp_name(final)
    f = open(tmp, "wb")  # noqa: SIM115 - closed explicitly on both paths below
    try:
        yield f
        f.flush()
        os.fsync(f.fileno())
        f.close()
        os.replace(tmp, final)
        fsync_dir(final.parent)
    except BaseException:
        f.close()
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_bytes(final: Path | str, data: bytes) -> None:
    with atomic_open(final) as f:
        f.write(data)


def commit(src: Path, final: Path) -> None:
    """Move an already-complete file (written elsewhere in the same filesystem) into place."""
    with open(src, "rb") as f:
        os.fsync(f.fileno())
    os.replace(src, final)
    fsync_dir(final.parent)
