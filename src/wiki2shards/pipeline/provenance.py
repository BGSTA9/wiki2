"""Build provenance for the manifest. No timestamps, no hostnames."""

from __future__ import annotations

import platform
import subprocess
import unicodedata
from pathlib import Path

from wiki2shards import __version__
from wiki2shards.io.hashing import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/wiki2shards/pipeline/ -> repo root


def git_commit() -> str:
    """Full HEAD SHA, with -dirty if tracked files changed; "unknown" outside a git checkout."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def lockfile_sha256() -> str | None:
    lock = REPO_ROOT / "uv.lock"
    return sha256_file(lock) if lock.is_file() else None


def build_info(config_sha256: str) -> dict[str, str | None]:
    return {
        "wiki2shards_version": __version__,
        "git_commit": git_commit(),
        "lockfile_sha256": lockfile_sha256(),
        "python": platform.python_version(),
        "unicode_version": unicodedata.unidata_version,
        "config_sha256": config_sha256,
    }
