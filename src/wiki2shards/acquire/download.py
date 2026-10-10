"""Fetch a dated multistream dump, its index and sha1sums; verify SHA-1; resume with HTTP Range.

One connection, a descriptive User-Agent, a dated directory (never "latest"). Partial downloads
live in .<name>.part and are renamed into place only after the SHA-1 matches.
"""

from __future__ import annotations

import os
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from wiki2shards import __version__
from wiki2shards.io.atomic import commit
from wiki2shards.io.hashing import sha1_file
from wiki2shards.pipeline import faults

USER_AGENT = f"wiki2shards/{__version__} (+https://github.com/BGSTA9/wiki2shards)"
CHUNK = 1 << 20


class DownloadError(Exception):
    pass


def dump_names(wiki: str, date: str) -> dict[str, str]:
    stem = f"{wiki}-{date}"
    return {
        "sha1sums": f"{stem}-sha1sums.txt",
        "index": f"{stem}-pages-articles-multistream-index.txt.bz2",
        "xml": f"{stem}-pages-articles-multistream.xml.bz2",
    }


def source_url(mirror: str, wiki: str, date: str, name: str) -> str:
    return f"{mirror.rstrip('/')}/{wiki}/{date}/{name}"


class Transport(Protocol):
    def open(self, url: str, start: int) -> tuple[Iterator[bytes], bool]:
        """Return (chunks, range_honored). If range_honored is False, chunks start at byte 0."""
        ...


class HttpTransport:
    def open(self, url: str, start: int) -> tuple[Iterator[bytes], bool]:
        headers = {"User-Agent": USER_AGENT}
        if start:
            headers["Range"] = f"bytes={start}-"
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
        honored = start > 0 and resp.status == 206

        def chunks() -> Iterator[bytes]:
            with resp:
                while chunk := resp.read(CHUNK):
                    yield chunk

        return chunks(), honored


class LocalTransport:
    """A mirror that is a local directory (the fixture). Range is honored by seeking."""

    def open(self, url: str, start: int) -> tuple[Iterator[bytes], bool]:
        def chunks() -> Iterator[bytes]:
            with open(url, "rb") as f:
                f.seek(start)
                while chunk := f.read(CHUNK):
                    yield chunk

        return chunks(), True


def transport_for(mirror: str) -> Transport:
    return HttpTransport() if mirror.startswith(("http://", "https://")) else LocalTransport()


def fetch(
    url: str, dest: Path, transport: Transport, expected_sha1: str | None, retries: int = 3
) -> None:
    part = dest.with_name(f".{dest.name}.part")
    for attempt in range(retries + 1):
        start = part.stat().st_size if part.exists() else 0
        try:
            chunks, honored = transport.open(url, start)
            with open(part, "ab" if honored and start else "wb") as f:
                for i, chunk in enumerate(chunks):
                    f.write(chunk)
                    if i == 0:
                        f.flush()
                        faults.hit("download", "mid_file")
                f.flush()
                os.fsync(f.fileno())
            break
        except OSError as exc:
            if attempt == retries:
                raise DownloadError(f"{url}: {exc}") from exc
            time.sleep(2**attempt)  # production backoff; tests use an in-memory transport
    # Re-hash the whole file after any resume.
    if expected_sha1 is not None:
        got = sha1_file(part)
        if got != expected_sha1:
            part.unlink()
            raise DownloadError(f"{dest.name}: SHA-1 {got} != expected {expected_sha1}")
    commit(part, dest)


def parse_sha1sums(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if line.strip():
            sha1, name = line.split(maxsplit=1)
            out[name.strip().lstrip("*")] = sha1.lower()
    return out
