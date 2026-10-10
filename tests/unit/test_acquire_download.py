import hashlib
import pytest
from pathlib import Path
from typing import Iterator
from typing import Any
from wiki2shards.acquire.download import DownloadError, fetch, parse_sha1sums


class FakeTransport:
    """In-memory server: honors Range only if asked to; fails once mid-file if asked to."""
    calls: list[Any]  # Explicit attribute type declaration (or list[Tuple[...]] if known)

    def __init__(self, data: Any, honor_range: bool = True, fail_after: int | None = None) -> None:
        self.data, self.honor, self.fail_after, self.calls = data, honor_range, fail_after, []

    def open(self, url: str, start: int) -> tuple[Iterator[bytes], bool]:
        self.calls.append(start)
        begin = start if self.honor else 0
        fail_after, self.fail_after = self.fail_after, None

        def chunks() -> Iterator[bytes]:
            for i in range(begin, len(self.data), 10):
                if fail_after is not None and i - begin >= fail_after:
                    raise OSError("connection reset")
                yield self.data[i : i + 10]

        return chunks(), self.honor and start > 0


@pytest.mark.parametrize("honor", [True, False])
def test_fetch_resumes_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, honor: bool
) -> None:
    monkeypatch.setattr("time.sleep", lambda s: None)
    data = bytes(range(256)) * 4
    sha1 = hashlib.sha1(data).hexdigest()
    t = FakeTransport(data, honor_range=honor, fail_after=300)
    fetch("fake://x", tmp_path / "x.bin", t, sha1)
    assert (tmp_path / "x.bin").read_bytes() == data
    assert t.calls[0] == 0 and t.calls[1] > 0  # second attempt asked to resume
    assert not (tmp_path / ".x.bin.part").exists()


def test_fetch_rejects_bad_sha1(tmp_path: Path) -> None:
    with pytest.raises(DownloadError):
        fetch("fake://y", tmp_path / "y.bin", FakeTransport(b"abc"), "0" * 40)
    assert not (tmp_path / "y.bin").exists()


def test_parse_sha1sums() -> None:
    assert parse_sha1sums("ABC  a.bz2\ndef *b.txt\n") == {"a.bz2": "abc", "b.txt": "def"}
