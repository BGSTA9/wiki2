"""Stable hashing, canonical JSON and document identity.
Nothing here uses the built-in hash function: str hashing is salted per process (PYTHONHASHSEED),
so any value derived from it would differ between runs. Everything persisted goes through sha256.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha1_file(path: Path | str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _check(obj: Any, where: str = "$") -> None:
    """Reject anything whose JSON encoding is not unique: floats, NaN, non-string keys."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return
    if isinstance(obj, float):
        raise TypeError(f"float at {where}: canonical JSON is integers only")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"non-string key {k!r} at {where}")
            _check(v, f"{where}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check(v, f"{where}[{i}]")
        return
    raise TypeError(f"unsupported type {type(obj).__name__} at {where}")


def canonical_json(obj: Any) -> bytes:
    """Compact form: stage keys, JSONL lines, journals."""
    _check(obj)
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return text.encode("utf-8")


def canonical_json_pretty(obj: Any) -> bytes:
    """Manifest form: sorted keys, indent 2, UTF-8, trailing newline."""
    _check(obj)
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2, separators=(",", ": "))
    return (text + "\n").encode("utf-8")


def doc_uid(wiki: str, page_id: int) -> int:
    """Little-endian u64 of the first 8 bytes of sha256(f"{wiki}:{page_id}")."""
    digest = hashlib.sha256(f"{wiki}:{page_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def uid_hex(uid: int) -> str:
    """JSON form of a doc_uid: hex of the integer, so string order equals numeric order."""
    return f"{uid:016x}"


def parse_uid_hex(text: str) -> int:
    if len(text) != 16:
        raise ValueError(f"doc_uid must be 16 hex digits, got {text!r}")
    return int(text, 16)


def text_sha256(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))
