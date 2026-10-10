"""JSONL records passed between stages: extracted pages and cleaned documents.

Lines are canonical JSON. Readers split on b"\\n" only: raw U+2028 is legal inside a JSON string,
and str.splitlines() would break a record there.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, TypedDict

from wiki2shards.io.atomic import atomic_open
from wiki2shards.io.hashing import canonical_json


class ExtractedPage(TypedDict):
    page_id: int
    rev_id: int
    title: str
    wikitext: str


class CleanedDoc(TypedDict):
    doc_uid: str  # 16 hex digits, see hashing.uid_hex
    page_id: int
    rev_id: int
    title: str
    text: str
    n_chars: int
    text_sha256: str
    verdicts: dict[str, bool]


def write_jsonl(path: Path | str, records: Iterable[Mapping[str, Any]]) -> None:
    with atomic_open(path) as f:
        for rec in records:
            f.write(canonical_json(dict(rec)) + b"\n")


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    data = Path(path).read_bytes()
    for line in data.split(b"\n"):
        if line:
            yield json.loads(line)
