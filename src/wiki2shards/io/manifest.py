"""dataset.manifest: schema, reader, writer (SPEC 1.4). Frozen at contracts-v1.

The manifest is canonical JSON written last; its presence commits the dataset and the SHA-256 of
its bytes is the dataset id (never stored inside it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wiki2shards.io.atomic import atomic_write_bytes
from wiki2shards.io.hashing import canonical_json_pretty, sha256_hex

MANIFEST_NAME = "dataset.manifest"
FORMAT = "wiki2shards"
FORMAT_VERSION = (1, 0)


class ManifestError(Exception):
    pass


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class BuildInfo(_Model):
    wiki2shards_version: str
    git_commit: str
    lockfile_sha256: str | None
    python: str
    unicode_version: str
    config_sha256: str


class SourceFile(_Model):
    name: str
    bytes: int
    sha1: str


class SourceInfo(_Model):
    wiki: str
    dump_date: str
    files: list[SourceFile]


class TokenizerInfo(_Model):
    spec: str
    fingerprint: str
    vocab_size: int
    eos_id: int
    pad_id: int
    dtype: Literal["<u2", "<u4"]


class PackingInfo(_Model):
    mode: Literal["concat", "bfd"]
    seq_len: int
    row_len: int
    doc_order: Literal["doc_uid_asc"]
    last_row: Literal["drop", "pad"]


class SelectionInfo(_Model):
    policy: str
    soft_filters: list[str]
    dedup: str
    val_per_10k: int


class ShardEntry(_Model):
    index: int
    n_rows: int
    bin: str
    bin_bytes: int
    bin_sha256: str
    idx: str
    idx_bytes: int
    idx_sha256: str


class SplitInfo(_Model):
    n_rows: int
    n_tokens: int
    n_pad_tokens: int
    n_docs: int
    n_tokens_dropped_tail: int
    shards: list[ShardEntry]


class SideFile(_Model):
    name: str
    bytes: int
    sha256: str


class Manifest(_Model):
    format: Literal["wiki2shards"]
    format_version: list[int] = Field(min_length=2, max_length=2)
    name: str
    build: BuildInfo
    source: SourceInfo
    tokenizer: TokenizerInfo
    packing: PackingInfo
    selection: SelectionInfo
    splits: dict[Literal["train", "val"], SplitInfo]
    side_files: list[SideFile]

    def to_bytes(self) -> bytes:
        return canonical_json_pretty(self.model_dump(mode="python"))


def _drop(data: Any, loc: tuple[int | str, ...]) -> None:
    for key in loc[:-1]:
        data = data[key]
    del data[loc[-1]]


def parse_manifest(raw: bytes) -> Manifest:
    """Validate manifest bytes. Unknown keys are errors at minor 0 and ignored at higher minors."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"manifest is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ManifestError("not a wiki2shards manifest")
    version = data.get("format_version")
    if not (isinstance(version, list) and len(version) == 2 and version[0] == FORMAT_VERSION[0]):
        raise ManifestError(f"unsupported format_version {version!r}")
    try:
        return Manifest.model_validate(data)
    except ValidationError as exc:
        extras = [e["loc"] for e in exc.errors() if e["type"] == "extra_forbidden"]
        if version[1] > FORMAT_VERSION[1] and len(extras) == len(exc.errors()):
            for loc in sorted(extras, key=len, reverse=True):
                _drop(data, loc)
            return Manifest.model_validate(data)
        raise ManifestError(f"manifest schema error: {exc}") from exc


def read_manifest(dataset_dir: Path | str) -> tuple[Manifest, str]:
    """Return the manifest and the dataset id (sha256 of its bytes)."""
    raw = (Path(dataset_dir) / MANIFEST_NAME).read_bytes()
    manifest = parse_manifest(raw)
    if manifest.format_version == list(FORMAT_VERSION) and manifest.to_bytes() != raw:
        raise ManifestError("manifest bytes are not in canonical form")
    return manifest, sha256_hex(raw)


def write_manifest(dataset_dir: Path | str, manifest: Manifest) -> str:
    raw = manifest.to_bytes()
    atomic_write_bytes(Path(dataset_dir) / MANIFEST_NAME, raw)
    return sha256_hex(raw)
