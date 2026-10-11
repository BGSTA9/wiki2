import json
from pathlib import Path
from typing import Any

import pytest

from wiki2shards.io.hashing import canonical_json_pretty
from wiki2shards.io.manifest import (
    Manifest,
    ManifestError,
    parse_manifest,
    read_manifest,
    write_manifest,
)

SHA = "ab" * 32


def sample() -> dict[str, Any]:
    shard = {
        "index": 0,
        "n_rows": 2,
        "bin": "train-00000.bin",
        "bin_bytes": 20,
        "bin_sha256": SHA,
        "idx": "train-00000.idx",
        "idx_bytes": 168,
        "idx_sha256": SHA,
    }
    split = {
        "n_rows": 2,
        "n_tokens": 20,
        "n_pad_tokens": 0,
        "n_docs": 1,
        "n_tokens_dropped_tail": 3,
        "shards": [shard],
    }
    return {
        "format": "wiki2shards",
        "format_version": [1, 0],
        "name": "demo",
        "build": {
            "wiki2shards_version": "0.1.0",
            "git_commit": "unknown",
            "lockfile_sha256": None,
            "python": "3.12.14",
            "unicode_version": "15.0.0",
            "config_sha256": SHA,
        },
        "source": {
            "wiki": "fixturewiki",
            "dump_date": "20260901",
            "files": [{"name": "x.xml.bz2", "bytes": 1, "sha1": "cd" * 20}],
        },
        "tokenizer": {
            "spec": "tiktoken:gpt2",
            "fingerprint": SHA,
            "vocab_size": 50257,
            "eos_id": 50256,
            "pad_id": 50256,
            "dtype": "<u2",
        },
        "packing": {
            "mode": "concat",
            "seq_len": 9,
            "row_len": 10,
            "doc_order": "doc_uid_asc",
            "last_row": "drop",
        },
        "selection": {
            "policy": "default",
            "soft_filters": ["min_words"],
            "dedup": "exact_text_sha256",
            "val_per_10k": 50,
        },
        "splits": {"train": split, "val": {**split, "shards": []}},
        "side_files": [{"name": "docs.jsonl", "bytes": 1, "sha256": SHA}],
    }


def test_manifest_roundtrip_and_versions(tmp_path: Path) -> None:
    m = Manifest.model_validate(sample())
    dataset_id = write_manifest(tmp_path, m)
    m2, id2 = read_manifest(tmp_path)
    assert m2 == m and id2 == dataset_id
    raw = (tmp_path / "dataset.manifest").read_bytes()
    assert raw == canonical_json_pretty(sample()) and raw.endswith(b"}\n")

    data = sample()
    data["format_version"] = [1, 3]
    data["future_field"] = {"x": 1}
    data["build"]["also_new"] = "y"
    assert parse_manifest(json.dumps(data).encode()).name == "demo"  # higher minor: ignored
    data["format_version"] = [1, 0]
    with pytest.raises(ManifestError):
        parse_manifest(json.dumps(data).encode())  # minor 0: unknown keys are errors
    data["format_version"] = [2, 0]
    with pytest.raises(ManifestError):
        parse_manifest(json.dumps(data).encode())
    with pytest.raises(ManifestError):
        parse_manifest(
            json.dumps({**sample(), "build": {**sample()["build"], "python": 3.12}}).encode()
        )

    (tmp_path / "dataset.manifest").write_bytes(json.dumps(sample()).encode())
    with pytest.raises(ManifestError, match="canonical"):
        read_manifest(tmp_path)
