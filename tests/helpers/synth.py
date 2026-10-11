"""Synthetic datasets written with the M1 writer (io.binfmt + io.manifest) only: no pipeline.

Each row holds one document piece followed by padding, so loss masks are non-trivial.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from wiki2shards.io.atomic import atomic_write_bytes
from wiki2shards.io.binfmt import IdxHeader, Segments, encode_idx
from wiki2shards.io.hashing import sha256_file
from wiki2shards.io.manifest import Manifest, write_manifest

TOKENIZER_FP = "11" * 32
SHA = "00" * 32


def write_synthetic_dataset(
    out: Path, name: str, n_rows: int, row_len: int = 17, rows_per_shard: int = 7, vocab: int = 500
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    )
    shards = []
    for index, r0 in enumerate(range(0, n_rows, rows_per_shard)):
        n = min(rows_per_shard, n_rows - r0)
        filled = rng.integers(row_len // 2, row_len + 1, size=n)
        rows = np.ones((n, row_len), dtype="<u2")  # pad_id 1
        for i, f in enumerate(filled):
            rows[i, :f] = rng.integers(2, vocab, size=f)
            rows[i, f - 1] = 0  # eos_id 0 ends the piece
        (out / f"train-{index:05d}.bin").write_bytes(rows.tobytes())
        seg = Segments(
            np.arange(n + 1, dtype=np.uint64),
            np.arange(r0, r0 + n, dtype=np.uint64),
            np.zeros(n, np.uint32),
            filled.astype(np.uint32),
            np.zeros(n, np.uint32),
        )
        header = IdxHeader(
            2,
            1,
            0,
            row_len,
            index,
            n,
            n,
            int(n * row_len - filled.sum()),
            r0,
            bytes.fromhex(sha256_file(out / f"train-{index:05d}.bin")),
            bytes.fromhex(TOKENIZER_FP),
        )
        atomic_write_bytes(out / f"train-{index:05d}.idx", encode_idx(header, seg))
        shards.append(
            {
                "index": index,
                "n_rows": n,
                "bin": f"train-{index:05d}.bin",
                "bin_bytes": rows.nbytes,
                "bin_sha256": header.bin_sha256.hex(),
                "idx": f"train-{index:05d}.idx",
                "idx_bytes": (out / f"train-{index:05d}.idx").stat().st_size,
                "idx_sha256": sha256_file(out / f"train-{index:05d}.idx"),
                "pad": header.n_pad_tokens,
            }
        )
    pads = sum(s.pop("pad") for s in shards)
    empty = {
        "n_rows": 0,
        "n_tokens": 0,
        "n_pad_tokens": 0,
        "n_docs": 0,
        "n_tokens_dropped_tail": 0,
        "shards": [],
    }
    write_manifest(
        out,
        Manifest.model_validate(
            {
                "format": "wiki2shards",
                "format_version": [1, 0],
                "name": name,
                "build": {
                    "wiki2shards_version": "0.1.0",
                    "git_commit": "synthetic",
                    "lockfile_sha256": None,
                    "python": "3.12",
                    "unicode_version": "15.0.0",
                    "config_sha256": SHA,
                },
                "source": {"wiki": "synthetic", "dump_date": "20260901", "files": []},
                "tokenizer": {
                    "spec": "synthetic",
                    "fingerprint": TOKENIZER_FP,
                    "vocab_size": vocab,
                    "eos_id": 0,
                    "pad_id": 1,
                    "dtype": "<u2",
                },
                "packing": {
                    "mode": "bfd",
                    "seq_len": row_len - 1,
                    "row_len": row_len,
                    "doc_order": "doc_uid_asc",
                    "last_row": "pad",
                },
                "selection": {
                    "policy": "default",
                    "soft_filters": [],
                    "dedup": "exact_text_sha256",
                    "val_per_10k": 0,
                },
                "splits": {
                    "train": {
                        "n_rows": n_rows,
                        "n_tokens": n_rows * row_len - pads,
                        "n_pad_tokens": pads,
                        "n_docs": n_rows,
                        "n_tokens_dropped_tail": 0,
                        "shards": shards,
                    },
                    "val": empty,
                },
                "side_files": [],
            }
        ),
    )
    return out
