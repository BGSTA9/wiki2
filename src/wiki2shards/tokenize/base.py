"""Tokenizer protocol and fingerprints. Frozen at contracts-v1.

Spec strings: "tiktoken:<encoding name>" or "hf:<path to tokenizer.json>" (CWD-relative).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from wiki2shards.io.hashing import canonical_json, sha256_hex

EOS_TEXT = "<|endoftext|>"
PAD_TEXT = "<|pad|>"


@runtime_checkable
class Tokenizer(Protocol):
    @property
    def name(self) -> str: ...  # the spec string; goes to manifest tokenizer.spec

    @property
    def fingerprint(self) -> str: ...  # sha256 hex

    @property
    def vocab_size(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def pad_id(self) -> int | None: ...

    def encode_batch(self, texts: Sequence[str]) -> list[NDArray[np.int64]]:
        """Encode plain text. Special-token strings inside texts are ordinary text."""
        ...

    def decode(self, ids: Sequence[int] | NDArray[np.integer]) -> str: ...


def effective_pad_id(tok: Tokenizer) -> int:
    """GPT-2 has no pad token, so pad = eos; padding is identified by segments, not values."""
    return tok.pad_id if tok.pad_id is not None else tok.eos_id


def fingerprint_hf_json(raw: bytes) -> str:
    """sha256 of the canonical form of tokenizer.json (floats are rejected: BPE files have none)."""
    return sha256_hex(canonical_json(json.loads(raw)))


def fingerprint_tiktoken(
    name: str,
    pat_str: str,
    special_tokens: Mapping[str, int],
    mergeable_ranks: Mapping[bytes, int],
) -> str:
    """sha256 over a canonical header, then (<u4 rank, <u4 len, bytes) for each rank in order."""
    h = hashlib.sha256()
    header = {"name": name, "pat_str": pat_str, "special_tokens": dict(special_tokens)}
    h.update(canonical_json(header) + b"\n")
    for token, rank in sorted(mergeable_ranks.items(), key=lambda kv: kv[1]):
        h.update(struct.pack("<II", rank, len(token)))
        h.update(token)
    return h.hexdigest()


_CACHE: dict[str, Tokenizer] = {}


def load_tokenizer(spec: str) -> Tokenizer:
    """Construct (once per process) the tokenizer named by a spec string.

    Adapters are imported by name, so this contract module does not depend on them (they are
    written later, in tokenize/hf_tok.py and tokenize/tiktoken_tok.py).
    """
    if spec in _CACHE:
        return _CACHE[spec]
    kind, _, arg = spec.partition(":")
    tok: Tokenizer
    if kind == "tiktoken":
        tok = importlib.import_module(
            "wiki2shards.tokenize.tiktoken_tok"
        ).TiktokenAdapter.from_name(arg)
    elif kind == "hf":
        tok = importlib.import_module("wiki2shards.tokenize.hf_tok").HFAdapter.from_file(
            arg, spec=spec
        )
    else:
        raise ValueError(f"unknown tokenizer spec {spec!r}")
    _CACHE[spec] = tok
    return tok
