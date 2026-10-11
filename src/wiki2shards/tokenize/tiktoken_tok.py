"""tiktoken adapter.

Measured: encode() raises on a literal "<|endoftext|>" by default, and allowed_special="all" turns
it into a real EOS. encode_ordinary_batch treats it as plain text, which is what we want.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tiktoken
from numpy.typing import NDArray

from wiki2shards.tokenize.base import fingerprint_tiktoken


class TiktokenAdapter:
    def __init__(self, enc: tiktoken.Encoding, *, spec: str) -> None:
        self._enc = enc
        self._spec = spec
        self._fp = fingerprint_tiktoken(
            enc.name, enc._pat_str, enc._special_tokens, enc._mergeable_ranks
        )

    @classmethod
    def from_name(cls, name: str) -> TiktokenAdapter:
        return cls(tiktoken.get_encoding(name), spec=f"tiktoken:{name}")

    @property
    def name(self) -> str:
        return self._spec

    @property
    def fingerprint(self) -> str:
        return self._fp

    @property
    def vocab_size(self) -> int:
        return int(self._enc.n_vocab)

    @property
    def eos_id(self) -> int:
        return int(self._enc.eot_token)

    @property
    def pad_id(self) -> int | None:
        return None

    def encode_batch(self, texts: Sequence[str]) -> list[NDArray[np.int64]]:
        return [
            np.asarray(ids, dtype=np.int64) for ids in self._enc.encode_ordinary_batch(list(texts))
        ]

    def decode(self, ids: Sequence[int] | NDArray[np.integer]) -> str:
        return self._enc.decode([int(i) for i in ids])
