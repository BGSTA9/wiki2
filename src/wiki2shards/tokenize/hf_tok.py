"""HF tokenizers adapter.

Measured (tokenizers 0.23.3): by default "<|endoftext|>" inside text becomes the EOS id. Setting
encode_special_tokens = True makes it encode as ordinary text, so documents cannot inject EOS.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer as HFTokenizer

from wiki2shards.tokenize.base import EOS_TEXT, PAD_TEXT, fingerprint_hf_json


class HFAdapter:
    def __init__(self, tok: HFTokenizer, *, spec: str, fingerprint: str) -> None:
        tok.encode_special_tokens = True
        eos = tok.token_to_id(EOS_TEXT)
        if eos is None:
            raise ValueError(f"{spec}: tokenizer has no {EOS_TEXT} token")
        self._tok = tok
        self._spec = spec
        self._fp = fingerprint
        self._eos = int(eos)
        pad = tok.token_to_id(PAD_TEXT)
        self._pad = None if pad is None else int(pad)

    @classmethod
    def from_file(cls, path: str | Path, *, spec: str | None = None) -> HFAdapter:
        raw = Path(path).read_bytes()
        tok = HFTokenizer.from_str(raw.decode("utf-8"))
        return cls(tok, spec=spec or f"hf:{path}", fingerprint=fingerprint_hf_json(raw))

    @property
    def name(self) -> str:
        return self._spec

    @property
    def fingerprint(self) -> str:
        return self._fp

    @property
    def vocab_size(self) -> int:
        return int(self._tok.get_vocab_size(with_added_tokens=True))

    @property
    def eos_id(self) -> int:
        return self._eos

    @property
    def pad_id(self) -> int | None:
        return self._pad

    def encode_batch(self, texts: Sequence[str]) -> list[NDArray[np.int64]]:
        encodings = self._tok.encode_batch(list(texts), add_special_tokens=False)
        return [np.asarray(e.ids, dtype=np.int64) for e in encodings]

    def decode(self, ids: Sequence[int] | NDArray[np.integer]) -> str:
        return str(self._tok.decode([int(i) for i in ids], skip_special_tokens=False))
