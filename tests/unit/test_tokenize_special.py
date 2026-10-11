from typing import Any

import numpy as np
import pytest
from helpers.tok import ADAPTERS

from wiki2shards.tokenize.cache import encode_docs, read_token_part, write_token_part

DOCS = ["plain text", "a <|endoftext|> b", "<|endoftext|>", "x<|endoftext|><|endoftext|>y"]


@pytest.mark.parametrize("kind", sorted(ADAPTERS))
def test_special_token_injection(kind: Any, tmp_path: Any) -> None:
    tok: Any = ADAPTERS[kind]()
    encoded = encode_docs(tok, DOCS)
    for text, ids in zip(DOCS, encoded, strict=True):
        assert ids[-1] == tok.eos_id  # EOS appended
        assert tok.eos_id not in ids[:-1].tolist(), text  # never parsed from the text
        assert tok.decode(ids[:-1]) == text
    write_token_part(
        tmp_path / "p.tok.bin", tmp_path / "p.tok.idx", [1, 2, 3, 4], encoded, tok.vocab_size
    )
    part = read_token_part(tmp_path / "p.tok.bin", tmp_path / "p.tok.idx")
    eos_at = np.nonzero(np.asarray(part.tokens) == tok.eos_id)[0]
    assert eos_at.tolist() == (part.offsets[1:].astype(np.int64) - 1).tolist()
