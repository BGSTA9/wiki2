import numpy as np
import pytest
from pathlib import Path
from wiki2shards.io.binfmt import DtypeRangeError
from wiki2shards.tokenize.base import load_tokenizer
from wiki2shards.tokenize.cache import read_token_part, write_token_part

FIXTURE_TOKENIZER_FP = "0702a626806e1147757dc1a4c41f3794c09a15083948e19abf6aa09ac94a99e4"


def test_token_cache_refuses_out_of_range(tmp_path: Path) -> None:
    with pytest.raises(DtypeRangeError):  # vocab 50,000 -> <u2; id 70,000 would wrap to 4464
        write_token_part(
            tmp_path / "p.tok.bin",
            tmp_path / "p.tok.idx",
            [1],
            [np.array([5, 70_000], dtype=np.int64)],
            vocab_size=50_000,
        )
    assert list(tmp_path.iterdir()) == []  # refused before writing anything


def test_token_cache_roundtrip(tmp_path: Path) -> None:
    docs = [np.array([1, 2, 0]), np.array([7, 0]), np.array([0])]
    write_token_part(tmp_path / "p.tok.bin", tmp_path / "p.tok.idx", [30, 10, 20], docs, 1000)
    part = read_token_part(tmp_path / "p.tok.bin", tmp_path / "p.tok.idx")
    assert part.doc_uids.tolist() == [30, 10, 20]
    assert part.offsets.tolist() == [0, 3, 5, 6]
    assert part.n_tokens.tolist() == [3, 2, 1]
    assert part.tokens.dtype == np.dtype("<u2") and part.tokens.tolist() == [1, 2, 0, 7, 0, 0]


def test_fixture_tokenizer_fingerprint() -> None:
    tok = load_tokenizer("hf:tests/fixtures/tokenizer.json")
    assert tok.fingerprint == FIXTURE_TOKENIZER_FP
    assert (tok.vocab_size, tok.eos_id, tok.pad_id) == (1000, 0, 1)
