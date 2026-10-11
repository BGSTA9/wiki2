"""Offline tokenizers for tests: the committed HF fixture and a tiny in-memory tiktoken."""

from __future__ import annotations

import tiktoken

from wiki2shards.tokenize.hf_tok import HFAdapter
from wiki2shards.tokenize.tiktoken_tok import TiktokenAdapter

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def tiny_tiktoken() -> TiktokenAdapter:
    ranks = {bytes([i]): i for i in range(256)}
    for tok in (b"th", b"he", b"in", b"er", b"the", b" t", b" the", b"<|", b"|>"):
        ranks[tok] = len(ranks)
    enc = tiktoken.Encoding(
        name="tiny",
        pat_str=GPT2_PATTERN,
        mergeable_ranks=ranks,
        special_tokens={"<|endoftext|>": len(ranks)},
    )
    return TiktokenAdapter(enc, spec="tiktoken:tiny")


def fixture_hf() -> HFAdapter:
    return HFAdapter.from_file(
        "tests/fixtures/tokenizer.json", spec="hf:tests/fixtures/tokenizer.json"
    )


ADAPTERS = {"hf": fixture_hf, "tiktoken": tiny_tiktoken}
