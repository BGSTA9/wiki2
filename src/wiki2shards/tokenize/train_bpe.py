"""Train a byte-level BPE tokenizer with HF tokenizers (the `train-tokenizer` command).

The build never trains: it consumes a fingerprinted tokenizer.json. So even if the trainer were
nondeterministic, G1 would be unaffected; the sample itself is deterministic.
"""

from __future__ import annotations

import heapq
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

from wiki2shards.io.atomic import atomic_write_bytes
from wiki2shards.io.hashing import canonical_json_pretty
from wiki2shards.io.records import iter_jsonl
from wiki2shards.tokenize.base import EOS_TEXT, PAD_TEXT
from wiki2shards.tokenize.cache import sample_key


def sample_texts(cleaned_parts: list[Path], n_docs: int, seed: int) -> list[str]:
    """The n_docs docs with the smallest sha256(f"{seed}|{doc_uid}"), in doc_uid order."""
    best: list[tuple[bytes, str, str]] = []  # max-heap via negated key is awkward; use nsmallest
    for part in sorted(cleaned_parts):
        for doc in iter_jsonl(part):
            best.append((sample_key(seed, doc["doc_uid"]), doc["doc_uid"], doc["text"]))
            if len(best) > 4 * n_docs:
                best = heapq.nsmallest(n_docs, best)
    chosen = heapq.nsmallest(n_docs, best)
    return [text for _, _, text in sorted(chosen, key=lambda x: x[1])]


def train_bpe(texts: list[str], vocab_size: int) -> HFTokenizer:
    tok = HFTokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS_TEXT, PAD_TEXT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer, length=len(texts))
    return tok


def train_to_file(
    cleaned_parts: list[Path], out: Path, vocab_size: int, n_docs: int, seed: int
) -> None:
    import json

    tok = train_bpe(sample_texts(cleaned_parts, n_docs, seed), vocab_size)
    # Re-serialize canonically so the committed file is stable byte-for-byte.
    atomic_write_bytes(out, canonical_json_pretty(json.loads(tok.to_str())))
