"""Per-part token cache: part-NNNNN.tok.bin (concatenated ids) + part-NNNNN.tok.idx.

.tok.idx: header `<8sHHB3xQ` (magic, major, minor, itemsize, n_docs; 24 bytes), then
doc_uid <u8[n] and token offsets <u8[n+1]. Docs stay in stream order; each ends with EOS.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wiki2shards.io.atomic import atomic_open, atomic_write_bytes
from wiki2shards.io.binfmt import U8, FormatError, checked_cast, dtype_for_vocab
from wiki2shards.io.hashing import parse_uid_hex
from wiki2shards.io.records import iter_jsonl
from wiki2shards.tokenize.base import Tokenizer, load_tokenizer

TOK_MAGIC = b"W2S-TOK\0"
TOK_STRUCT = struct.Struct("<8sHHB3xQ")
assert TOK_STRUCT.size == 24
BATCH_DOCS = 256


@dataclass(frozen=True)
class TokenPart:
    doc_uids: NDArray[np.uint64]
    offsets: NDArray[np.uint64]
    tokens: NDArray[np.unsignedinteger]

    @property
    def n_tokens(self) -> NDArray[np.int64]:
        return self.offsets[1:].astype(np.int64) - self.offsets[:-1].astype(np.int64)


def encode_docs(tok: Tokenizer, texts: list[str]) -> list[NDArray[np.int64]]:
    """Encode documents and append EOS to each. Special tokens are never parsed from text."""
    out = []
    for start in range(0, len(texts), BATCH_DOCS):
        for ids in tok.encode_batch(texts[start : start + BATCH_DOCS]):
            out.append(np.append(ids, np.int64(tok.eos_id)))
    return out


def write_token_part(
    bin_path: Path,
    idx_path: Path,
    doc_uids: list[int],
    docs: list[NDArray[np.int64]],
    vocab_size: int,
) -> None:
    dtype = dtype_for_vocab(vocab_size)
    lengths = np.array([len(d) for d in docs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(U8)
    flat = np.concatenate(docs) if docs else np.zeros(0, dtype=np.int64)
    tokens = checked_cast(flat, dtype)  # dtype range guard, BEFORE anything is written
    if tokens.size and int(tokens.max()) >= vocab_size:
        raise FormatError(f"token id {int(tokens.max())} >= vocab_size {vocab_size}")
    with atomic_open(bin_path) as f:
        f.write(tokens.tobytes())
    header = TOK_STRUCT.pack(TOK_MAGIC, 1, 0, dtype.itemsize, len(docs))
    atomic_write_bytes(
        idx_path, header + np.asarray(doc_uids, dtype=U8).tobytes() + offsets.tobytes()
    )


def read_token_part(bin_path: Path, idx_path: Path) -> TokenPart:
    raw = idx_path.read_bytes()
    magic, major, _minor, itemsize, n_docs = TOK_STRUCT.unpack(raw[: TOK_STRUCT.size])
    if magic != TOK_MAGIC or major != 1 or itemsize not in (2, 4):
        raise FormatError(f"bad token cache header in {idx_path}")
    if len(raw) != TOK_STRUCT.size + 8 * n_docs + 8 * (n_docs + 1):
        raise FormatError(f"token cache index {idx_path} has the wrong size")
    uids = np.frombuffer(raw, U8, n_docs, TOK_STRUCT.size)
    offsets = np.frombuffer(raw, U8, n_docs + 1, TOK_STRUCT.size + 8 * n_docs)
    dtype = np.dtype("<u2") if itemsize == 2 else np.dtype("<u4")
    n = int(offsets[-1])
    tokens = np.memmap(bin_path, dtype=dtype, mode="r") if n else np.zeros(0, dtype=dtype)
    if tokens.shape[0] != n:
        raise FormatError(f"{bin_path} holds {tokens.shape[0]} tokens, index says {n}")
    return TokenPart(uids, offsets, tokens)


def tokenize_part(in_path: Path, out_dir: Path, tokenizer_spec: str) -> list[str]:
    """cleaned/part-NNNNN.jsonl -> tokens/part-NNNNN.tok.{bin,idx}. Depends on this part only."""
    tok = load_tokenizer(tokenizer_spec)
    stem = in_path.name.removesuffix(".jsonl")
    uids, texts = [], []
    for doc in iter_jsonl(in_path):
        uids.append(parse_uid_hex(doc["doc_uid"]))
        texts.append(doc["text"])
    docs = encode_docs(tok, texts)
    write_token_part(
        out_dir / f"{stem}.tok.bin", out_dir / f"{stem}.tok.idx", uids, docs, tok.vocab_size
    )
    return [f"{stem}.tok.bin", f"{stem}.tok.idx"]


def sample_key(seed: int, uid_hex: str) -> bytes:
    return hashlib.sha256(f"{seed}|{uid_hex}".encode()).digest()
