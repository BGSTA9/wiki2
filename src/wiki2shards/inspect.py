"""Answer "what did the model see": one row's ids, segments and decoded text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wiki2shards.loader.dataset import Dataset, DatasetError
from wiki2shards.tokenize.base import Tokenizer, load_tokenizer


def tokenizer_for(ds: Dataset) -> Tokenizer:
    """The dataset's tokenizer, refusing to decode with one whose fingerprint differs."""
    tok = load_tokenizer(ds.manifest.tokenizer.spec)
    if tok.fingerprint != ds.tokenizer_fp:
        raise DatasetError("tokenizer fingerprint differs from the dataset's; refusing to decode")
    return tok


def inspect_row(path: Path | str, split: str, row: int, decode: bool) -> list[dict[str, Any]]:
    ds = Dataset.open(path, split)
    uids, starts, lengths, pieces = ds.segments(row)
    tokens = ds.row(row)
    out: list[dict[str, Any]] = [{"row": row, "split": split, "tokens": tokens.tolist()}]
    tok = tokenizer_for(ds) if decode else None
    for u, s, n, p in zip(
        uids.tolist(), starts.tolist(), lengths.tolist(), pieces.tolist(), strict=True
    ):
        rec: dict[str, Any] = {"doc_uid": f"{u:016x}", "start": s, "len": n, "piece": p}
        if tok is not None:
            rec["text"] = tok.decode(tokens[s : s + n])
        out.append(rec)
    return out
