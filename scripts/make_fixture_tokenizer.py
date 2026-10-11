"""Train the committed test tokenizer (tests/fixtures/tokenizer.json) from the fixture dump.

Offline and deterministic: extract + clean the fixture with the library's own stage functions,
then train a 1,000-token byte-level BPE on every cleaned document.

Usage: uv run python scripts/make_fixture_tokenizer.py --dump data/fixture_dump
shasum -a 256 tests/fixtures/tokenizer.json
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from wiki2shards.clean.wikitext import clean_part
from wiki2shards.config import CleanConfig
from wiki2shards.extract.multistream import plan_units, stream_offsets
from wiki2shards.extract.xmlpages import extract_unit
from wiki2shards.tokenize.train_bpe import train_to_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, default=Path("data/fixture_dump"))
    ap.add_argument("--wiki", default="fixturewiki")
    ap.add_argument("--date", default="20260901")
    ap.add_argument("--out", type=Path, default=Path("tests/fixtures/tokenizer.json"))
    ap.add_argument("--vocab-size", type=int, default=1000)
    args = ap.parse_args()

    d = args.dump / args.wiki / args.date
    stem = f"{args.wiki}-{args.date}-pages-articles-multistream"
    xml, index = d / f"{stem}.xml.bz2", d / f"{stem}-index.txt.bz2"
    with tempfile.TemporaryDirectory() as tmp:
        extracted, cleaned = Path(tmp) / "x", Path(tmp) / "c"
        extracted.mkdir()
        cleaned.mkdir()
        for unit in plan_units(stream_offsets(index), xml.stat().st_size, 1):
            extract_unit(xml, unit, extracted)
            clean_part(extracted / f"{unit.name}.jsonl", cleaned, args.wiki, CleanConfig())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        train_to_file(sorted(cleaned.glob("part-*.jsonl")), args.out, args.vocab_size, 10_000, 0)
    print(args.out)


if __name__ == "__main__":
    main()
