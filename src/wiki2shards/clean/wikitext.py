"""Wikitext -> plain text with an explicit node policy (SPEC 1.3).

Measured on mwparserfromhell 0.7.2: strip_code() alone keeps citation text, table cells,
"thumb|caption" fragments and "Category:Foo". The policy below removes all four.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import mwparserfromhell as mwp
from mwparserfromhell.nodes import Comment, ExternalLink, Heading, Tag, Template, Wikilink
from mwparserfromhell.wikicode import Wikicode

from wiki2shards.clean import filters
from wiki2shards.clean.normalize import normalize
from wiki2shards.config import CleanConfig
from wiki2shards.io.atomic import atomic_write_bytes
from wiki2shards.io.hashing import canonical_json_pretty, doc_uid, text_sha256, uid_hex
from wiki2shards.io.records import CleanedDoc, iter_jsonl, write_jsonl

DROP_TAGS = frozenset(
    {"ref", "references", "table", "gallery", "math", "chem", "score", "timeline", "imagemap"}
)
LINK_PREFIXES = frozenset({"file", "image", "media", "category"})

if not mwp.parser.use_c:  # the pure-Python fallback tokenizer is not covered by the goldens
    raise ImportError("mwparserfromhell C tokenizer unavailable; reinstall mwparserfromhell")


def load_language_codes(path: Path | str) -> frozenset[str]:
    codes = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            codes.add(line.casefold())
    return frozenset(codes)


def _link_prefix(title: str) -> str:
    title = title.strip()
    if title.startswith(":"):
        title = title[1:]
    if ":" not in title:
        return ""
    return title.split(":", 1)[0].replace("_", " ").strip().casefold()


class Policy:
    def __init__(self, drop_sections: Iterable[str], language_codes: frozenset[str]) -> None:
        self.drop_sections = frozenset(s.strip().casefold() for s in drop_sections)
        self.prefixes = LINK_PREFIXES | language_codes

    def drops(self, node: Any) -> bool:
        if isinstance(node, (Template, Comment)):
            return True
        if isinstance(node, Tag):
            return str(node.tag).strip().lower() in DROP_TAGS
        if isinstance(node, ExternalLink):
            return node.title is None  # bare URL or [http://x] without a label
        if isinstance(node, Wikilink):
            return _link_prefix(str(node.title)) in self.prefixes
        return False

    def filter(self, code: Wikicode) -> None:
        kept = []
        for node in code.nodes:
            if self.drops(node):
                continue
            for child in _children(node):
                self.filter(child)
            kept.append(node)
        code.nodes = kept

    def drop_boilerplate_sections(self, code: Wikicode) -> Wikicode:
        """Drop each level-2 section (with its subsections) whose heading is listed."""
        kept, dropping = [], False
        for node in code.nodes:
            if isinstance(node, Heading) and node.level <= 2:
                title = node.title.strip_code().strip().casefold()
                dropping = node.level == 2 and title in self.drop_sections
            if not dropping:
                kept.append(node)
        return Wikicode(kept)


def _children(node: Any) -> list[Wikicode]:
    if isinstance(node, Tag):
        return [node.contents] if node.contents is not None else []
    if isinstance(node, Wikilink):
        return [node.title] + ([node.text] if node.text is not None else [])
    if isinstance(node, ExternalLink):
        return [node.title] if node.title is not None else []
    if isinstance(node, Heading):
        return [node.title]
    return []


def clean_wikitext(wikitext: str, policy: Policy) -> tuple[str, list[str]]:
    """Return (normalized text, raw template names) for one page."""
    code = mwp.parse(wikitext)
    templates = [str(t.name) for t in code.filter_templates(recursive=True)]
    code = policy.drop_boilerplate_sections(code)
    policy.filter(code)
    text = code.strip_code(normalize=True, collapse=True)
    return normalize(text), templates


def clean_page(
    page: dict[str, Any], wiki: str, cfg: CleanConfig, policy: Policy
) -> CleanedDoc | None:
    """One page -> one cleaned document, or None if it is empty after cleaning (hard filter)."""
    text, templates = clean_wikitext(page["wikitext"], policy)
    if not text:
        return None
    sf = cfg.soft_filters
    verdicts: dict[str, bool] = {}
    if sf.min_words is not None:
        verdicts.update([filters.min_words(text, sf.min_words)])
    if sf.max_symbol_pct is not None:
        verdicts.update([filters.max_symbol_pct(text, sf.max_symbol_pct)])
    if sf.disambiguation:
        verdicts.update([filters.disambiguation(templates, cfg.disambiguation_templates)])
    if sf.list_page:
        verdicts.update([filters.list_page(page["title"])])
    return CleanedDoc(
        doc_uid=uid_hex(doc_uid(wiki, page["page_id"])),
        page_id=page["page_id"],
        rev_id=page["rev_id"],
        title=page["title"],
        text=text,
        n_chars=len(text),
        text_sha256=text_sha256(text),
        verdicts=verdicts,
    )


def clean_part(in_path: Path, out_dir: Path, wiki: str, cfg: CleanConfig) -> list[str]:
    """extracted/part-NNNNN.jsonl -> cleaned/part-NNNNN.jsonl + .stats.json."""
    policy = Policy(cfg.drop_sections, load_language_codes(cfg.language_codes_file))
    stem = in_path.name.removesuffix(".jsonl")
    docs, n_in = [], 0
    for page in iter_jsonl(in_path):
        n_in += 1
        doc = clean_page(page, wiki, cfg, policy)
        if doc is not None:
            docs.append(doc)
    write_jsonl(out_dir / f"{stem}.jsonl", docs)
    stats = {"pages": n_in, "docs": len(docs), "dropped_empty": n_in - len(docs)}
    atomic_write_bytes(out_dir / f"{stem}.stats.json", canonical_json_pretty(stats))
    return [f"{stem}.jsonl", f"{stem}.stats.json"]
