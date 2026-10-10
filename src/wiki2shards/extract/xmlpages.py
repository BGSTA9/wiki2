"""Stream <page> elements out of one unit and keep ns=0, non-redirect, wikitext pages.

Measured: a page parsed from one stream is tagged `page`, but parsed from the whole file it is
`{http://www.mediawiki.org/xml/export-0.11/}page`. Every tag is matched by local name.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from wiki2shards.extract.multistream import Unit, local_name, read_unit_bytes
from wiki2shards.io.atomic import atomic_write_bytes
from wiki2shards.io.hashing import canonical_json_pretty
from wiki2shards.io.records import ExtractedPage, write_jsonl

_FEED = 1 << 20

SKIP_REASONS = ("skipped_model", "skipped_namespace", "skipped_redirect")


def _child(elem: ET.Element, name: str) -> ET.Element | None:
    for c in elem:
        if local_name(c.tag) == name:
            return c
    return None


def _children(elem: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in elem if local_name(c.tag) == name]


def iter_pages(data: bytes, stats: dict[str, int]) -> Iterator[ExtractedPage]:
    """Yield kept pages in stream order; count the rest by reason in `stats`."""
    parser: ET.XMLPullParser[ET.Element] = ET.XMLPullParser(events=("start", "end"))
    parser.feed(b"<root>")
    root: ET.Element | None = None
    pos = 0
    while True:
        chunk = data[pos : pos + _FEED]
        pos += len(chunk)
        parser.feed(chunk if chunk else b"</root>")
        for item in parser.read_events():
            event, elem = item[0], item[-1]  # typeshed: some events are 1-tuples
            if not isinstance(elem, ET.Element):
                continue
            if event == "start":
                if root is None:
                    root = elem
                continue
            if local_name(elem.tag) != "page":
                continue
            stats["pages"] = stats.get("pages", 0) + 1
            page = _page(elem, stats)
            if root is not None:
                root.clear()  # free processed pages; the parser still holds the open one
            if page is not None:
                yield page
        if not chunk:
            parser.close()
            return


def _text(elem: ET.Element | None) -> str:
    # Never write `_child(...) or default`: an Element without children is falsy.
    return "" if elem is None else (elem.text or "")


def _page(elem: ET.Element, stats: dict[str, int]) -> ExtractedPage | None:
    ns = int(_text(_child(elem, "ns")) or "-1")
    revisions = _children(elem, "revision")
    if len(revisions) != 1:
        raise ValueError(f"page has {len(revisions)} revisions; pages-articles has exactly one")
    rev = revisions[0]
    model = _text(_child(rev, "model"))
    reason = None
    if ns != 0:
        reason = "skipped_namespace"
    elif _child(elem, "redirect") is not None:
        reason = "skipped_redirect"
    elif model != "wikitext":
        reason = "skipped_model"
    if reason:
        stats[reason] = stats.get(reason, 0) + 1
        return None
    stats["kept"] = stats.get("kept", 0) + 1
    return ExtractedPage(
        page_id=int(_text(_child(elem, "id"))),
        rev_id=int(_text(_child(rev, "id"))),
        title=_text(_child(elem, "title")),
        wikitext=_text(_child(rev, "text")),  # missing or deleted text -> "" (dropped by clean)
    )


def extract_unit(xml_bz2: Path, unit: Unit, out_dir: Path) -> list[str]:
    """Write part-NNNNN.jsonl and part-NNNNN.stats.json into out_dir; return their names."""
    stats = {"pages": 0, "kept": 0, **dict.fromkeys(SKIP_REASONS, 0)}
    pages = list(iter_pages(read_unit_bytes(xml_bz2, unit), stats))
    write_jsonl(out_dir / f"{unit.name}.jsonl", pages)
    atomic_write_bytes(out_dir / f"{unit.name}.stats.json", canonical_json_pretty(stats))
    return [f"{unit.name}.jsonl", f"{unit.name}.stats.json"]
