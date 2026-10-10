"""Multistream dump geometry: stream offsets from the index, and units of consecutive streams.

Unit boundaries depend on the index and the target unit size only, never on the worker count,
which is what makes extract output identical for any number of workers (G1).
"""

from __future__ import annotations

import bz2
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Unit:
    index: int
    start: int  # byte offset of the unit's first stream
    end: int  # next unit's first offset, or the file size for the last unit
    n_streams: int

    @property
    def name(self) -> str:
        return f"part-{self.index:05d}"


def stream_offsets(index_bz2: Path | str) -> list[int]:
    """Sorted unique stream starts from `offset:page_id:title` lines."""
    offsets: set[int] = set()
    last = -1
    with bz2.open(index_bz2, "rt", encoding="utf-8") as f:
        for line in f:
            off = int(line.split(":", 1)[0])
            if off != last:  # lines are grouped by stream; skip the set lookup
                offsets.add(off)
                last = off
    return sorted(offsets)


def plan_units(offsets: list[int], file_size: int, target_bytes: int) -> list[Unit]:
    """Greedy: add streams in index order; close the unit once it reaches target_bytes."""
    if not offsets:
        return []
    bounds = [*offsets, file_size]
    units: list[Unit] = []
    start_i = 0
    for i in range(len(offsets)):
        if bounds[i + 1] - bounds[start_i] >= target_bytes or i == len(offsets) - 1:
            units.append(Unit(len(units), bounds[start_i], bounds[i + 1], i + 1 - start_i))
            start_i = i + 1
    # The last unit runs to the end of the file: it includes the </mediawiki> footer stream.
    last = units[-1]
    units[-1] = Unit(last.index, last.start, file_size, last.n_streams)
    return units


def local_name(tag: str) -> str:
    """'{http://www.mediawiki.org/xml/export-0.11/}page' -> 'page' (export versions vary)."""
    return tag.rsplit("}", 1)[-1]


def read_siteinfo(xml_bz2: Path | str, first_offset: int) -> dict[str, Any]:
    """Parse dbname and namespaces from the header region; assert it holds no <page>."""
    with open(xml_bz2, "rb") as f:
        raw = bz2.decompress(f.read(first_offset))
    parser: ET.XMLPullParser[ET.Element] = ET.XMLPullParser(events=("start", "end"))
    parser.feed(raw)
    info: dict[str, Any] = {"dbname": None, "namespaces": {}}
    for item in parser.read_events():
        event, elem = item[0], item[-1]  # typeshed: some events are 1-tuples
        if not isinstance(elem, ET.Element):
            continue
        name = local_name(elem.tag)
        if event == "start" and name == "page":
            raise ValueError("header region before the first indexed offset contains a <page>")
        if event == "end" and name == "dbname":
            info["dbname"] = (elem.text or "").strip()
        elif event == "end" and name == "namespace":
            info["namespaces"][str(int(elem.attrib["key"]))] = elem.text or ""
    if not info["dbname"]:
        raise ValueError("no <dbname> in the dump header")
    return info


def write_siteinfo(xml_bz2: Path, first_offset: int, out_dir: Path, wiki: str) -> list[str]:
    from wiki2shards.io.atomic import atomic_write_bytes
    from wiki2shards.io.hashing import canonical_json_pretty

    info = read_siteinfo(xml_bz2, first_offset)
    if info["dbname"] != wiki:
        raise ValueError(f"dump dbname {info['dbname']!r} does not match source.wiki {wiki!r}")
    atomic_write_bytes(out_dir / "siteinfo.json", canonical_json_pretty(info))
    return ["siteinfo.json"]


def read_unit_bytes(xml_bz2: Path | str, unit: Unit) -> bytes:
    """Decompress [start, end); bz2.decompress handles concatenated streams."""
    with open(xml_bz2, "rb") as f:
        f.seek(unit.start)
        data = bz2.decompress(f.read(unit.end - unit.start))
    stripped = data.rstrip()
    if stripped.endswith(b"</mediawiki>"):
        data = stripped[: -len(b"</mediawiki>")]
    return data
