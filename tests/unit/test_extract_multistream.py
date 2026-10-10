import bz2
from pathlib import Path

from wiki2shards.extract.multistream import (
    local_name,
    plan_units,
    read_siteinfo,
    read_unit_bytes,
    stream_offsets,
)
from wiki2shards.extract.xmlpages import iter_pages

NS = "{http://www.mediawiki.org/xml/export-0.11/}"


def _paths(dumps: Path)-> tuple[Path, Path]:
    d = dumps / "fixturewiki" / "20260901"
    return (
        d / "fixturewiki-20260901-pages-articles-multistream.xml.bz2",
        d / "fixturewiki-20260901-pages-articles-multistream-index.txt.bz2",
    )


def test_multistream_fragment(dumps):
    xml, index = _paths(dumps)
    offsets = stream_offsets(index)
    size = xml.stat().st_size
    assert len(offsets) == 3  # 300 pages, 100 per stream

    # One stream parses on its own, with un-namespaced tags.
    single = plan_units(offsets, size, 1)
    assert [u.n_streams for u in single] == [1, 1, 1]
    stats: dict[str, int] = {}
    pages = list(iter_pages(read_unit_bytes(xml, single[1]), stats))
    assert stats["pages"] == 100 and len(pages) == stats["kept"]

    # Unit boundaries come from the index alone; one big unit sees the same pages in order.
    whole = plan_units(offsets, size, 1 << 30)
    assert len(whole) == 1 and whole[0].end == size
    all_pages = list(iter_pages(read_unit_bytes(xml, whole[0]), {}))
    by_unit = [p for u in single for p in iter_pages(read_unit_bytes(xml, u), {})]
    assert all_pages == by_unit

    # Header region: siteinfo, no pages. Footer: </mediawiki> stripped from the last unit.
    assert read_siteinfo(xml, offsets[0])["dbname"] == "fixturewiki"
    assert not read_unit_bytes(xml, single[-1]).rstrip().endswith(b"</mediawiki>")

    # Tags match by local name: the same page parsed with its namespace still matches.
    assert local_name(f"{NS}page") == "page" == local_name("page")
    with open(xml, "rb") as f:
        whole_file = bz2.decompress(f.read())
    import xml.etree.ElementTree as ET

    root = ET.fromstring(whole_file)
    assert root.tag == f"{NS}mediawiki"
    assert sum(1 for e in root if local_name(e.tag) == "page") == 300
