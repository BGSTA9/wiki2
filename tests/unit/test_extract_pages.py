import json

from wiki2shards.extract.multistream import plan_units, stream_offsets
from wiki2shards.extract.xmlpages import extract_unit
from wiki2shards.io.records import iter_jsonl


def test_page_filters(dumps, tmp_path):
    d = dumps / "fixturewiki" / "20260901"
    xml = d / "fixturewiki-20260901-pages-articles-multistream.xml.bz2"
    units = plan_units(
        stream_offsets(d / "fixturewiki-20260901-pages-articles-multistream-index.txt.bz2"),
        xml.stat().st_size,
        1,
    )
    stats: dict[str, int] = {}
    kept = set()
    for u in units:
        extract_unit(xml, u, tmp_path)
        for k, v in json.loads((tmp_path / f"{u.name}.stats.json").read_text()).items():
            stats[k] = stats.get(k, 0) + v
        kept |= {p["page_id"] for p in iter_jsonl(tmp_path / f"{u.name}.jsonl")}
    expected = json.loads((d / "fixture_expected.json").read_text())
    ids = expected["page_ids"]
    for skipped in ("redirect", "talk", "user", "non_wikitext"):
        assert kept.isdisjoint(ids[skipped]), skipped
    assert set(ids["article"]) <= kept and set(ids["empty"]) <= kept
    assert stats == expected["extract"]
