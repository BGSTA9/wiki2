import json

from helpers.build import fixture_config

from wiki2shards.io.hashing import doc_uid, uid_hex
from wiki2shards.pipeline.runner import Build
from wiki2shards.select import is_val, read_selection


def _split_by_page(work):
    sel = read_selection(work / "selection" / "default")
    page_of = {}
    for part in sorted((work / "cleaned").glob("part-*.jsonl")):
        for line in part.read_text().splitlines():
            d = json.loads(line)
            page_of[int(d["doc_uid"], 16)] = d["page_id"]
    return {page_of[int(u)]: split for split in ("train", "val") for u in sel[split]}


def test_split_stable(dumps, tmp_path):
    # Pinned identities: a hash()-based doc_uid fails here even within a single process.
    assert uid_hex(doc_uid("fixturewiki", 10)) == "40d718020f63f3ce"
    assert doc_uid("enwiki", 12) % 10_000 == 7522 and not is_val(doc_uid("enwiki", 12), 50)

    splits = {}
    for date in ("20260901", "20261001"):
        cfg = fixture_config(tmp_path / date, dumps, date=date)
        Build(cfg).run(to_stage="select")
        splits[date] = _split_by_page(tmp_path / date / "work")
    a, b = splits["20260901"], splits["20261001"]
    common = set(a) & set(b)
    assert len(common) > 200 and set(a) != set(b)  # the later dump added and removed pages
    assert all(a[p] == b[p] for p in common)
    assert all(a[p] == ("val" if is_val(doc_uid("fixturewiki", p), 2000) else "train") for p in a)
