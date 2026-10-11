import json

from wiki2shards.io.binfmt import read_idx


def test_no_doc_in_both_splits(fixture_build):
    out = fixture_build.out
    manifest = json.loads((out / "dataset.manifest").read_text())
    uids = {}
    for split, info in manifest["splits"].items():
        uids[split] = set()
        for sh in info["shards"]:
            uids[split] |= set(read_idx(out / sh["idx"])[1].doc_uid.tolist())
    assert uids["train"] and uids["val"]
    assert uids["train"].isdisjoint(uids["val"])
    docs = [json.loads(line) for line in (out / "docs.jsonl").read_text().splitlines()]
    assert len({d["doc_uid"] for d in docs}) == len(docs)
