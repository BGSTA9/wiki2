import pytest

from wiki2shards.io.hashing import (
    canonical_json,
    canonical_json_pretty,
    doc_uid,
    parse_uid_hex,
    uid_hex,
)


def test_canonical_json():
    a = {"b": [1, 2, {"z": None, "a": True}], "a": "é", "c": {"y": 1, "x": 2}}
    b = {"c": {"x": 2, "y": 1}, "a": "é", "b": [1, 2, {"a": True, "z": None}]}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_json(a) == '{"a":"é","b":[1,2,{"a":true,"z":null}],"c":{"x":2,"y":1}}'.encode()
    assert canonical_json_pretty(a).endswith(b"}\n")
    assert canonical_json_pretty(a) == canonical_json_pretty(b)
    for bad in (1.5, {"x": [0.0]}, {"x": float("nan")}, {1: "non-string key"}, {"s": {1, 2}}):
        with pytest.raises(TypeError):
            canonical_json(bad)


def test_doc_uid_golden():
    # Pinned values: a built-in hash() implementation would fail here in-process.
    assert uid_hex(doc_uid("enwiki", 12)) == "6b94dc5bf1a07c42"
    assert uid_hex(doc_uid("simplewiki", 12)) == "0e055fda4b07921c"
    assert parse_uid_hex(uid_hex(2**64 - 1)) == 2**64 - 1
