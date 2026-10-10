import json

from wiki2shards.tokenize.base import fingerprint_hf_json, fingerprint_tiktoken


def test_hf_fingerprint_is_canonical():
    a = json.dumps({"b": 1, "a": [1, 2]}).encode()
    b = json.dumps({"a": [1, 2], "b": 1}, indent=4).encode()
    assert fingerprint_hf_json(a) == fingerprint_hf_json(b)


def test_fingerprint_goldens():
    ranks = {bytes([i]): i for i in range(256)} | {b"ab": 256}
    assert fingerprint_tiktoken("tiny", r"\S+|\s+", {"<|endoftext|>": 257}, ranks) == (
        "57fd155206ade5e0dff18a8ccb6e13b457bf88fab53aa5385d196a10d4491fb4"
    )
