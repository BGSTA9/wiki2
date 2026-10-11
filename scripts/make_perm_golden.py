"""Freeze tests/golden/perm_v1.json from the scalar reference. Run once; never regenerate
unless the permutation's version changes (that is a contract change)."""

import json
from pathlib import Path

from wiki2shards.loader.permutation import permutation

CASES = [
    (1, 0, "a", 0),
    (2, 1, "a", 0),
    (7, 1234, "enwiki", 0),
    (7, 1234, "enwiki", 1),
    (100, 1234, "simplewiki", 3),
    (1000, 2**63 + 5, "x", 0),
    (65_537, 42, "src", 9),
    (10**9, 7, "big", 2),
]

out = []
for n, seed, source, epoch in CASES:
    p = permutation(n, seed, source, epoch)
    idx = sorted({0, n - 1, n // 2, n // 3, (7 * n) // 9} | set(range(min(n, 16))))
    out.append(
        {
            "n": n,
            "seed": seed,
            "source": source,
            "epoch": epoch,
            "index": idx,
            "perm": [p(i) for i in idx],
        }
    )
Path("tests/golden/perm_v1.json").write_text(json.dumps(out, indent=1) + "\n")
print(f"{len(out)} cases written")
