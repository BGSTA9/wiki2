import json
import os
import subprocess
import sys

from wiki2shards.loader.mixture import Sampler

ARGS = (["enwiki", "simplewiki", "code"], [1_000_003, 50_021, 7], [7, 3, 1], 1234)
T = [0, 1, 2, 9, 10, 11, 12_345, 10**9, 10**12 + 17]
SCRIPT = (
    "import json, sys\nfrom wiki2shards.loader.mixture import Sampler\n"
    f"s = Sampler(*{ARGS!r})\nprint(json.dumps([s.sample_at(t) for t in {T!r}]))\n"
)


def test_sample_at_pure() -> None:
    here = [list(Sampler(*ARGS).sample_at(t)) for t in T]
    for hashseed in ("1", "2", "random"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        out = subprocess.run(
            [sys.executable, "-c", SCRIPT], env=env, check=True, capture_output=True, text=True
        ).stdout
        assert json.loads(out) == here
    assert here[-2][0] in (0, 1, 2)  # O(1) at t = 10^9 as well
