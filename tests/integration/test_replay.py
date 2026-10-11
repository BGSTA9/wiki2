import hashlib
import json

from helpers.build import write_mix
from typer.testing import CliRunner

from wiki2shards.cli import app
from wiki2shards.loader import Loader, Mixture


def test_replay_matches_training(fixture_build, fixture_build_b, tmp_path):
    mix_path = write_mix(
        tmp_path / "mix.yaml", [("a", fixture_build.out, 3), ("b", fixture_build_b.out, 1)]
    )
    G = 12
    with Loader(Mixture.from_yaml(mix_path), G, prefetch=4) as loader:
        trained = [next(loader) for _ in range(20)]
    for b in (0, 7, 19):
        result = CliRunner().invoke(
            app, ["replay", "--mix", str(mix_path), "--global-batch", str(G), "--step", str(b)]
        )
        assert result.exit_code == 0, result.output
        rows = [json.loads(line) for line in result.output.splitlines()]
        batch = trained[b]
        assert [r["step"] for r in rows] == [b] * G
        assert [r["t"] for r in rows] == batch.sample_index.tolist()
        assert [r["row"] for r in rows] == batch.row_id.tolist()
        assert [r["source"] for r in rows] == [("a", "b")[s] for s in batch.source_index]
        assert [r["tokens_sha256"] for r in rows] == [
            hashlib.sha256(row.tobytes()).hexdigest() for row in batch.tokens
        ]
