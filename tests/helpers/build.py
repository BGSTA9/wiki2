"""Helpers to generate fixture dumps and build them into tmp directories."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wiki2shards.config import BuildConfig, load_build_config

REPO = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = REPO / "configs" / "fixture.yaml"


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


def make_dumps(out: Path) -> Path:
    gen = _load_script("make_fixture_dump")
    gen.make_dump(out, "fixturewiki", "20260901", 1)
    gen.make_dump(out, "fixturewiki", "20261001", 2)
    gen.make_dump(out, "fixture2wiki", "20260901", 1)
    return out


def fixture_config(
    root: Path, dumps: Path, wiki: str = "fixturewiki", date: str = "20260901", **changes: Any
) -> BuildConfig:
    data = load_build_config(FIXTURE_CONFIG).model_dump(mode="python")
    data["name"] = f"{wiki}-{date}-fixture-s64"
    data["source"].update(wiki=wiki, date=date, mirror=str(dumps))
    data["workdir"] = str(root / "work")
    data["output"] = str(root / "out")
    for dotted, value in changes.items():
        node = data
        *parents, leaf = dotted.split("__")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return BuildConfig.model_validate(data)


@dataclass
class Built:
    cfg: BuildConfig
    root: Path

    @property
    def out(self) -> Path:
        return Path(self.cfg.output)

    @property
    def work(self) -> Path:
        return Path(self.cfg.workdir)

    def expected(self) -> dict[str, Any]:
        d = Path(self.cfg.source.mirror) / self.cfg.source.wiki / self.cfg.source.date
        return json.loads((d / "fixture_expected.json").read_text())  # type: ignore[no-any-return]


def build_fixture(root: Path, dumps: Path, **kw: Any) -> Built:
    from wiki2shards.pipeline.runner import Build  # Phase 7; imported lazily for early phases

    cfg = fixture_config(root, dumps, **kw)
    Build(cfg).run()
    return Built(cfg, root)


def write_config(cfg: BuildConfig, path: Path) -> Path:
    import yaml

    path.write_text(yaml.safe_dump(cfg.model_dump(mode="python"), sort_keys=True))
    return path


@dataclass
class CliResult:
    returncode: int
    stdout: str
    stderr: str
    pid: int  # with new_session=True this is also the process-group id


def run_cli(
    args: list[str], env: dict[str, str] | None = None, new_session: bool = False
) -> CliResult:
    """Run `python -m wiki2shards ...`. Output goes to files, not pipes: after a crash, orphaned
    pool workers may still hold inherited pipes open, and reading them would block."""
    full_env = {**os.environ, **(env or {})}
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen(
            [sys.executable, "-m", "wiki2shards", *args],
            cwd=REPO,
            env=full_env,
            stdout=out,
            stderr=err,
            start_new_session=new_session,
        )
        code = proc.wait(timeout=300)
        out.seek(0)
        err.seek(0)
        return CliResult(code, out.read(), err.read(), proc.pid)


def digests(root: Path, exclude: tuple[str, ...] = ("_state",)) -> dict[str, str]:
    """sha256 of every file under a final name below root (relative path -> digest).

    Skips the stage journals (_state) and hidden files (.<name>.tmp and .<name>.part are
    in-progress writes, never final names)."""
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if (
            p.is_file()
            and not any(part in exclude for part in rel.parts)
            and not p.name.startswith(".")
        ):
            out[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def build_digests(built_root: Path) -> dict[str, str]:
    """Digests of a build's stage outputs (work/) and dataset (out/)."""
    return {f"{k}/{rel}": d for k in ("work", "out") for rel, d in digests(built_root / k).items()}


def write_mix(
    path: Path, sources: list[tuple[str, Path, int]], seed: int = 1234, split: str = "train"
) -> Path:
    lines = [f"seed: {seed}", f"split: {split}", "sources:"]
    lines += [f"  - {{name: {n}, path: '{p}', weight: {w}}}" for n, p, w in sources]
    path.write_text("\n".join(lines) + "\n")
    return path
