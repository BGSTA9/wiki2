"""Stage mechanics shared by every stage (SPEC 1.3): stage keys, journal, DONE.json, cleanup.

Journal (PLAN A3, A6): first line {"stage_key": ...}; then one line per committed unit,
{"unit": id, "files": [{"file", "bytes", "sha256"}, ...]}, appended and fsynced after the unit's
files are in place. DONE.json is written last.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from wiki2shards.io.atomic import TMP_SUFFIX, atomic_write_bytes
from wiki2shards.io.hashing import canonical_json, canonical_json_pretty, sha256_file, sha256_hex
from wiki2shards.io.manifest import MANIFEST_NAME

STAGES = ("download", "extract", "clean", "tokenize", "select", "pack")


class StageError(Exception):
    pass


def stage_key(stage: str, version: int, config_section: Any, input_fingerprints: Any) -> str:
    return sha256_hex(
        canonical_json(
            {
                "stage": stage,
                "stage_version": version,
                "config_section": config_section,
                "input_fingerprints": input_fingerprints,
            }
        )
    )


def file_record(out_dir: Path, name: str) -> dict[str, Any]:
    p = out_dir / name
    return {"file": name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def _verifies(out_dir: Path, rec: dict[str, Any]) -> bool:
    p = out_dir / rec["file"]
    return bool(
        p.is_file() and p.stat().st_size == rec["bytes"] and sha256_file(p) == rec["sha256"]
    )


class StageState:
    def __init__(self, workdir: Path, stage: str, out_dir: Path, patterns: Iterable[str]) -> None:
        self.stage = stage
        self.dir = workdir / "_state" / stage
        self.out_dir = out_dir
        self.journal = self.dir / "journal.jsonl"
        self.done = self.dir / "DONE.json"
        self.pattern = re.compile("|".join(f"(?:{p})" for p in patterns))
        self.units: dict[str, list[dict[str, Any]]] = {}

    def is_done(self, key: str) -> str | None:
        """Output fingerprint if DONE.json matches `key` and every journaled file verifies."""
        if not self.done.exists():
            return None
        done = json.loads(self.done.read_bytes())
        if done.get("stage_key") != key:
            return None
        units = self._read_journal(key)
        if units is None or not all(
            _verifies(self.out_dir, r) for rs in units.values() for r in rs
        ):
            return None
        self.units = units
        return str(done["output_fingerprint"])

    def _read_journal(self, key: str) -> dict[str, list[dict[str, Any]]] | None:
        if not self.journal.exists():
            return None
        lines = self.journal.read_bytes().split(b"\n")
        entries: list[dict[str, Any]] = []
        for i, line in enumerate(lines):
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:  # torn final line from a crash mid-append
                    break
                raise StageError(f"{self.journal}: corrupt line {i + 1}") from None
        if not entries or entries[0].get("stage_key") != key:
            return None
        return {e["unit"]: e["files"] for e in entries[1:]}

    def begin(self, key: str, force: bool = False) -> set[str]:
        """Prepare a (re)run; return the ids of units that are already committed and verify."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.done.unlink(missing_ok=True)
        old = None if force else self._read_journal(key)
        self.units = {
            u: recs
            for u, recs in (old or {}).items()
            if all(_verifies(self.out_dir, r) for r in recs)
        }
        lines = [canonical_json({"stage_key": key})]
        lines += [canonical_json({"unit": u, "files": r}) for u, r in sorted(self.units.items())]
        atomic_write_bytes(self.journal, b"".join(line + b"\n" for line in lines))
        self._cleanup()
        return set(self.units)

    def _cleanup(self) -> None:
        keep = {r["file"] for recs in self.units.values() for r in recs}
        names = sorted(os.listdir(self.out_dir))
        foreign = [n for n in names if n not in keep and not self._ours(n)]
        if foreign:
            raise StageError(
                f"{self.out_dir} holds files this stage does not own: {foreign[:5]}; "
                "refusing to delete them (move them or choose another directory)"
            )
        # A stale manifest goes first, so a crash mid-cleanup never leaves a committed-looking dir.
        for n in sorted(names, key=lambda n: n != MANIFEST_NAME):
            if n not in keep and not n.endswith(".part"):
                (self.out_dir / n).unlink()

    def _ours(self, name: str) -> bool:
        return bool(self.pattern.fullmatch(name)) or (
            name.startswith(".") and name.endswith((TMP_SUFFIX, ".part"))
        )

    def commit(self, unit: str, files: list[str]) -> None:
        recs = [file_record(self.out_dir, f) for f in files]
        line = canonical_json({"unit": unit, "files": recs}) + b"\n"
        with open(self.journal, "ab") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self.units[unit] = recs

    def finish(self, key: str) -> str:
        fp = sha256_hex(canonical_json(sorted([u, r] for u, r in self.units.items())))
        atomic_write_bytes(
            self.done,
            canonical_json_pretty(
                {"output_fingerprint": fp, "stage": self.stage, "stage_key": key}
            ),
        )
        return fp
