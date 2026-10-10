"""Build and mixture configs: pydantic models loaded from YAML. Frozen at contracts-v1.

extra="forbid" turns typos into errors; strict=True stops YAML 1.1 coercions (an unquoted
20260901 is an int, an unquoted `no` is False) from being silently accepted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wiki2shards.io.hashing import canonical_json, sha256_file, sha256_hex

MIB = 1 << 20


class ConfigError(Exception):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SourceConfig(_Strict):
    wiki: str = Field(pattern=r"^[a-z0-9_]+$")
    date: str = Field(pattern=r"^\d{8}$")
    # Base URL (https://...), or a local directory laid out as <mirror>/<wiki>/<date>/
    mirror: str = "https://dumps.wikimedia.org"


class ExtractConfig(_Strict):
    unit_mb: int = Field(default=32, ge=1)
    unit_bytes: int | None = Field(default=None, ge=1)  # overrides unit_mb; for tiny fixtures

    @property
    def target_bytes(self) -> int:
        return self.unit_bytes if self.unit_bytes is not None else self.unit_mb * MIB


class SoftFilterConfig(_Strict):
    min_words: int | None = Field(default=50, ge=0)
    max_symbol_pct: int | None = Field(default=30, ge=0, le=100)
    disambiguation: bool = True
    list_page: bool = False

    def enabled(self) -> list[str]:
        on = {
            "min_words": self.min_words is not None,
            "max_symbol_pct": self.max_symbol_pct is not None,
            "disambiguation": self.disambiguation,
            "list_page": self.list_page,
        }
        return sorted(name for name, flag in on.items() if flag)


class CleanConfig(_Strict):
    drop_sections: list[str] = [
        "References",
        "Notes",
        "External links",
        "See also",
        "Further reading",
        "Bibliography",
        "Sources",
    ]
    soft_filters: SoftFilterConfig = SoftFilterConfig()
    disambiguation_templates: list[str] = ["disambiguation", "disambig", "dab", "hndis", "geodis"]
    language_codes_file: str = "configs/language_codes.txt"


class SelectConfig(_Strict):
    policy: str = Field(default="default", pattern=r"^[A-Za-z0-9_-]+$")
    apply: list[str] = ["min_words", "disambiguation"]
    dedup: Literal["exact"] = "exact"
    val_per_10k: int = Field(default=50, ge=0, le=10_000)


class PackConfig(_Strict):
    mode: Literal["concat", "bfd"] = "concat"
    seq_len: int = Field(ge=1)
    last_row: Literal["drop", "pad"] | None = None  # concat default: drop; bfd always pads
    shard_mb: int = Field(default=256, ge=1)
    shard_bytes: int | None = Field(default=None, ge=1)  # overrides shard_mb; for tiny fixtures
    bfd_window: int = Field(default=100_000, ge=1)

    @model_validator(mode="after")
    def _last_row(self) -> PackConfig:
        if self.mode == "bfd" and self.last_row == "drop":
            raise ValueError("last_row: drop applies to concat only; bfd always pads")
        return self

    @property
    def row_len(self) -> int:
        return self.seq_len + 1

    @property
    def effective_last_row(self) -> Literal["drop", "pad"]:
        if self.mode == "bfd":
            return "pad"
        return self.last_row or "drop"

    @property
    def target_shard_bytes(self) -> int:
        return self.shard_bytes if self.shard_bytes is not None else self.shard_mb * MIB


class BuildConfig(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    source: SourceConfig
    workdir: str
    output: str
    workers: int = Field(default=1, ge=1)
    extract: ExtractConfig = ExtractConfig()
    clean: CleanConfig = CleanConfig()
    select: SelectConfig = SelectConfig()
    tokenizer: str = Field(pattern=r"^(tiktoken|hf):.+$")
    pack: PackConfig

    @model_validator(mode="after")
    def _cross_checks(self) -> BuildConfig:
        enabled = set(self.clean.soft_filters.enabled())
        unknown = sorted(set(self.select.apply) - enabled)
        if unknown:
            raise ValueError(f"select.apply names filters that clean does not compute: {unknown}")
        return self

    def hashed_view(self) -> dict[str, Any]:
        """The output-affecting part of the config (PLAN A1).

        Excludes workers, workdir, output and source.mirror, which never change output bytes,
        and replaces file paths with the SHA-256 of their content.
        """
        data = self.model_dump(mode="python")
        for key in ("workers", "workdir", "output"):
            data.pop(key)
        data["source"].pop("mirror")
        data["clean"]["language_codes_file"] = sha256_file(self.clean.language_codes_file)
        if self.tokenizer.startswith("hf:"):
            data["tokenizer"] = "hf:sha256:" + sha256_file(self.tokenizer[3:])
        return data

    def config_sha256(self) -> str:
        return sha256_hex(canonical_json(self.hashed_view()))


class MixSource(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    path: str
    weight: int = Field(ge=1)


class MixConfig(_Strict):
    seed: int = Field(ge=0, lt=1 << 64)
    split: Literal["train", "val"] = "train"
    sources: list[MixSource] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> MixConfig:
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate source names: {names}")
        return self


def _load_yaml(path: Path | str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc


def load_build_config(path: Path | str) -> BuildConfig:
    try:
        return BuildConfig.model_validate(_load_yaml(path))
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise ConfigError(f"{path}: {exc}") from exc


def load_mix_config(path: Path | str) -> MixConfig:
    try:
        return MixConfig.model_validate(_load_yaml(path))
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
