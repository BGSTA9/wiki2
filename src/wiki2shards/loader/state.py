"""Mixture: datasets + weights + seed, its fingerprint, and the loader_state.json document."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wiki2shards.config import load_mix_config
from wiki2shards.io.hashing import canonical_json, sha256_hex
from wiki2shards.loader.api import STATE_FORMAT, STATE_VERSION, LoaderStateError
from wiki2shards.loader.dataset import Dataset, DatasetError
from wiki2shards.loader.mixture import SCHEDULER_ID, Sampler, reduce_weights
from wiki2shards.loader.permutation import PERM_ID


class Mixture:
    """Mixture({"enwiki": (en, 7), "simplewiki": (sw, 3)}, seed=1234). Source index = order."""

    def __init__(self, sources: Mapping[str, tuple[Dataset, int]], seed: int) -> None:
        if not sources:
            raise ValueError("a mixture needs at least one source")
        self.names = list(sources)
        self.datasets = [sources[n][0] for n in self.names]
        self.weights = reduce_weights([sources[n][1] for n in self.names])
        self.seed = int(seed)
        first = self.datasets[0]
        for name, ds in zip(self.names, self.datasets, strict=True):
            if ds.tokenizer_fp != first.tokenizer_fp or ds.row_len != first.row_len:
                raise DatasetError(
                    f"source {name!r}: tokenizer or row_len differs from "
                    f"{self.names[0]!r}; refusing to mix"
                )
        self.sampler = Sampler(self.names, [len(d) for d in self.datasets], self.weights, seed)

    @classmethod
    def from_yaml(cls, path: Path | str, verify: str = "header") -> Mixture:
        cfg = load_mix_config(path)
        return cls(
            {
                s.name: (
                    Dataset.open(s.path, cfg.split, verify=verify),  # type: ignore[arg-type]
                    s.weight,
                )
                for s in cfg.sources
            },
            seed=cfg.seed,
        )

    @property
    def row_len(self) -> int:
        return self.datasets[0].row_len

    def sample_at(self, t: int) -> tuple[int, int]:
        """(source_index, row_id); pure and O(1)."""
        return self.sampler.sample_at(t)

    @property
    def dataset_ids(self) -> dict[str, str]:
        return {n: d.dataset_id for n, d in zip(self.names, self.datasets, strict=True)}

    @property
    def fingerprint(self) -> str:
        return sha256_hex(
            canonical_json(
                {
                    "seed": self.seed,
                    "sources": [
                        {"name": n, "dataset_id": d.dataset_id, "split": d.split, "weight": w}
                        for n, d, w in zip(self.names, self.datasets, self.weights, strict=True)
                    ],
                    "scheduler": SCHEDULER_ID,
                    "perm": PERM_ID,
                }
            )
        )

    def per_source_draws(self, samples: int) -> dict[str, int]:
        return dict(zip(self.names, self.sampler.schedule.draws_before(samples), strict=True))


def make_state(mix: Mixture, samples_consumed: int, global_batch_size: int) -> dict[str, Any]:
    return {
        "format": STATE_FORMAT,
        "version": STATE_VERSION,
        "samples_consumed": samples_consumed,
        "global_batch_size": global_batch_size,
        "seed": mix.seed,
        "mixture_fingerprint": mix.fingerprint,
        "dataset_ids": mix.dataset_ids,
        "per_source_draws": mix.per_source_draws(samples_consumed),
    }


def check_state(mix: Mixture, state: Mapping[str, Any], allow_mismatch: bool) -> int:
    """Validate a saved state against this mixture; return samples_consumed."""
    if state.get("format") != STATE_FORMAT or state.get("version") != STATE_VERSION:
        raise LoaderStateError(
            f"unsupported loader state {state.get('format')!r} v{state.get('version')!r}"
        )
    samples = int(state["samples_consumed"])
    if not allow_mismatch:
        if state.get("dataset_ids") != mix.dataset_ids:
            raise LoaderStateError("dataset ids differ from the checkpoint")
        if state.get("mixture_fingerprint") != mix.fingerprint:
            raise LoaderStateError("mixture fingerprint differs (weights, seed or sources)")
        if state.get("per_source_draws") != mix.per_source_draws(samples):
            raise LoaderStateError("per_source_draws does not match samples_consumed")
    return samples
