"""Dataset statistics and mixture planning (the `stats` and `mix-plan` commands)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from wiki2shards.config import load_mix_config
from wiki2shards.io.manifest import read_manifest
from wiki2shards.loader.mixture import Schedule


def dataset_stats(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    manifest, dataset_id = read_manifest(path)
    stats = json.loads((path / "stats.json").read_bytes())
    return {
        "dataset_id": dataset_id,
        "name": manifest.name,
        "tokenizer": manifest.tokenizer.spec,
        "packing": manifest.packing.model_dump(),
        **stats,
    }


def mixture_plan(mix_path: Path | str, steps: int, global_batch: int) -> dict[str, Any]:
    """Period, worst lag over one period, and per-source draws and epochs at the horizon.

    Weights are per sample; with bfd padding the token mix differs, so both are reported.
    """
    cfg = load_mix_config(mix_path)
    sched = Schedule.build([s.weight for s in cfg.sources])
    D, horizon = sched.D, steps * global_batch
    t = np.arange(D + 1, dtype=np.int64)
    worst = 0.0
    for s, w in enumerate(sched.weights):
        c = np.concatenate([[0], np.cumsum(sched.period == s)]).astype(np.int64)
        worst = max(worst, float(np.max(np.abs(D * c - t * w))) / D)
    draws = sched.draws_before(horizon)
    sources: list[dict[str, Any]] = []
    token_mass: list[float] = []
    for src, w, k in zip(cfg.sources, sched.weights, draws, strict=True):
        manifest, _ = read_manifest(src.path)
        split = manifest.splits[cfg.split]
        n_rows = split.n_rows
        real_per_row = split.n_tokens / n_rows if n_rows else 0.0
        token_mass.append(k * real_per_row)
        sources.append(
            {
                "name": src.name,
                "weight": w,
                "rows": n_rows,
                "draws": k,
                "epochs": round(k / n_rows, 4) if n_rows else None,
                "full_epochs": k // n_rows if n_rows else None,
            }
        )
    total = sum(token_mass) or 1.0
    for info, mass in zip(sources, token_mass, strict=True):
        info["sample_share"] = round(info["draws"] / horizon, 6) if horizon else 0.0
        info["token_share"] = round(mass / total, 6)
    return {
        "period_D": D,
        "worst_lag_samples": round(worst, 6),
        "horizon_samples": horizon,
        "scheduler": "edf-pfair-v1",
        "sources": sources,
    }
