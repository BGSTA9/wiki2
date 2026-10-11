"""A stand-in trainer for the resume tests (G6). No model: the "model state" is a running hash.

Each step logs {"step", "sha256"} for the batch it consumed. Every --ckpt-every steps it writes a
checkpoint atomically: {"step", "loader_state", "model_state"}, the loader state inside the same
file as the model state, as SPEC 1.5 requires. --kill-at-step N sends SIGKILL to this process
right after step N is logged, while the prefetch thread is still running.

Usage:
  python scripts/trainer_sim.py --mix MIX.yaml --global-batch 16 --steps 300 --prefetch 8 \
      --log run.jsonl [--ckpt-dir DIR --ckpt-every 25] [--resume] [--kill-at-step N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
from pathlib import Path

from wiki2shards.io.atomic import atomic_write_bytes
from wiki2shards.io.hashing import canonical_json
from wiki2shards.loader import Batch, Loader, Mixture


def batch_sha256(b: Batch) -> str:
    h = hashlib.sha256()
    for arr in (b.tokens, b.loss_mask, b.sample_index, b.source_index, b.row_id):
        h.update(arr.tobytes())
    if b.segment_ids is not None:
        h.update(b.segment_ids.tobytes())
    return h.hexdigest()


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    ckpts = sorted(ckpt_dir.glob("ckpt-*.json"))
    return ckpts[-1] if ckpts else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", type=Path, required=True)
    ap.add_argument("--global-batch", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--prefetch", type=int, default=8)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path)
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--kill-at-step", type=int)
    ap.add_argument(
        "--sync-prefetch",
        action="store_true",
        help="wait until the prefetch queue is full before each checkpoint",
    )
    args = ap.parse_args()

    loader = Loader(Mixture.from_yaml(args.mix), args.global_batch, prefetch=args.prefetch)
    step, model = 0, hashlib.sha256(b"init").hexdigest()
    if args.resume and args.ckpt_dir and (ck := latest_checkpoint(args.ckpt_dir)):
        state = json.loads(ck.read_bytes())
        loader.load_state_dict(state["loader_state"])
        step, model = state["step"], state["model_state"]
    if args.ckpt_dir:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(args.log, "a", encoding="utf-8") as log:
        while step < args.steps:
            batch = next(loader)
            if batch.step != step:
                raise SystemExit(f"loader returned step {batch.step}, trainer expected {step}")
            digest = batch_sha256(batch)
            model = hashlib.sha256((model + digest).encode()).hexdigest()
            log.write(json.dumps({"step": step, "sha256": digest}) + "\n")
            log.flush()
            killed_here = args.kill_at_step == step
            step += 1
            if args.ckpt_dir and args.ckpt_every and step % args.ckpt_every == 0:
                if args.sync_prefetch:
                    loader.prefetch_full.wait(timeout=60)
                body = {"step": step, "loader_state": loader.state_dict(), "model_state": model}
                atomic_write_bytes(args.ckpt_dir / f"ckpt-{step:08d}.json", canonical_json(body))
            if killed_here:
                os.kill(os.getpid(), signal.SIGKILL)
    loader.close()


if __name__ == "__main__":
    main()
