"""Loader: batches for one rank, with a background prefetch thread, and exact resume (G5, G6).

Global batch b holds samples t in [b·G, (b+1)·G); rank r of W reads the contiguous block
[b·G + r·G/W, b·G + (r+1)·G/W). samples_consumed advances only when __next__ hands a batch to the
caller, never when the prefetch thread produces one.

Batches always start at t = samples_consumed and step = t_start // G. With a constant G that is
exactly b; after a G change on resume (PLAN A10) steps stay unique and increasing.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np

from wiki2shards.loader.api import Batch
from wiki2shards.loader.state import Mixture, check_state, make_state

log = logging.getLogger(__name__)
_STOP = object()


class Loader:
    def __init__(
        self,
        mixture: Mixture,
        global_batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        prefetch: int = 2,
        with_segments: bool = False,
    ) -> None:
        if global_batch_size % world_size:
            raise ValueError("global_batch_size must be divisible by world_size")
        if not 0 <= rank < world_size:
            raise ValueError("rank out of range")
        self.mix = mixture
        self.G = global_batch_size
        self.rank, self.world = rank, world_size
        self.prefetch = prefetch
        self.with_segments = with_segments
        self._consumed = 0  # samples_consumed: global samples handed out
        self.prefetch_full = threading.Event()  # set when the queue is full (tests wait on it)
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(prefetch, 1))
        self._stop = threading.Event()

    # ---- batch construction (pure) -----------------------------------------------------------
    def make_batch(self, t_start: int) -> Batch:
        per = self.G // self.world
        t = np.arange(t_start + self.rank * per, t_start + (self.rank + 1) * per, dtype=np.int64)
        src, rows = self.mix.sampler.sample_many(t)
        row_len = self.mix.row_len
        tokens = np.empty((per, row_len), dtype=self.mix.datasets[0].dtype)
        loss_mask = np.ones((per, row_len - 1), dtype=np.uint8)
        seg_ids = np.full((per, row_len), -1, dtype=np.int32) if self.with_segments else None
        for j, (s, r) in enumerate(zip(src.tolist(), rows.tolist(), strict=True)):
            ds = self.mix.datasets[s]
            tokens[j] = ds.row(r)
            _, _, lengths, _ = ds.segments(r)
            filled = int(lengths.astype(np.int64).sum())
            loss_mask[j, max(filled - 1, 0) :] = 0  # target position p+1 >= filled is padding
            if seg_ids is not None:
                seg_ids[j, :filled] = np.repeat(
                    np.arange(lengths.shape[0], dtype=np.int32), lengths.astype(np.int64)
                )
        return Batch(
            tokens=tokens,
            loss_mask=loss_mask,
            segment_ids=seg_ids,
            sample_index=t,
            source_index=src.astype(np.int16),
            row_id=rows,
            step=t_start // self.G,
        )

    # ---- iteration ---------------------------------------------------------------------------
    def __iter__(self) -> Loader:
        return self

    def __next__(self) -> Batch:
        if self.prefetch <= 0:
            batch = self.make_batch(self._consumed)
        else:
            if self._thread is None:
                self._start(self._consumed)
            self.prefetch_full.clear()  # clear BEFORE get, so a refill can always set it again
            item = self._queue.get()
            if isinstance(item, BaseException):
                raise item
            batch = item
        self._consumed += self.G  # consumed, not produced
        return batch

    def _start(self, t: int) -> None:
        self._stop.clear()
        self._queue = queue.Queue(maxsize=self.prefetch)
        self._thread = threading.Thread(
            target=self._produce,
            args=(t, self._queue, self._stop),
            daemon=True,
            name="w2s-prefetch",
        )
        self._thread.start()

    def _produce(self, t: int, q: queue.Queue[Any], stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                item: Any = self.make_batch(t)
            except BaseException as exc:  # surfaced to the consumer by __next__
                item = exc
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.05)
                    break
                except queue.Full:
                    self.prefetch_full.set()
                    continue
            if q.full():
                self.prefetch_full.set()
            if isinstance(item, BaseException):
                return
            t += self.G

    def _halt(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        self.prefetch_full.clear()

    def close(self) -> None:
        self._halt()

    def __enter__(self) -> Loader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- state -------------------------------------------------------------------------------
    @property
    def samples_consumed(self) -> int:
        return self._consumed

    def state_dict(self) -> dict[str, Any]:
        """JSON-serializable loader_state.json; identical on every rank."""
        return make_state(self.mix, self._consumed, self.G)

    def load_state_dict(self, state: Mapping[str, Any], allow_mismatch: bool = False) -> None:
        samples = check_state(self.mix, state, allow_mismatch)
        if int(state["global_batch_size"]) != self.G:
            log.warning(
                "global batch size changed from %s to %s; continuing at sample %s",
                state["global_batch_size"],
                self.G,
                samples,
            )
        self._halt()  # drain: discard anything produced from the old position
        self._consumed = samples
