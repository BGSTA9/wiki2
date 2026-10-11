"""Dataset: one split of one dataset directory, read through lazily opened memmaps."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from wiki2shards.io.binfmt import Segments, open_bin, read_idx
from wiki2shards.io.manifest import Manifest, ShardEntry
from wiki2shards.verify import verify_dataset

DEFAULT_LRU = 64


class LRU[V]:
    """Keep at most `size` opened shards (memmaps or parsed .idx arrays)."""

    def __init__(self, size: int, load: Callable[[int], V]) -> None:
        self.size, self.load = size, load
        self.items: OrderedDict[int, V] = OrderedDict()

    def __getitem__(self, k: int) -> V:
        if k in self.items:
            self.items.move_to_end(k)
            return self.items[k]
        value = self.items[k] = self.load(k)
        if len(self.items) > self.size:
            self.items.popitem(last=False)
        return value


class DatasetError(Exception):
    pass


class Dataset:
    def __init__(
        self, path: Path, split: str, manifest: Manifest, dataset_id: str, lru: int = DEFAULT_LRU
    ) -> None:
        if split not in manifest.splits:
            raise DatasetError(f"{path}: no split {split!r}")
        self.path = path
        self.split = split
        self.manifest = manifest
        self.dataset_id = dataset_id
        self.shards: list[ShardEntry] = manifest.splits[split].shards
        rows = [s.n_rows for s in self.shards]
        self._first_row = np.concatenate([[0], np.cumsum(rows)]).astype(np.int64)
        self._bins = LRU(lru, self._open_bin)
        self._idx = LRU(lru, self._open_idx)

    @classmethod
    def open(
        cls,
        path: Path | str,
        split: str = "train",
        verify: Literal["none", "header", "full"] = "header",
    ) -> Dataset:
        path = Path(path)
        if verify != "none":
            rep = verify_dataset(path, full=verify == "full")
            if rep.exit_code:
                raise DatasetError(f"{path} failed verification: {rep.fatal or rep.errors[:3]}")
        from wiki2shards.io.manifest import read_manifest

        manifest, dataset_id = read_manifest(path)
        return cls(path, split, manifest, dataset_id)

    # ---- metadata --------------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self._first_row[-1])

    @property
    def row_len(self) -> int:
        return self.manifest.packing.row_len

    @property
    def seq_len(self) -> int:
        return self.manifest.packing.seq_len

    @property
    def dtype(self) -> np.dtype[np.unsignedinteger]:
        return np.dtype(self.manifest.tokenizer.dtype)

    @property
    def tokenizer_fp(self) -> str:
        return self.manifest.tokenizer.fingerprint

    @property
    def vocab_size(self) -> int:
        return self.manifest.tokenizer.vocab_size

    @property
    def eos_id(self) -> int:
        return self.manifest.tokenizer.eos_id

    @property
    def pad_id(self) -> int:
        return self.manifest.tokenizer.pad_id

    # ---- rows ------------------------------------------------------------------------------
    def _locate(self, row: int) -> tuple[int, int]:
        if not 0 <= row < len(self):
            raise IndexError(f"row {row} out of range [0, {len(self)})")
        k = int(np.searchsorted(self._first_row, row, side="right")) - 1
        return k, row - int(self._first_row[k])

    def _open_bin(self, k: int) -> NDArray[np.unsignedinteger]:
        sh = self.shards[k]
        return open_bin(self.path / sh.bin, self.dtype.itemsize, sh.n_rows, self.row_len)

    def _open_idx(self, k: int) -> Segments:
        return read_idx(self.path / self.shards[k].idx)[1]

    def row(self, i: int) -> NDArray[np.unsignedinteger]:
        k, r = self._locate(i)
        return np.array(self._bins[k][r])  # a copy, never a view into the memmap

    def rows(self, ids: NDArray[np.integer]) -> NDArray[np.unsignedinteger]:
        out = np.empty((len(ids), self.row_len), dtype=self.dtype)
        for j, i in enumerate(np.asarray(ids).tolist()):
            out[j] = self.row(int(i))
        return out

    def segments(
        self, i: int
    ) -> tuple[NDArray[np.uint64], NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint32]]:
        """(doc_uid, start, length, piece) of the segments in row i."""
        k, r = self._locate(i)
        seg = self._idx[k]
        a, b = int(seg.row_seg_ptr[r]), int(seg.row_seg_ptr[r + 1])
        return seg.doc_uid[a:b], seg.start[a:b], seg.length[a:b], seg.piece[a:b]
