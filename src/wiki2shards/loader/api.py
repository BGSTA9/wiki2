"""Public loader types. Frozen at contracts-v1.

from wiki2shards.loader import Dataset, Mixture, Loader
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

STATE_FORMAT = "wiki2shards.loader_state"
STATE_VERSION = 1


@dataclass(frozen=True)
class Batch:
    tokens: NDArray[np.unsignedinteger]  # [B, row_len]; inputs tokens[:, :-1], targets [:, 1:]
    loss_mask: NDArray[np.uint8]  # [B, seq_len]; 0 where the target is padding
    segment_ids: NDArray[np.int32] | None  # [B, row_len]; segment index in row, -1 on padding
    sample_index: NDArray[np.int64]  # [B]; the t of each row
    source_index: NDArray[np.int16]  # [B]; index into the mixture's sources
    row_id: NDArray[np.int64]  # [B]; global row within its source
    step: int  # global batch index


class LoaderStateError(ValueError):
    """load_state_dict refused: format, fingerprint, dataset or draw-count mismatch."""
