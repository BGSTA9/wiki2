import numpy as np
import pytest
from typing import Tuple

from wiki2shards.io.binfmt import (
    HEADER_SIZE,
    U2,
    U4,
    BadMagic,
    DtypeRangeError,
    IdxHeader,
    Segments,
    SizeMismatch,
    UnsupportedVersion,
    checked_cast,
    decode_idx,
    dtype_for_vocab,
    encode_idx,
    idx_size,
)


def _sample(n_rows: int = 3) -> tuple[IdxHeader, Segments]:
    seg = Segments(
        row_seg_ptr=np.array([0, 2, 3, 5][: n_rows + 1], dtype=np.uint64),
        doc_uid=np.array([11, 12, 12, 13, 14], dtype=np.uint64),
        start=np.array([0, 3, 0, 0, 4], dtype=np.uint32),
        length=np.array([3, 5, 8, 4, 4], dtype=np.uint32),
        piece=np.array([0, 0, 1, 0, 0], dtype=np.uint32),
    )
    hdr = IdxHeader(
        itemsize=2,
        pack_mode=0,
        split=1,
        row_len=8,
        shard_index=4,
        n_rows=3,
        n_segments=5,
        n_pad_tokens=0,
        first_global_row=120,
        bin_sha256=bytes(range(32)),
        tokenizer_fp=bytes(range(32, 64)),
    )
    return hdr, seg


def test_idx_roundtrip() -> None:
    hdr, seg = _sample()
    raw = encode_idx(hdr, seg)
    assert len(raw) == idx_size(3, 5) == HEADER_SIZE + 8 * 4 + 20 * 5
    assert idx_size(1000, 1600) == 40_136  # SPEC 1.4 worked example
    hdr2, seg2 = decode_idx(raw)
    assert hdr2 == hdr
    for name in ("row_seg_ptr", "doc_uid", "start", "length", "piece"):
        assert np.array_equal(getattr(seg2, name), getattr(seg, name))

    with pytest.raises(BadMagic):
        decode_idx(b"W2S-IDY\0" + raw[8:])
    bumped_major = raw[:8] + (2).to_bytes(2, "little") + raw[10:]
    with pytest.raises(UnsupportedVersion):
        decode_idx(bumped_major)
    with pytest.raises(SizeMismatch):
        decode_idx(raw[:-1])
    with pytest.raises(SizeMismatch):
        decode_idx(raw + b"\0")
    with pytest.raises(SizeMismatch):
        decode_idx(raw[:100])
    higher_minor = raw[:10] + (7).to_bytes(2, "little") + raw[12:]
    assert decode_idx(higher_minor)[0].version_minor == 7  # readers accept higher minors


def test_dtype_guard():
    ids = np.array([1, 2, 70_000], dtype=np.int64)
    assert int(ids.astype(U2)[-1]) == 4464  # what a plain cast would silently store
    with pytest.raises(DtypeRangeError):
        checked_cast(ids, U2)
    with pytest.raises(DtypeRangeError):
        checked_cast(np.array([-1]), U4)
    assert checked_cast(ids, U4).tolist() == [1, 2, 70_000]
    assert dtype_for_vocab(65_536) == U2 and dtype_for_vocab(65_537) == U4
