"""RUM15 IQ8R scale precision contract; uses original checkpoint planes when supplied.

The GPU layout must retain the exact F32 U8-group * BF16-row product.
Do not weaken this oracle to BF16 simply to match a lossy device layout.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import struct

import pytest

from test_ltx_iq8r_numeric import SelectedImageRows, bf16


def as_f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def to_bf16(value: float) -> int:
    u = struct.unpack("<I", struct.pack("<f", value))[0]
    if u & 0x7FFFFFFF > 0x7F800000:
        return (u >> 16) | 0x40
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


@pytest.mark.parametrize("gain", (0x0001, 0x007F, 0x3C81, 0x3F81,
                                   0x3F80, 0x4081, 0xBF81, 0xC081))
def test_u8_times_bf16_has_an_exact_f32_product(gain: int) -> None:
    value = bf16(gain)
    for coefficient in range(256):
        exact = float(coefficient) * value
        assert math.isfinite(exact)
        assert exact == as_f32(exact), "IQ8R scale needs no additional rounding in F32"


def test_bf16_fold_would_destroy_representable_checkpoint_scale() -> None:
    scale = 127 * bf16(0x3F81)
    assert scale == as_f32(scale)
    assert bf16(to_bf16(scale)) != scale


def test_real_checkpoint_requires_f32_fold_when_supplied() -> None:
    model = os.environ.get("BC_LTX_IQ8R_MODEL")
    if not model:
        pytest.skip("opt in with BC_LTX_IQ8R_MODEL")
    image = SelectedImageRows(Path(model))
    changed = total = 0
    try:
        for modality in ("attn1", "audio_attn1"):
            name = f"dit.blk.0.{modality}.qkv.weight"
            entry = image.entries[name]
            rows, k = entry[5:7]
            assert entry[3:5] == (1, 2) and entry[14] == 3
            assert k % 256 == 0
            for row in (0, rows // 3, rows // 2, rows - 1):
                gains = image.read_plane_row(entry, 2, row, 2)
                gain = bf16(struct.unpack("<H", gains)[0])
                code_scales = image.read_plane_row(entry, 1, row, k // 256)
                for coefficient in code_scales:
                    correct = float(coefficient) * gain
                    assert correct == as_f32(correct)
                    total += 1
                    changed += bf16(to_bf16(correct)) != correct
    finally:
        image.close()
    assert total > 0 and changed > 0, "real IQ8R model must expose precision loss"
