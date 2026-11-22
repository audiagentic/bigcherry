"""Read-only C01 IQ8R raw-plane shard contract against optional real .rad model.

This validates *logical* row/column spans and every raw plane, not the
Radiance weight loader, kernel layout packer, TP3 execution, or hardware.
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct

from test_ltx_iq8r_numeric import GROUP, HEADS, SPLITS, SelectedImageRows

OUTPUTS = (
    "attn1.out.weight", "attn2.out.weight",
    "audio_attn1.out.weight", "audio_attn2.out.weight",
    "a2v.out.weight", "v2a.out.weight",
    "ff_down.weight", "audio_ff_down.weight",
)
FUSED = (
    ("attn1.qkv.weight", 3),
    ("audio_attn1.qkv.weight", 3),
    ("attn2.kv.weight", 2),
    ("audio_attn2.kv.weight", 2),
    ("a2v.kv.weight", 2),
    ("v2a.kv.weight", 2),
)
INVALID_101012 = {
    "audio_attn1.out.weight", "audio_attn2.out.weight",
    "a2v.out.weight", "v2a.out.weight",
}
BIASLESS_OUTPUTS = {"ff_down.weight"}


@dataclass(frozen=True)
class RawRow:
    codes: bytes
    group_scales: bytes
    row_scale: bytes


def k_cuts(k: int, shares: tuple[int, int, int]) -> tuple[tuple[int, int], ...]:
    if k <= 0 or k % HEADS or len(shares) != 3 or any(x <= 0 for x in shares) or sum(shares) != HEADS:
        raise ValueError("invalid TP3 K partition")
    widths = [k * count // HEADS for count in shares]
    if sum(widths) != k:
        raise ValueError("nonintegral rank width")
    lo = 0
    cuts = []
    for width in widths:
        hi = lo + width
        if lo % GROUP or hi % GROUP:
            raise ValueError("IQ8R 256-K group split")
        cuts.append((lo, hi))
        lo = hi
    if lo != k:
        raise ValueError("incomplete rank coverage")
    return tuple(cuts)


def pack_raw_shards(raw: RawRow, cuts: tuple[tuple[int, int], ...]) -> tuple[RawRow, ...]:
    """The two-level scale views must slice alongside the code plane."""
    k = len(raw.codes)
    if k <= 0 or k % GROUP or len(raw.group_scales) != k // GROUP or len(raw.row_scale) != 2:
        raise ValueError("corrupt IQ8R raw row planes")
    if len(cuts) != 3 or cuts[0][0] != 0 or cuts[-1][1] != k:
        raise ValueError("incomplete 3-rank spans")
    result = []
    prev = 0
    for lo, hi in cuts:
        if lo != prev or hi <= lo or lo % GROUP or hi % GROUP:
            raise ValueError("invalid IQ8R K span")
        result.append(RawRow(
            raw.codes[lo:hi],
            raw.group_scales[lo // GROUP:hi // GROUP],
            raw.row_scale,  # the BF16 secondary scale is per whole N row
        ))
        prev = hi
    if (b"".join(x.codes for x in result) != raw.codes or
            b"".join(x.group_scales for x in result) != raw.group_scales or
            any(x.row_scale != raw.row_scale for x in result)):
        raise AssertionError("IQ8R planes do not roundtrip through ranks")
    return tuple(result)


def raw_row(image: SelectedImageRows, name: str, row: int) -> RawRow:
    e = image.entries[name]
    if e[3] != 1 or e[4] != 2 or e[14] != 3:
        raise ValueError("not a 2D IQ8R linear")
    n, k = e[5:7]
    if row < 0 or row >= n or k % GROUP:
        raise ValueError("invalid IQ8R row or K")
    result = RawRow(
        image.read_plane_row(e, 0, row, k),
        image.read_plane_row(e, 1, row, k // GROUP),
        image.read_plane_row(e, 2, row, 2),
    )
    if not math.isfinite(struct.unpack("<f", struct.pack(
            "<I", struct.unpack("<H", result.row_scale)[0] << 16))[0]):
        raise ValueError("nonfinite BF16 secondary scale")
    return result


def fused_row_index(inner: int, lo: int, width: int, banks: int, local_row: int) -> int:
    if banks not in (2, 3) or inner <= 0 or width <= 0 or lo < 0 or lo + width > inner:
        raise ValueError("invalid fused K/V or Q/K/V ownership")
    if local_row < 0 or local_row >= banks * width:
        raise ValueError("local row out of range")
    bank, inside = divmod(local_row, width)
    return bank * inner + lo + inside


def audit_real_model(path: Path) -> dict[str, int]:
    image = SelectedImageRows(path)
    outputs_checked = 0
    valid_slices = 0
    blocked_slices = 0
    fused_checked = 0
    fused_boundary_rows = 0
    checked_biased = 0
    try:
        for block in range(48):
            prefix = f"dit.blk.{block}."
            for suffix in OUTPUTS:
                name = prefix + suffix
                e = image.entries[name]
                if e[3] != 1 or e[4] != 2 or e[14] != 3:
                    raise ValueError("invalid IQ8R DiT output")
                n, k = e[5:7]
                bias_name = name.removesuffix(".weight") + ".bias"
                expected = suffix not in BIASLESS_OUTPUTS
                if (bias_name in image.entries) != expected:
                    raise AssertionError("per-projection bias contract changed")
                if expected:
                    b = image.entries[bias_name]
                    if b[3] != 0 or b[4] != 1 or b[5] != n:
                        raise AssertionError("output bias shape mismatch")
                    checked_biased += 1
                for row in (0, n - 1):
                    raw = raw_row(image, name, row)
                    outputs_checked += 1
                    for shares in SPLITS:
                        try:
                            cuts = k_cuts(k, shares)
                        except ValueError:
                            if shares != (10, 10, 12) or suffix not in INVALID_101012:
                                raise AssertionError("unexpected unavailable K shard")
                            blocked_slices += 1
                            continue
                        ranks = pack_raw_shards(raw, cuts)
                        for r, (lo, hi) in enumerate(cuts):
                            # Compare each rank's logical code and U8 scale plane
                            # with source offsets, not just full reassembly.
                            if (ranks[r].codes != raw.codes[lo:hi] or
                                    ranks[r].group_scales != raw.group_scales[lo // GROUP:hi // GROUP] or
                                    ranks[r].row_scale != raw.row_scale):
                                raise AssertionError("rank's IQ8R plane offset mismatch")
                        valid_slices += 1

            for suffix, banks in FUSED:
                name = prefix + suffix
                e = image.entries[name]
                if e[3] != 1 or e[4] != 2 or e[14] != 3:
                    raise ValueError("invalid DiT fused projection")
                rows, k = e[5:7]
                if rows % banks:
                    raise AssertionError("noninteger fused bank rows")
                inner = rows // banks
                if inner % HEADS:
                    raise AssertionError("fused head ownership nonintegral")
                bias = image.entries.get(name.removesuffix(".weight") + ".bias")
                if bias is None or bias[3] != 0 or bias[4] != 1 or bias[5] != rows:
                    raise AssertionError("fused bias does not have full bank extents")
                fused_checked += 1
                # Check all 48 geometries, plus boundary weight plane bytes at
                # the tricky STG block and the first/last blocks.
                if block not in (0, 28, 47):
                    continue
                for shares in SPLITS:
                    lo = 0
                    for count in shares:
                        width = count * inner // HEADS
                        hi = lo + width
                        if width <= 0 or width % 16 or hi > inner:
                            raise AssertionError("fused row span not native-layout-aligned")
                        for bank in range(banks):
                            for offset in (0, width - 1):
                                local = bank * width + offset
                                source = fused_row_index(inner, lo, width, banks, local)
                                if source != bank * inner + lo + offset:
                                    raise AssertionError("wrong fused QKV/KV bank map")
                                raw = raw_row(image, name, source)
                                if (len(raw.codes) != k or
                                        len(raw.group_scales) != k // GROUP or
                                        len(raw.row_scale) != 2):
                                    raise AssertionError("fused source plane shape mismatch")
                                fused_boundary_rows += 1
                        lo = hi
                    if lo != inner:
                        raise AssertionError("missing fused-bank source rows")

        assert outputs_checked == 768
        assert valid_slices == 1920  # 768 * 3 - 384 invalid sampled K cuts
        assert blocked_slices == 384  # 4 invalid output types * 48 blocks * 2 rows
        assert fused_checked == 288
        assert fused_boundary_rows == 756  # (2*3+4*2) banks * 3 ranks * 2 rows * 3 splits * 3 blocks
        assert checked_biased == 336  # seven of eight output types per block
        return {
            "output_raw_rows": outputs_checked,
            "valid_rank_row_roundtrips": valid_slices,
            "invalid_rank_row_splits": blocked_slices,
            "fused_projection_inventory": fused_checked,
            "fused_boundary_source_rows": fused_boundary_rows,
            "output_bias_tensors": checked_biased,
            "biasless_video_ff_down": 48,
        }
    finally:
        image.close()


def test_synthetic_raw_plane_roundtrip_and_bad_cuts() -> None:
    import pytest
    k = 1024
    raw = RawRow(bytes((i * 17 + 13) % 256 for i in range(k)),
                 bytes([3, 7, 19, 23]), struct.pack("<H", 0x3e80))
    cuts = k_cuts(k, (8, 8, 16))
    assert cuts == ((0, 256), (256, 512), (512, 1024))
    shards = pack_raw_shards(raw, cuts)
    assert [x.group_scales for x in shards] == [bytes([3]), bytes([7]), bytes([19, 23])]
    assert all(x.row_scale == raw.row_scale for x in shards)
    with pytest.raises(ValueError, match="256-K"):
        k_cuts(2048, (10, 10, 12))
    with pytest.raises(ValueError, match="invalid IQ8R K span"):
        pack_raw_shards(raw, ((0, 256), (128, 512), (512, 1024)))
    with pytest.raises(ValueError, match="corrupt IQ8R"):
        pack_raw_shards(RawRow(raw.codes, b"\x01", raw.row_scale), cuts)
    with pytest.raises(ValueError, match="corrupt IQ8R"):
        pack_raw_shards(RawRow(raw.codes, raw.group_scales, b"\x00"), cuts)
    assert fused_row_index(4096, 1024, 1024, 3, 1024) == 4096 + 1024
    assert fused_row_index(2048, 512, 512, 2, 512) == 2048 + 512
    with pytest.raises(ValueError, match="out of range"):
        fused_row_index(2048, 512, 512, 2, 1024)


def test_real_model_raw_shards_when_supplied() -> None:
    import pytest
    p = os.environ.get("BC_LTX_IQ8R_MODEL")
    if not p:
        pytest.skip("requires optional BC_LTX_IQ8R_MODEL")
    outcome = audit_real_model(Path(p))
    assert outcome["biasless_video_ff_down"] == 48


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 test_ltx_iq8r_shards.py /path/to/ltx-iq8r.rad")
    print(json.dumps(audit_real_model(Path(sys.argv[1])), indent=2, sort_keys=True))
