"""CPU-only IQ8R numeric TP3 sharding oracle using actual .rad tensor rows.

This is NOT the Radiance runtime, a quantization/kernel implementation, or
evidence of TP3 GPU execution. It reads selected real model rows only.
"""
import json
import math
import os
from pathlib import Path
import struct

from test_ltx_iq8r_planes import HEADER, ENTRY, PLANE, inspect_iq8r_planes

HEADS = 32
VIDEO_HD = 128
AUDIO_HD = 64
FWHT = 128
GROUP = 256
SPLITS = ((8, 8, 16), (12, 12, 8), (10, 10, 12))


def fwht_blocks(values: list[float]) -> list[float]:
    """Unnormalised Sylvester H128, as libquant/lq_common.h::fwht_inplace."""
    if len(values) == 0 or len(values) % FWHT:
        raise ValueError("FWHT requires nonempty 128-wide K blocks")
    out = list(values)
    for start in range(0, len(out), FWHT):
        stride = 1
        while stride < FWHT:
            for off in range(start, start + FWHT, 2 * stride):
                for j in range(off, off + stride):
                    a, b = out[j], out[j + stride]
                    out[j], out[j + stride] = a + b, a - b
            stride *= 2
    return out


def bf16(u16: int) -> float:
    if not 0 <= u16 < 65536:
        raise ValueError("BF16 out of range")
    return struct.unpack("<f", struct.pack("<I", u16 << 16))[0]


def shard_columns(head_dim: int, shares: tuple[int, int, int]) -> tuple[tuple[int, int], ...]:
    """Refuse any K cut that would split an IQ8R U8 scale or H128."""
    if len(shares) != 3 or sum(shares) != HEADS or any(h <= 0 for h in shares):
        raise ValueError("invalid TP3 head shares")
    out = []
    lo = 0
    for heads in shares:
        hi = lo + heads * head_dim
        if lo % GROUP or hi % GROUP or lo % FWHT or hi % FWHT:
            raise ValueError("IQ8R K boundary splits a scale/FWHT group")
        out.append((lo, hi))
        lo = hi
    return tuple(out)


def bank_rows(inner: int, cuts: tuple[tuple[int, int], ...]) -> list[list[tuple[int, int]]]:
    """Each local fused QKV is [Q-part, K-part, V-part], not one contiguous slab."""
    if cuts[0][0] != 0 or cuts[-1][1] != inner:
        raise ValueError("bank partition incomplete")
    return [[(bank * inner + lo, bank * inner + hi) for bank in range(3)]
            for lo, hi in cuts]


def activation(n: int) -> list[float]:
    return [math.sin(i * 0.037) * 0.5 + math.cos(i * 0.011) * 0.75
            + ((i * 17) % 13 - 6) * 0.003 for i in range(n)]


def dequant_rotated(codes: list[int], scales: bytes, row_scale: float) -> list[float]:
    """The container's *stored rotated* weight vector, not a kernel layout."""
    k = len(codes)
    if k < GROUP or k % GROUP or len(scales) != k // GROUP:
        raise ValueError("IQ8R code/group length mismatch")
    if not math.isfinite(row_scale):
        raise ValueError("nonfinite BF16 secondary scale")
    return [float(c) * float(scales[i // GROUP]) * row_scale
            for i, c in enumerate(codes)]


def compare_row(
    rotated: list[float], x: list[float], hx: list[float],
    cuts: tuple[tuple[int, int], ...] | None = None,
) -> tuple[float, float, float]:
    """Two independent formulations of a decoded-weight matvec.

    Container stores v = w H128 / 128, so decoded w = v H128.
    Compare <vH, x> and <v, Hx>; partitioning the latter by rank must
    preserve whole scale groups. Uses fp64 summation, NOT native i8a8h math.
    """
    if len(rotated) != len(x) or len(hx) != len(x):
        raise ValueError("shape mismatch")
    w = fwht_blocks(rotated)
    dense = math.fsum(w[i] * x[i] for i in range(len(x)))
    rotated_dot = math.fsum(rotated[i] * hx[i] for i in range(len(x)))
    if cuts is None:
        tp = rotated_dot
    else:
        if cuts[0][0] != 0 or cuts[-1][1] != len(x):
            raise ValueError("incomplete K partition")
        terms = []
        previous = 0
        for lo, hi in cuts:
            if lo != previous or hi <= lo or hi % GROUP or lo % GROUP:
                raise ValueError("invalid IQ8R K cut")
            terms.append(math.fsum(rotated[i] * hx[i] for i in range(lo, hi)))
            previous = hi
        tp = math.fsum(terms)
    tolerance = 1e-8 * (1.0 + abs(dense))
    if abs(dense - rotated_dot) > tolerance or abs(dense - tp) > tolerance:
        raise AssertionError("decoded dense/rotated/TP3 matvec mismatch")
    return dense, rotated_dot, tp


class SelectedImageRows:
    """Selected rows only: no mmap and no full model-weight data reads."""

    def __init__(self, path: Path):
        self.file = path.open("rb")
        self.size = path.stat().st_size
        h = HEADER.unpack(self.file.read(HEADER.size))
        if (h[0], h[1], h[2]) != (0x31444152, 2, self.size):
            raise ValueError("not a complete RAD2 header")
        self.file.seek(h[8])
        strings = self.file.read(h[9])
        if len(strings) != h[9] or len(strings) > 16 * 1024 * 1024:
            raise ValueError("invalid string blob")

        def name(off: int) -> str:
            if off < 0 or off >= len(strings):
                raise ValueError("bad weight name offset")
            end = strings.find(b"\0", off)
            if end < 0:
                raise ValueError("unterminated name")
            return strings[off:end].decode("utf8")

        self.file.seek(h[14])
        self.planes = list(PLANE.iter_unpack(self.file.read(h[15] * PLANE.size)))
        if len(self.planes) != h[15]:
            raise ValueError("truncated plane table")
        self.file.seek(h[12])
        raw = self.file.read(h[13] * ENTRY.size)
        if len(raw) != h[13] * ENTRY.size:
            raise ValueError("truncated directory")
        self.entries = {name(row[0]): row for row in ENTRY.iter_unpack(raw)}

    def close(self) -> None:
        self.file.close()

    def read_plane_row(self, entry: tuple, plane: int, row: int, width: int) -> bytes:
        first = entry[15]
        off, total = self.planes[first + plane]
        if row < 0 or width <= 0 or (row + 1) * width > total:
            raise ValueError("plane row out of bounds")
        self.file.seek(off + row * width)
        data = self.file.read(width)
        if len(data) != width:
            raise ValueError("short plane row read")
        return data

    def iq8r_row(self, name: str, row: int) -> list[float]:
        entry = self.entries[name]
        enc, rank = entry[3:5]
        shape = entry[5:11][:rank]
        if enc != 1 or rank != 2 or entry[14] != 3:
            raise ValueError("expected IQ8R 2D linear")
        n, k = shape
        if row < 0 or row >= n or k % GROUP:
            raise ValueError("invalid row/K")
        data = self.read_plane_row(entry, 0, row, k)
        sc = self.read_plane_row(entry, 1, row, k // GROUP)
        row_scale = bf16(struct.unpack("<H", self.read_plane_row(entry, 2, row, 2))[0])
        codes = list(struct.unpack("<" + str(k) + "b", data))
        return dequant_rotated(codes, sc, row_scale)

    def bias(self, name: str, row: int) -> float:
        e = self.entries[name]
        if e[3] != 0 or e[4] != 1 or e[5] <= row or e[14] != 1:
            raise ValueError("expected BF16 one-plane bias")
        return bf16(struct.unpack("<H", self.read_plane_row(e, 0, row, 2))[0])



def audit_all_dit_projections(image: SelectedImageRows) -> dict[str, object]:
    """Real container shape contract for all 48 DiT blocks, not just block 0."""
    required = (
        "a2v.q.weight", "a2v.kv.weight", "a2v.out.weight",
        "attn1.qkv.weight", "attn1.out.weight",
        "attn2.q.weight", "attn2.kv.weight", "attn2.out.weight",
        "audio_attn1.qkv.weight", "audio_attn1.out.weight",
        "audio_attn2.q.weight", "audio_attn2.kv.weight", "audio_attn2.out.weight",
        "v2a.q.weight", "v2a.kv.weight", "v2a.out.weight",
        "ff_up.weight", "ff_down.weight", "audio_ff_up.weight", "audio_ff_down.weight",
    )
    out_or_down = tuple(x for x in required if x.endswith(("out.weight", "ff_down.weight")))
    shapes = {}
    for suffix in required:
        entry = image.entries["dit.blk.0." + suffix]
        assert entry[3] == 1 and entry[4] == 2 and entry[14] == 3
        shapes[suffix] = entry[5:7]

    blocked = {"/".join(map(str, s)): set() for s in SPLITS}
    totals = {key: 0 for key in blocked}
    count = 0
    for block in range(48):
        for suffix in required:
            entry = image.entries["dit.blk." + str(block) + "." + suffix]
            if entry[3] != 1 or entry[4] != 2 or entry[14] != 3 or entry[5:7] != shapes[suffix]:
                raise AssertionError("DiT IQ8R projection shape differs across blocks")
            count += 1
        for suffix in out_or_down:
            n, k = shapes[suffix]
            if k % HEADS:
                raise AssertionError("output K not head-partitionable")
            for shares in SPLITS:
                key = "/".join(map(str, shares))
                totals[key] += 1
                try:
                    cuts = shard_columns(k // HEADS, shares)
                    assert cuts[-1][1] == k
                except ValueError:
                    blocked[key].add(suffix)
    assert count == 48 * len(required) == 960
    assert len(out_or_down) == 8
    assert totals == {key: 384 for key in blocked}
    invalid_audio = {
        "audio_attn1.out.weight", "audio_attn2.out.weight",
        "a2v.out.weight", "v2a.out.weight",
    }
    assert blocked["8/8/16"] == set() and blocked["12/12/8"] == set()
    assert blocked["10/10/12"] == invalid_audio
    return {
        "checked_dit_blocks": 48, "iq8r_projection_entries": count,
        "output_k_shard_checks_per_split": 384,
        "rejected_10_10_12_ops_per_block": sorted(invalid_audio),
        "rejected_10_10_12_total": 48 * len(invalid_audio),
    }



def audit_connector_and_cross_modal(image: SelectedImageRows) -> dict[str, object]:
    """Model-backed, bounded CPU seam for connector and DiT cross-modal tensors.

    Current LTX ownership keeps connectors on the leader; the hypothetical
    per-rank IQ8R K cuts below are informative, not a connector TP3 request.
    """
    connectors = 0
    selected_rows = 0
    bad_candidate_cuts = {}
    for name, width in (("video", 4096), ("audio", 2048)):
        expected = {
            "qkv.weight": (3 * width, width),
            "out.weight": (width, width),
            "ff_up.weight": (4 * width, width),
            "ff_down.weight": (width, 4 * width),
        }
        for block in range(8):
            prefix = "conn." + name + ".blk." + str(block) + "."
            for suffix, shape in expected.items():
                entry = image.entries[prefix + suffix]
                if (entry[3], entry[4], entry[14], entry[5:7]) != (1, 2, 3, shape):
                    raise AssertionError("connector IQ8R projection mismatch: " + prefix + suffix)
                connectors += 1
            for suffix, shape in (
                ("q_norm.weight", (width,)),
                ("k_norm.weight", (width,)),
                ("gate.weight", (HEADS, width)),
            ):
                e = image.entries[prefix + suffix]
                if e[3] != 0 or e[5:5 + e[4]] != shape:
                    raise AssertionError("connector BF16 norm/gate mismatch: " + prefix + suffix)
            if block in (0, 7):
                for suffix in ("qkv.weight", "out.weight", "ff_down.weight"):
                    e = image.entries[prefix + suffix]
                    rows, k = e[5:7]
                    x = activation(k)
                    hx = fwht_blocks(x)
                    for row in (0, rows - 1):
                        compare_row(image.iq8r_row(prefix + suffix, row), x, hx)
                        selected_rows += 1
            if name == "audio":
                try:
                    shard_columns(width // HEADS, (10, 10, 12))
                except ValueError:
                    bad_candidate_cuts["audio_connector_out"] = (
                        bad_candidate_cuts.get("audio_connector_out", 0) + 1)

    cross_types = (
        "attn2.q.weight", "attn2.kv.weight",
        "audio_attn2.q.weight", "audio_attn2.kv.weight",
        "a2v.q.weight", "a2v.kv.weight", "v2a.q.weight", "v2a.kv.weight",
        "ff_down.weight", "audio_ff_down.weight",
    )
    cross_entries = 0
    for block in (0, 27, 28, 47):
        for suffix in cross_types:
            name = "dit.blk." + str(block) + "." + suffix
            e = image.entries[name]
            if e[3] != 1 or e[4] != 2 or e[14] != 3:
                raise AssertionError("cross-modal IQ8R projection absent: " + name)
            rows, k = e[5:7]
            if k % GROUP:
                raise AssertionError("nonrepresentable IQ8R K")
            x, hx = activation(k), None
            hx = fwht_blocks(x)
            for row in (0, rows - 1):
                compare_row(image.iq8r_row(name, row), x, hx)
                selected_rows += 1
            cross_entries += 1
    assert connectors == 64
    assert cross_entries == 40
    assert selected_rows == 24 + 80 == 104
    assert bad_candidate_cuts == {"audio_connector_out": 8}
    return {
        "connector_layers_per_modality": 8,
        "connector_iq8r_entries": connectors,
        "cross_modal_sampled_projection_entries": cross_entries,
        "real_rows_checked": selected_rows,
        "hypothetical_audio_connector_k_cut_rejections": 8,
        "connector_owner": "leader_only",
    }


def run_real_model(path: Path) -> dict[str, object]:
    inspected = inspect_iq8r_planes(path)
    image = SelectedImageRows(path)
    verified = {}
    try:
        for label, hd, inner in (("video", VIDEO_HD, 4096), ("audio", AUDIO_HD, 2048)):
            prefix = "dit.blk.0.attn1" if label == "video" else "dit.blk.0.audio_attn1"
            fused = prefix + ".qkv.weight"
            out = prefix + ".out.weight"
            bias = prefix + ".out.bias"
            assert image.entries[fused][5:7] == (3 * inner, inner)
            assert image.entries[out][5:7] == (inner, inner)
            x, hx = activation(inner), None
            hx = fwht_blocks(x)
            checks = 0
            valid = []
            refused = []
            for shares in SPLITS:
                head_cuts = []
                at = 0
                for count in shares:
                    head_cuts.append((at, at + count * hd))
                    at += count * hd
                assert at == inner
                mapped = bank_rows(inner, tuple(head_cuts))
                # The compact local QKV comprises three corresponding banks.
                expanded = sorted(i for bank in range(3) for r in mapped
                                  for i in range(r[bank][0], r[bank][1]))
                assert expanded == list(range(3 * inner))
                assert mapped[1][0][0] != head_cuts[1][0] * 3  # wrong contiguous-fused slice
                for rank in range(3):
                    for bank in range(3):
                        a, b = mapped[rank][bank]
                        for row in {a, b - 1}:
                            rotated = image.iq8r_row(fused, row)
                            compare_row(rotated, x, hx)
                            checks += 1
                try:
                    cuts = shard_columns(hd, shares)
                except ValueError:
                    refused.append("/".join(map(str, shares)))
                    continue
                valid.append("/".join(map(str, shares)))
                for out_row in (0, 17, inner - 1):
                    rotated = image.iq8r_row(out, out_row)
                    dense, native_form, partials = compare_row(rotated, x, hx, cuts)
                    real_bias = image.bias(bias, out_row)
                    once = dense + real_bias
                    tol = 1e-8 * (1 + abs(once))
                    assert abs(once - (partials + real_bias)) <= tol
                    # Negative control: adding each ranks bias before reduction
                    # overcounts it three times; the real model has nonzero biases.
                    assert abs(real_bias) > tol
                    assert abs(once - (partials + 3 * real_bias)) > tol
                    checks += 1
            verified[label] = {"head_dim": hd, "model_rows_checked": checks,
                               "valid_k_splits": valid, "refused_k_splits": refused}
        assert verified["video"]["refused_k_splits"] == []
        assert verified["audio"]["refused_k_splits"] == ["10/10/12"]
        return {"iq8r_entries_structurally_valid": inspected["count"], "results": verified,
                "dit_inventory": audit_all_dit_projections(image),
                "connector_cross_modal": audit_connector_and_cross_modal(image)}
    finally:
        image.close()


def test_fwht_and_uneven_group_cuts() -> None:
    data = [float((i * 31) % 47 - 20) / 13 for i in range(768)]
    roundtrip = fwht_blocks(fwht_blocks(data))
    assert all(abs(z - 128 * a) < 1e-9 for z, a in zip(roundtrip, data))
    rotated = dequant_rotated([(i * 11) % 255 - 127 for i in range(768)],
                              bytes([2, 5, 7]), bf16(0x3e80))
    x = activation(768)
    compare_row(rotated, x, fwht_blocks(x), ((0, 256), (256, 512), (512, 768)))
    assert shard_columns(VIDEO_HD, (10, 10, 12)) == ((0, 1280), (1280, 2560), (2560, 4096))
    assert shard_columns(AUDIO_HD, (8, 8, 16)) == ((0, 512), (512, 1024), (1024, 2048))
    import pytest
    with pytest.raises(ValueError, match="IQ8R K boundary"):
        shard_columns(AUDIO_HD, (10, 10, 12))
    with pytest.raises(ValueError, match="invalid IQ8R K cut"):
        compare_row(rotated, x, fwht_blocks(x), ((0, 128), (128, 512), (512, 768)))
    with pytest.raises(ValueError, match="group length"):
        dequant_rotated([1] * 768, bytes([3, 4]), 1.0)


def test_actual_ltx_iq8r_numeric_when_supplied() -> None:
    p = os.environ.get("BC_LTX_IQ8R_MODEL")
    if not p:
        import pytest
        pytest.skip("requires opt-in BC_LTX_IQ8R_MODEL")
    result = run_real_model(Path(p))
    assert result["iq8r_entries_structurally_valid"] > 1000
    assert result["dit_inventory"]["iq8r_projection_entries"] == 960
    assert result["dit_inventory"]["rejected_10_10_12_total"] == 192
    assert result["connector_cross_modal"]["connector_iq8r_entries"] == 64
    assert result["connector_cross_modal"]["real_rows_checked"] == 104


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 test_ltx_iq8r_numeric.py /path/to/model.rad")
    print(json.dumps(run_real_model(Path(sys.argv[1])), indent=2, sort_keys=True))
