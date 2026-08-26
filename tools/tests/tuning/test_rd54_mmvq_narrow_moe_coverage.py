"""RD54: does BigCherry's measured MMVQ dispatch already cover the narrow-K
MoE nwarps policy from llama.cpp PR #20831 ("cuda: dynamic MMVQ nwarps for
narrow matrices", open, head 40dc8a1bebcb2570ea56921a2d1a4aa4d750e170)?

PR #20831 adds, for RDNA3_0/RDNA4 MMVQ parameter tables only:

    blocks_per_row     = ncols_x / qk
    max_useful_nwarps  = (blocks_per_row * qi) / (vdr * warp_size)
    nwarps             = largest power of two <= max_useful_nwarps, capped at
                          the native max (8) and floored at 1

BigCherry never needs K as an explicit *candidate* dimension because
`ggml_hip_make_signature()` hashes the full device-local src0 shape
(including K, via canonical ne0[0] == upstream's ncols_x) -- so the same
compiled {nwarps, rows_per_block, small_k} candidate set already gets
independently measured per distinct K-width signature. This module pins two
offline, verifiable facts that the real-hardware measurement (recorded
separately, not fabricated here) depends on:

1. The compiled candidate catalog already spans the power-of-two nwarps
   space {1,2,4,8} PR #20831 selects from, for width==1 (ncols_dst==1,
   ordinary single-token decode) Q8_0 MMVQ.
2. BigCherry's own forced-dispatch entry point (patches/0650) already fails
   closed (GGML_ABORT) rather than silently attributing a measurement to the
   wrong kernel, for `has_ids && ncols_dst > 1` -- the dedicated MoE
   multi-token kernel (`mul_mat_vec_q_moe_launch`) that b10502 routes to
   instead of the ordinary `calc_nwarps()`-driven path. This is why RD54's
   real-hardware experiment must use ordinary width-1 decode signatures, not
   multi-token speculative/MTP MoE signatures, to test PR #20831's
   mechanism -- and why that boundary is worth pinning as a regression, not
   just stated in the plan-item notes.

The Q8_0/wave32 formula reduces to ``max_useful_nwarps = K // 256`` (QK8_0=32,
QI8_0=8, VDR_Q8_0_Q8_1_MMVQ=2, warp_size=32: (K/32*8)/(2*32) = K/256). The
``predicted_nwarps_q8_0`` helper below is test-only scaffolding for comparing
real measured winners (from a real record+tune run) against PR #20831's
closed-form prediction -- it is not production code and is not part of any
patch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import catalog as cat

ROOT = Path(__file__).resolve().parents[3]


def predicted_nwarps_q8_0(k: int) -> int:
    """PR #20831's closed-form nwarps policy, specialised to Q8_0/wave32.

    Mirrors `max_useful_nwarps = (blocks_per_row * qi) / (vdr * warp_size)`
    then "largest power of two <= that, capped at 8, floored at 1" exactly --
    independently re-derived from the real diff and type-trait constants
    (QK8_0=32, QI8_0=8, VDR_Q8_0_Q8_1_MMVQ=2), not copied from the PR.
    """
    blocks_per_row = k // 32
    raw = (blocks_per_row * 8) // (2 * 32)
    nwarps = 1
    w = 2
    while w <= raw and w <= 8:
        nwarps = w
        w *= 2
    return nwarps


def test_predicted_nwarps_formula_matches_known_boundaries():
    # K=256->1, 512->2, 1024->4, 2048->8: the exact boundaries the plan
    # discussion with dev-gpt-agent derived from the real PR #20831 formula.
    assert predicted_nwarps_q8_0(256) == 1
    assert predicted_nwarps_q8_0(511) == 1
    assert predicted_nwarps_q8_0(512) == 2
    assert predicted_nwarps_q8_0(1024) == 4
    assert predicted_nwarps_q8_0(2047) == 4
    assert predicted_nwarps_q8_0(2048) == 8
    assert predicted_nwarps_q8_0(4096) == 8  # capped at native max


def _width1_q8_0_candidates() -> list[cat.Candidate]:
    inventory = cat.Inventory()  # empty + restrict=False == full-max (unrestricted)
    architectures = list(cat.DEFAULT_ARCHITECTURES)
    candidates = cat.enumerate_mmvq(
        architectures, ["GGML_TYPE_Q8_0"], inventory, restrict=False, staged=True,
    )
    return [c for c in candidates if c.config["width"] == 1]


def test_catalog_covers_pr20831_power_of_two_nwarps_at_width1_q8_0():
    """The compiled candidate set already spans PR #20831's selection space
    (nwarps in {1,2,4,8}) for ordinary width-1 (ncols_dst==1) Q8_0 decode --
    the exact shape family the real-hardware experiment targets."""
    candidates = _width1_q8_0_candidates()
    assert candidates, "expected width==1 q8_0 MMVQ candidates to exist"
    nwarps_present = {c.config["nwarps"] for c in candidates}
    assert {1, 2, 4, 8} <= nwarps_present, nwarps_present
    # BigCherry's extra search point beyond upstream's power-of-two policy.
    assert 6 in nwarps_present


def test_catalog_candidates_are_geometry_valid():
    for c in _width1_q8_0_candidates():
        assert cat.mmvq_geometry_is_valid(
            c.config["width"], c.config["nwarps"], c.config["rows_per_block"]
        )


def test_forced_geometry_aborts_for_moe_multitoken_not_silently_misattributed():
    """Pins the boundary RD54's real-hardware experiment design depends on:
    BigCherry's own forced-dispatch entry point (0650) must keep refusing
    `has_ids && ncols_dst > 1` rather than silently running the dedicated MoE
    multi-token kernel (mul_mat_vec_q_moe_launch, b10502) under an nwX
    candidate's name. If this guard is ever removed, multi-token/MTP MoE
    signatures could get silently (and wrongly) credited to an ordinary MMVQ
    nwarps candidate -- which would invalidate the width-1-only methodology
    RD54's real-hardware comparison against PR #20831 relies on."""
    patch_src = (ROOT / "patches" / "0650_mmvq_native_variant.py").read_text(encoding="utf-8")
    assert "has_ids && ncols_dst > 1" in patch_src
    assert "GGML_ABORT" in patch_src

    vendor_src = (ROOT / "vendor" / "llama.cpp" / "ggml" / "src"
                  / "ggml-cuda" / "mmvq.cu").read_text(encoding="utf-8")
    assert "if (has_ids && ncols_dst > 1)" in vendor_src
    assert "GGML_ABORT" in vendor_src
    # And the dedicated MoE multi-token kernel this guard exists because of
    # is real, not a claim taken on faith.
    assert "mul_mat_vec_q_moe_launch" in vendor_src


def test_ncols_x_is_src0_ne0_not_a_derived_quantity():
    """Pins that upstream's `ncols_x` (what PR #20831's formula keys on) is
    literally `src0->ne[0]` (== `ne00`), passed straight through -- so a
    canonical signature's `ne0[0]` is the correct, direct real-hardware value
    to feed `predicted_nwarps_q8_0()` against, with no unit conversion."""
    vendor_src = (ROOT / "vendor" / "llama.cpp" / "ggml" / "src"
                  / "ggml-cuda" / "mmvq.cu").read_text(encoding="utf-8")
    assert "const int64_t ne00 = src0->ne[0];" in vendor_src
    assert "mul_mat_vec_q_switch_type(" in vendor_src
