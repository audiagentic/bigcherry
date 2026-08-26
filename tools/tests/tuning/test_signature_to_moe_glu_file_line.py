"""HI119: tests for signature_to_moe_glu_file_line() -- maps a real fused
MUL_MAT_ID(gate)+MUL_MAT_ID(up)+GLU dispatch signature into a
--moe-glu-file line for the registered test_bigcherry_moe_glu_fusion
class (patches 1239/1240).

The real-shape/real-line correctness case (test_maps_hi108s_real_dispatch_
exactly) uses the ACTUAL canonical signature captured on real Brutus
hardware (GGML_HIP_DISPATCH_MODE=record against a live instance of the
registered test case, 2026-08-25) -- not a hand-picked example -- and
asserts the produced line matches what was independently confirmed on
that same hardware to reproduce the identical recorded signature via
--moe-glu-file. See docs/planning/active/hip-autotune/HI119.md for the
full real-hardware trail."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import signature_mapping as scm  # noqa: E402


def _write_fixture_vendor(tmp_path: Path) -> Path:
    vendor = tmp_path / "vendor" / "llama.cpp"
    (vendor / "ggml" / "include").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "src").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "include" / "ggml.h").write_text(
        "enum ggml_type {\n"
        "    GGML_TYPE_F32  = 0,\n"
        "    GGML_TYPE_Q8_0 = 8,\n"
        "};\n"
        "enum ggml_op {\n"
        "    GGML_OP_NONE,\n"
        "    GGML_OP_ADD,\n"
        "    GGML_OP_MUL_MAT,\n"
        "    GGML_OP_MUL_MAT_ID,\n"
        "    GGML_OP_GLU,\n"
        "    GGML_OP_COUNT,\n"
        "};\n",
        encoding="utf-8",
    )
    (vendor / "ggml" / "src" / "ggml.c").write_text(
        "static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {\n"
        '    [GGML_TYPE_F32] = {\n        .type_name = "f32",\n    },\n'
        '    [GGML_TYPE_Q8_0] = {\n        .type_name = "q8_0",\n    },\n'
        "};\n"
        'static const char * GGML_OP_NAME[GGML_OP_COUNT] = {\n'
        '    "NONE",\n    "ADD",\n    "MUL_MAT",\n    "MUL_MAT_ID",\n    "GLU",\n'
        "};\n",
        encoding="utf-8",
    )
    return vendor


# The real, complete canonical signature captured on real Brutus hardware
# (GGML_HIP_DISPATCH_MODE=record) against a live instance of
# test_bigcherry_moe_glu_fusion(GGML_TYPE_Q8_0, GGML_GLU_OP_SWIGLU, k=2048,
# n=256, m=1, n_mats=256, n_used=8, b=true) -- structurally identical to
# HI108's real blocked dispatch (7ef2471585a5aa6fbb49384efe566ac5) on
# every field: fusion=GATE, op=GLU, ne0/ne1/ned, n_expert, n_expert_used.
_REAL_SIGNATURE = {
    "dst_type": 0, "flags": 31, "fusion": 2, "glu_op": 2,
    "n_expert": 256, "n_expert_used": 8,
    "ne0": [2048, 256, 256, 1], "ne1": [2048, 1, 1, 1], "ned": [256, 8, 1, 1],
    "op": 4,  # GGML_OP_GLU in this fixture's enum
    "src0_type": 8, "src1_type": 0,
    "schema_version": scm.dispatch_abi.SIGNATURE_SCHEMA_VERSION,
}


def test_maps_hi108s_real_dispatch_exactly(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    line, target, digest = scm.signature_to_moe_glu_file_line(_REAL_SIGNATURE, vendor_root=vendor)
    # Independently confirmed on real Brutus hardware: this exact line,
    # fed through --moe-glu-file, produces the identical recorded
    # signature as the statically-registered instance (fusion=2, op=100,
    # ne0=[2048,256,256,1], ne1=[2048,1,1,1], ned=[256,8,1,1]).
    assert line == "8 2 2048 256 1 256 8 1"
    assert target == "fused_glu"
    assert digest == "ids"


def test_rejects_non_routed_dispatch(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    sig = dict(_REAL_SIGNATURE)
    sig["flags"] = 31 & ~(1 << 3)  # clear HAS_IDS
    try:
        scm.signature_to_moe_glu_file_line(sig, vendor_root=vendor)
        assert False, "expected SignatureMappingError"
    except scm.SignatureMappingError as exc:
        assert "HAS_IDS" in str(exc)


def test_rejects_x_bias_fusion_bit(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    sig = dict(_REAL_SIGNATURE)
    sig["flags"] = _REAL_SIGNATURE["flags"] | (1 << 7)  # GGML_HIP_SIG_FUSION_X_BIAS
    try:
        scm.signature_to_moe_glu_file_line(sig, vendor_root=vendor)
        assert False, "expected SignatureMappingError"
    except scm.SignatureMappingError as exc:
        assert "bias/scale" in str(exc)


def test_rejects_gate_scale_fusion_bit(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    sig = dict(_REAL_SIGNATURE)
    sig["flags"] = _REAL_SIGNATURE["flags"] | (1 << 10)  # GGML_HIP_SIG_FUSION_GATE_SCALE
    try:
        scm.signature_to_moe_glu_file_line(sig, vendor_root=vendor)
        assert False, "expected SignatureMappingError"
    except scm.SignatureMappingError as exc:
        assert "bias/scale" in str(exc)


def test_rejects_non_gate_fusion_kind(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    sig = dict(_REAL_SIGNATURE)
    sig["fusion"] = 1  # GGML_HIP_FUSION_BIAS, not GATE
    try:
        scm.signature_to_moe_glu_file_line(sig, vendor_root=vendor)
        assert False, "expected SignatureMappingError"
    except scm.SignatureMappingError as exc:
        assert "GGML_HIP_FUSION_GATE" in str(exc)


def test_rejects_unfusable_glu_op(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    sig = dict(_REAL_SIGNATURE)
    sig["glu_op"] = 0  # REGLU -- never fused by ggml_cuda_can_fuse
    try:
        scm.signature_to_moe_glu_file_line(sig, vendor_root=vendor)
        assert False, "expected SignatureMappingError"
    except scm.SignatureMappingError as exc:
        assert "GEGLU/SWIGLU/SWIGLU_OAI" in str(exc)


def test_rejects_non_glu_op(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    sig = dict(_REAL_SIGNATURE)
    sig["op"] = 3  # MUL_MAT_ID
    try:
        scm.signature_to_moe_glu_file_line(sig, vendor_root=vendor)
        assert False, "expected SignatureMappingError"
    except scm.SignatureMappingError as exc:
        assert "only supports GLU" in str(exc)


def test_non_broadcast_shape_maps_broadcast_zero(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    sig = dict(_REAL_SIGNATURE)
    sig["ne1"] = [2048, 8, 1, 1]  # ne1[1] == n_expert_used, the non-broadcast form
    line, _, _ = scm.signature_to_moe_glu_file_line(sig, vendor_root=vendor)
    assert line == "8 2 2048 256 1 256 8 0"


def test_rejects_stale_schema_version(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    sig = dict(_REAL_SIGNATURE)
    sig["schema_version"] = 1  # pre-HI118 schema -- flags bits 7-10 are meaningless
    try:
        scm.signature_to_moe_glu_file_line(sig, vendor_root=vendor)
        assert False, "expected SignatureMappingError"
    except scm.SignatureMappingError as exc:
        assert "schema_version" in str(exc)


def test_rejects_batched_m_dimension(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    sig = dict(_REAL_SIGNATURE)
    sig["ne1"] = [2048, 1, 3, 1]  # m=3 -- ids layout for m>1 isn't hashed
    sig["ned"] = [256, 8, 3, 1]
    try:
        scm.signature_to_moe_glu_file_line(sig, vendor_root=vendor)
        assert False, "expected SignatureMappingError"
    except scm.SignatureMappingError as exc:
        assert "m==1" in str(exc) or "m, the per-call batch dimension" in str(exc)


def test_rejects_invalid_ne1_middle_dimension(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    sig = dict(_REAL_SIGNATURE)
    sig["ne1"] = [2048, 3, 1, 1]  # neither 1 nor n_expert_used=8
    try:
        scm.signature_to_moe_glu_file_line(sig, vendor_root=vendor)
        assert False, "expected SignatureMappingError"
    except scm.SignatureMappingError as exc:
        assert "broadcast" in str(exc)
