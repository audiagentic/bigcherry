"""HI80: tests for tools/bigcherry/signature_correctness_mapping.py --
the enum-table extraction (against both a minimal synthetic fixture and,
where available, the real vendored source) and the MUL_MAT-only
op_filter/target_tensor derivation. No HIP hardware needed for this
module's own correctness; the exact op_filter format was independently
verified against real Brutus test-backend-ops output before this file
was written (see the module's own docstrings for the verification
transcript)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import signature_correctness_mapping as scm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_VENDOR_ROOT = REPO_ROOT / "vendor" / "llama.cpp"


def _write_fixture_vendor(tmp_path: Path) -> Path:
    """A minimal, hand-written stand-in for ggml.h/ggml.c's real structure --
    small enough to exercise the parser's own logic in isolation from
    whatever the real (much larger, and potentially locally-unavailable)
    vendored source currently looks like."""
    vendor = tmp_path / "vendor" / "llama.cpp"
    (vendor / "ggml" / "include").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "src").mkdir(parents=True, exist_ok=True)

    (vendor / "ggml" / "include" / "ggml.h").write_text(
        "enum ggml_type {\n"
        "    GGML_TYPE_F32  = 0,\n"
        "    GGML_TYPE_F16  = 1,\n"
        "    GGML_TYPE_Q8_0 = 8,\n"
        "};\n"
        "enum ggml_op {\n"
        "    GGML_OP_NONE,\n"
        "    GGML_OP_ADD,\n"
        "    GGML_OP_MUL_MAT,\n"
        "    GGML_OP_COUNT,\n"
        "};\n",
        encoding="utf-8",
    )
    (vendor / "ggml" / "src" / "ggml.c").write_text(
        "static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {\n"
        '    [GGML_TYPE_F32] = {\n'
        '        .type_name = "f32",\n'
        "        .blck_size = 1,\n"
        "    },\n"
        '    [GGML_TYPE_F16] = {\n'
        '        .type_name = "f16",\n'
        "    },\n"
        '    [GGML_TYPE_Q8_0] = {\n'
        '        .type_name = "q8_0",\n'
        "    },\n"
        "};\n"
        'static const char * GGML_OP_NAME[GGML_OP_COUNT] = {\n'
        '    "NONE",\n'
        '    "ADD",\n'
        '    "MUL_MAT",\n'
        "};\n",
        encoding="utf-8",
    )
    return vendor


def test_load_ggml_type_names_from_fixture(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    names = scm.load_ggml_type_names(vendor)
    assert names == {0: "f32", 1: "f16", 8: "q8_0"}


def test_load_ggml_op_names_from_fixture(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    names = scm.load_ggml_op_names(vendor)
    assert names[2] == "MUL_MAT"
    assert names[0] == "NONE"
    assert names[1] == "ADD"


def test_missing_vendor_file_fails_closed(tmp_path):
    empty = tmp_path / "nowhere"
    try:
        scm.load_ggml_type_names(empty)
        assert False, "expected SignatureMappingError"
    except scm.SignatureMappingError as exc:
        assert "not found" in str(exc)


def _mul_mat_signature(tmp_path, *, m=16, n=1, k=256, src0_type=0, src1_type=0, dst_type=0):
    vendor = _write_fixture_vendor(tmp_path)
    op_names = scm.load_ggml_op_names(vendor)
    op_id = next(i for i, name in op_names.items() if name == "MUL_MAT")
    signature = {
        "op": op_id,
        "src0_type": src0_type,
        "src1_type": src1_type,
        "dst_type": dst_type,
        "ne0": [k, m, 1, 1],
        "ne1": [k, n, 1, 1],
        "ned": [m, n, 1, 1],
    }
    return vendor, signature


def test_mul_mat_op_filter_matches_the_real_vars_string_shape(tmp_path):
    # This exact vars() string ("type_a=f32,type_b=f32,m=16,n=1,k=256,
    # bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1") was observed on real
    # Brutus hardware via test-backend-ops -o MUL_MAT -p 'type_a=f32'
    # before this test was written.
    vendor, signature = _mul_mat_signature(tmp_path)
    op_filter, target_tensor = scm.signature_to_op_filter(signature, vendor_root=vendor)
    real_vars_line = "type_a=f32,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1"
    assert re.search(op_filter, real_vars_line)
    assert target_tensor == "out"


def test_mul_mat_op_filter_does_not_match_a_different_signature(tmp_path):
    vendor, signature = _mul_mat_signature(tmp_path, m=16, n=1, k=256)
    op_filter, _ = scm.signature_to_op_filter(signature, vendor_root=vendor)
    other_vars_line = "type_a=f32,type_b=f32,m=32,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1"
    assert re.search(op_filter, other_vars_line) is None


def test_mul_mat_op_filter_does_not_match_a_batched_variant(tmp_path):
    # The 13-distinct-matches finding this module's docstring records:
    # confirms the anchored full-string filter rejects a same-mnk but
    # differently-batched case, not just a different mnk.
    vendor, signature = _mul_mat_signature(tmp_path, m=16, n=1, k=256)
    op_filter, _ = scm.signature_to_op_filter(signature, vendor_root=vendor)
    batched_vars_line = "type_a=f32,type_b=f32,m=16,n=1,k=256,bs=[3,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1"
    assert re.search(op_filter, batched_vars_line) is None


def test_test_file_line_matches_real_hardware_transcript(tmp_path):
    # HI80 (2026-08-23 real-hardware finding): a real gpt-oss-20B MUL_MAT
    # signature (m=32, n=21, k=2880, f32/f32) matches NONE of
    # test-backend-ops' own fixed synthetic corpus shapes, so
    # signature_to_op_filter() produces a filter that runs 0 tests.
    # signature_to_test_file_line() instead drives test-backend-ops'
    # --test-file escape hatch, verified end to end on real Brutus
    # hardware to fire BIGCHERRY_CORRECTNESS_METRIC for tensor "out" for
    # this exact line.
    vendor, signature = _mul_mat_signature(tmp_path, m=32, n=21, k=2880)
    line, target_tensor, digest_tensor = scm.signature_to_test_file_line(signature, vendor_root=vendor)
    op_id = signature["op"]
    assert line == (
        f"{op_id} 0 32 21 1 1 0 2 "
        "0 2880 32 1 1 4 11520 368640 368640 "
        "0 2880 21 1 1 4 11520 241920 241920 -"
    )
    assert target_tensor == "out"
    assert digest_tensor == "leaf_0"


def test_test_file_line_rejects_a_non_mul_mat_op(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    signature = {"op": 1, "src0_type": 0, "src1_type": 0, "dst_type": 0,
                 "ne0": [1, 1, 1, 1], "ne1": [1, 1, 1, 1], "ned": [1, 1, 1, 1]}
    try:
        scm.signature_to_test_file_line(signature, vendor_root=vendor)
        assert False, "expected SignatureMappingError for a non-MUL_MAT op"
    except scm.SignatureMappingError as exc:
        assert "MUL_MAT" in str(exc)


def test_test_file_line_rejects_batched_outer_dims(tmp_path):
    vendor, signature = _mul_mat_signature(tmp_path)
    signature["ne0"] = [256, 16, 3, 1]
    try:
        scm.signature_to_test_file_line(signature, vendor_root=vendor)
        assert False, "expected SignatureMappingError for batched outer dims"
    except scm.SignatureMappingError as exc:
        assert "outer dimensions" in str(exc)


def test_test_file_line_rejects_a_quantized_type(tmp_path):
    vendor, signature = _mul_mat_signature(tmp_path, src0_type=8, src1_type=8)
    try:
        scm.signature_to_test_file_line(signature, vendor_root=vendor)
        assert False, "expected SignatureMappingError for an unsupported (quantized) type"
    except scm.SignatureMappingError as exc:
        assert "q8_0" in str(exc).lower() or "F32" in str(exc)


def test_test_file_line_rejects_missing_ned(tmp_path):
    vendor, signature = _mul_mat_signature(tmp_path)
    del signature["ned"]
    try:
        scm.signature_to_test_file_line(signature, vendor_root=vendor)
        assert False, "expected SignatureMappingError for missing ned"
    except scm.SignatureMappingError as exc:
        assert "ned" in str(exc)


def test_different_types_produce_different_filters(tmp_path):
    vendor, sig_f32 = _mul_mat_signature(tmp_path, src0_type=0, src1_type=0)
    _, sig_q8 = _mul_mat_signature(tmp_path, src0_type=8, src1_type=8)
    filter_f32, _ = scm.signature_to_op_filter(sig_f32, vendor_root=vendor)
    filter_q8, _ = scm.signature_to_op_filter(sig_q8, vendor_root=vendor)
    assert filter_f32 != filter_q8
    assert "q8_0" in filter_q8
    assert "f32" in filter_f32


def test_non_mul_mat_op_is_rejected(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    signature = {"op": 1, "src0_type": 0, "src1_type": 0, "ne0": [1, 1, 1, 1], "ne1": [1, 1, 1, 1]}
    try:
        scm.signature_to_op_filter(signature, vendor_root=vendor)
        assert False, "expected SignatureMappingError for a non-MUL_MAT op"
    except scm.SignatureMappingError as exc:
        assert "MUL_MAT" in str(exc)


def test_unknown_op_id_is_rejected(tmp_path):
    vendor = _write_fixture_vendor(tmp_path)
    signature = {"op": 999, "src0_type": 0, "src1_type": 0, "ne0": [1, 1, 1, 1], "ne1": [1, 1, 1, 1]}
    try:
        scm.signature_to_op_filter(signature, vendor_root=vendor)
        assert False, "expected SignatureMappingError for an unknown op id"
    except scm.SignatureMappingError as exc:
        assert "unknown" in str(exc).lower()


def test_mismatched_inner_dimension_is_rejected(tmp_path):
    vendor, signature = _mul_mat_signature(tmp_path)
    signature["ne1"] = [999, 1, 1, 1]  # disagrees with ne0's k=256
    try:
        scm.signature_to_op_filter(signature, vendor_root=vendor)
        assert False, "expected SignatureMappingError for mismatched k"
    except scm.SignatureMappingError as exc:
        assert "disagrees" in str(exc)


def test_batched_src0_outer_dims_are_rejected(tmp_path):
    vendor, signature = _mul_mat_signature(tmp_path)
    signature["ne0"] = [256, 16, 3, 1]  # ne0[2]=3, not the assumed 1
    try:
        scm.signature_to_op_filter(signature, vendor_root=vendor)
        assert False, "expected SignatureMappingError for batched src0 outer dims"
    except scm.SignatureMappingError as exc:
        assert "src0 outer dimensions" in str(exc)


def test_batched_src1_outer_dims_are_rejected(tmp_path):
    vendor, signature = _mul_mat_signature(tmp_path)
    signature["ne1"] = [256, 1, 1, 2]  # ne1[3]=2, not the assumed 1
    try:
        scm.signature_to_op_filter(signature, vendor_root=vendor)
        assert False, "expected SignatureMappingError for batched src1 outer dims"
    except scm.SignatureMappingError as exc:
        assert "src1 outer dimensions" in str(exc)


def test_unknown_type_id_is_rejected(tmp_path):
    vendor, signature = _mul_mat_signature(tmp_path, src0_type=250)
    try:
        scm.signature_to_op_filter(signature, vendor_root=vendor)
        assert False, "expected SignatureMappingError for an unknown type id"
    except scm.SignatureMappingError as exc:
        assert "unknown ggml_type" in str(exc)


def test_malformed_ne_array_is_rejected(tmp_path):
    vendor, signature = _mul_mat_signature(tmp_path)
    signature["ne0"] = [1, 2, 3]  # only 3 elements, not 4
    try:
        scm.signature_to_op_filter(signature, vendor_root=vendor)
        assert False, "expected SignatureMappingError for a malformed ne0"
    except scm.SignatureMappingError as exc:
        assert "ne0" in str(exc)


class RealVendorSourceTests:
    """Only runs when the real vendored llama.cpp checkout is present next
    to this repo (it is, in this working tree, but this class is kept
    separable so the fixture-based tests above remain the load-bearing
    ones for CI environments that may not have vendor/llama.cpp checked
    out)."""

    @staticmethod
    def run_if_available():
        if not REAL_VENDOR_ROOT.is_dir():
            return
        type_names = scm.load_ggml_type_names(REAL_VENDOR_ROOT)
        assert type_names[0] == "f32"
        assert type_names[8] == "q8_0"
        op_names = scm.load_ggml_op_names(REAL_VENDOR_ROOT)
        assert "MUL_MAT" in op_names.values()


def test_real_vendor_source_parses_cleanly_when_present():
    RealVendorSourceTests.run_if_available()


# HI105: MUL_MAT_ID mapping tests, using RD54's own two real recorded
# canonical signatures (Brutus, Qwen3.6-35B-A3B Q8_0, 2026-08-24) rather
# than synthetic fixtures -- these are the exact real production shapes
# HI105 exists to unblock correctness evidence for.

# K=256 (down expert projection): op=30 (MUL_MAT_ID, confirmed via
# load_ggml_op_names against the real vendored source), fusion=0/glu_op=0
# (unfused), n_expert=256/n_expert_used=8, ne1[1]==8==n_expert_used (down's
# real non-broadcast input shape, confirmed against build_moe_ffn).
_RD54_K256_DOWN_SIGNATURE = {
    "dst_type": 0, "flags": 31, "fusion": 0, "glu_op": 0,
    "n_expert": 256, "n_expert_used": 8,
    "ne0": [256, 2048, 256, 1], "ne1": [256, 8, 1, 1], "ned": [2048, 8, 1, 1],
    "op": 30, "prec": 0, "schema_version": 1,
    "src0_type": 8, "src1_type": 0,
}

# K=2048 (RD54's other real signature): op=100, which resolves to GLU, NOT
# MUL_MAT_ID -- confirmed via load_ggml_op_names against the real vendored
# source. fusion=2/glu_op=2 (a real fused GLU-activation node) is why:
# GLU has its own real, distinct test-backend-ops class (test_glu,
# tests/test-backend-ops.cpp) with different vars()/build_graph() than
# MUL_MAT_ID's, so signature_to_mul_mat_id_test_file_line correctly
# refuses it -- matching HI80's own doctrine that a new op needs its own
# verified test_case-subclass derivation before this module maps it,
# never a same-op assumption from a shared has_ids/n_expert flag.
_RD54_K2048_GLU_SIGNATURE = {
    "dst_type": 0, "flags": 31, "fusion": 2, "glu_op": 2,
    "n_expert": 256, "n_expert_used": 8,
    "ne0": [2048, 256, 256, 1], "ne1": [2048, 1, 1, 1], "ned": [256, 8, 1, 1],
    "op": 100, "prec": 2, "schema_version": 1,
    "src0_type": 8, "src1_type": 0,
}


def _require_real_vendor_root():
    if not REAL_VENDOR_ROOT.is_dir():
        import pytest
        pytest.skip("real vendor/llama.cpp checkout not present")


def test_mul_mat_id_maps_rd54_k256_down_signature():
    _require_real_vendor_root()
    line, target_tensor, digest_tensor = scm.signature_to_mul_mat_id_test_file_line(
        _RD54_K256_DOWN_SIGNATURE, vendor_root=REAL_VENDOR_ROOT,
    )
    fields = line.split()
    assert fields[-1] == "-"
    assert target_tensor == "out"
    assert digest_tensor == "leaf_0"

    op_id, dst_type_id = int(fields[0]), int(fields[1])
    ned = [int(x) for x in fields[2:6]]
    op_params_count = int(fields[6])
    num_src = int(fields[7])
    assert op_id == 30
    assert dst_type_id == 0
    assert ned == [2048, 8, 1, 1]
    assert op_params_count == 0
    assert num_src == 3  # as, b, ids -- ggml_mul_mat_id's real source order

    idx = 8
    src0_type_id = int(fields[idx]); idx += 1
    src0_ne = [int(x) for x in fields[idx:idx + 4]]; idx += 4
    src0_nb = [int(x) for x in fields[idx:idx + 4]]; idx += 4
    assert src0_type_id == 8  # Q8_0
    assert src0_ne == [256, 2048, 256, 1]  # as: [k, m, n_expert]
    assert src0_nb == [0, 0, 0, 0]  # default-contiguous, no Q8_0 layout math in Python

    src1_type_id = int(fields[idx]); idx += 1
    src1_ne = [int(x) for x in fields[idx:idx + 4]]; idx += 4
    src1_nb = [int(x) for x in fields[idx:idx + 4]]; idx += 4
    assert src1_type_id == 0  # F32
    assert src1_ne == [256, 8, 1, 1]  # b, taken literally from the real signature
    assert src1_nb == [0, 0, 0, 0]

    ids_type_id = int(fields[idx]); idx += 1
    ids_ne = [int(x) for x in fields[idx:idx + 4]]; idx += 4
    ids_nb = [int(x) for x in fields[idx:idx + 4]]; idx += 4
    type_names = scm.load_ggml_type_names(REAL_VENDOR_ROOT)
    assert type_names[ids_type_id].upper() == "I32"
    assert ids_ne == [8, 1, 1, 1]  # [n_expert_used, n_tokens]
    assert ids_nb == [0, 0, 0, 0]

    assert idx == len(fields) - 1  # nothing left but the trailing "-"


def test_mul_mat_id_rejects_the_real_glu_signature():
    _require_real_vendor_root()
    # RD54's OTHER real signature is a real ggml GLU node, not MUL_MAT_ID
    # (op=100 resolves to GLU) -- fail-closed rejection here is correct,
    # not a bug: GLU needs its own test_glu-derived mapping, out of scope
    # for this HI105 slice.
    try:
        scm.signature_to_mul_mat_id_test_file_line(
            _RD54_K2048_GLU_SIGNATURE, vendor_root=REAL_VENDOR_ROOT,
        )
        assert False, "expected SignatureMappingError for a real GLU signature"
    except scm.SignatureMappingError as exc:
        assert "MUL_MAT_ID" in str(exc)
        assert "GLU" in str(exc)


def test_mul_mat_id_rejects_signature_without_has_ids_flag():
    _require_real_vendor_root()
    sig = dict(_RD54_K256_DOWN_SIGNATURE)
    sig["flags"] = 7  # contiguity bits only, HAS_IDS (bit 3) cleared
    try:
        scm.signature_to_mul_mat_id_test_file_line(sig, vendor_root=REAL_VENDOR_ROOT)
        assert False, "expected SignatureMappingError for a signature without HAS_IDS"
    except scm.SignatureMappingError as exc:
        assert "HAS_IDS" in str(exc)


def test_mul_mat_id_rejects_inconsistent_n_expert():
    _require_real_vendor_root()
    sig = dict(_RD54_K256_DOWN_SIGNATURE)
    sig["n_expert"] = 128  # disagrees with the real ne0[2]=256
    try:
        scm.signature_to_mul_mat_id_test_file_line(sig, vendor_root=REAL_VENDOR_ROOT)
        assert False, "expected SignatureMappingError for inconsistent n_expert"
    except scm.SignatureMappingError as exc:
        assert "n_expert" in str(exc)


def test_ggml_type_id_for_name_finds_i32():
    _require_real_vendor_root()
    type_names = scm.load_ggml_type_names(REAL_VENDOR_ROOT)
    i32_id = scm._ggml_type_id_for_name(type_names, "I32")
    assert type_names[i32_id].upper() == "I32"
