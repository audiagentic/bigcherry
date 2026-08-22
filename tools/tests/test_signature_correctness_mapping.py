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


def _mul_mat_signature(tmp_path, *, m=16, n=1, k=256, src0_type=0, src1_type=0):
    vendor = _write_fixture_vendor(tmp_path)
    op_names = scm.load_ggml_op_names(vendor)
    op_id = next(i for i, name in op_names.items() if name == "MUL_MAT")
    signature = {
        "op": op_id,
        "src0_type": src0_type,
        "src1_type": src1_type,
        "ne0": [k, m, 1, 1],
        "ne1": [k, n, 1, 1],
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
