"""HI121 M3: tests for hip_required_capabilities()'s fail-closed applicability
rules -- the central "zero-and-known != zero-and-unknown" distinction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import dispatch_abi  # noqa: E402
from bigcherry.tuning import hip_capabilities as hc  # noqa: E402
from bigcherry.tuning import signature_capabilities as sc  # noqa: E402


def _write_fixture_vendor(tmp_path: Path) -> Path:
    vendor = tmp_path / "vendor" / "llama.cpp"
    (vendor / "ggml" / "include").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "src").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "include" / "ggml.h").write_text(
        "enum ggml_type {\n    GGML_TYPE_F32  = 0,\n    GGML_TYPE_Q8_0 = 8,\n};\n"
        "enum ggml_op {\n    GGML_OP_NONE,\n    GGML_OP_ADD,\n    GGML_OP_MUL_MAT,\n"
        "    GGML_OP_MUL_MAT_ID,\n    GGML_OP_GLU,\n    GGML_OP_COUNT,\n};\n",
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


EPOCH = dispatch_abi.SIGNATURE_IDENTITY_EPOCH

PLAIN_MUL_MAT = {
    "schema_version": EPOCH, "op": 2, "flags": 0, "fusion": 0, "glu_op": 0,
}

MUL_MAT_ID = {
    "schema_version": EPOCH, "op": 3, "flags": 1 << 3, "fusion": 0, "glu_op": 0,
}

GLU_ALL_ZERO = {
    "schema_version": EPOCH, "op": 4, "flags": (1 << 3), "fusion": 2, "glu_op": 2,
}

ALL_FOUR_CAPS = hc.hip_capability_mask([
    hc.HipCapability.CORE_SIGNATURE_V1,
    hc.HipCapability.FUSION_X_BIAS_PRESENCE_V1,
    hc.HipCapability.FUSION_GATE_BIAS_PRESENCE_V1,
    hc.HipCapability.FUSION_X_SCALE_PRESENCE_V1,
    hc.HipCapability.FUSION_GATE_SCALE_PRESENCE_V1,
])
CORE_ONLY = hc.hip_capability_mask([hc.HipCapability.CORE_SIGNATURE_V1])


class MulMatTests(unittest.TestCase):
    def test_plain_mul_mat_requires_core_only(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            required = sc.hip_required_capabilities(PLAIN_MUL_MAT, vendor_root=vendor)
            self.assertEqual(required, CORE_ONLY)

    def test_mul_mat_id_requires_core_only(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            required = sc.hip_required_capabilities(MUL_MAT_ID, vendor_root=vendor)
            self.assertEqual(required, CORE_ONLY)

    def test_mul_mat_with_glu_op_set_is_unsupported(self):
        # Defensive per round 8: structurally unreachable today, but must
        # fail closed if a future refactor decouples fusion/glu_op.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(PLAIN_MUL_MAT, glu_op=2)
            with self.assertRaises(sc.UnsupportedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_mul_mat_with_aux_flags_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(PLAIN_MUL_MAT, flags=1 << 7)
            with self.assertRaises(sc.UnsupportedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_mul_mat_with_has_ids_set_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(PLAIN_MUL_MAT, flags=1 << 3)  # MUL_MAT should never have HAS_IDS
            with self.assertRaises(sc.UnsupportedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_mul_mat_id_without_has_ids_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(MUL_MAT_ID, flags=0)
            with self.assertRaises(sc.UnsupportedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)


class GluTests(unittest.TestCase):
    def test_glu_all_content_flags_zero_still_requires_all_four_caps(self):
        # The central distinction: a zero content bit still needs
        # authoritative producer knowledge, not an assumption of absence.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            required = sc.hip_required_capabilities(GLU_ALL_ZERO, vendor_root=vendor)
            self.assertEqual(required, ALL_FOUR_CAPS)

    def test_glu_without_has_ids_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, flags=0)
            with self.assertRaises(sc.UnsupportedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_glu_non_gate_fusion_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, fusion=1)  # BIAS, not GATE
            with self.assertRaises(sc.UnsupportedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_glu_unfusable_glu_op_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, glu_op=0)  # REGLU, never fused
            with self.assertRaises(sc.UnsupportedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_glu_x_scale_flag_set_is_unsupported_pending_hi120(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, flags=GLU_ALL_ZERO["flags"] | (1 << 9))
            with self.assertRaises(sc.UnsupportedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_glu_gate_scale_flag_set_is_unsupported_pending_hi120(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, flags=GLU_ALL_ZERO["flags"] | (1 << 10))
            with self.assertRaises(sc.UnsupportedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_glu_with_bias_content_bits_set_still_requires_all_four(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, flags=GLU_ALL_ZERO["flags"] | (1 << 7) | (1 << 8))
            required = sc.hip_required_capabilities(sig, vendor_root=vendor)
            self.assertEqual(required, ALL_FOUR_CAPS)


class GeneralTests(unittest.TestCase):
    def test_unknown_op_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(PLAIN_MUL_MAT, op=1)  # ADD
            with self.assertRaises(sc.UnsupportedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_wrong_schema_version_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(PLAIN_MUL_MAT, schema_version=1)
            with self.assertRaises(sc.UnsupportedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_non_int_field_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(PLAIN_MUL_MAT, op="not-an-int")
            with self.assertRaises(sc.UnsupportedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)


if __name__ == "__main__":
    unittest.main()
