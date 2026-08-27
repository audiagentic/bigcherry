"""HI121 M3: tests for hip_required_capabilities()'s fail-closed applicability
rules -- the central "zero-and-known != zero-and-unknown" distinction."""

from __future__ import annotations

import re
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
            with self.assertRaises(sc.InvalidSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_mul_mat_with_aux_flags_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(PLAIN_MUL_MAT, flags=1 << 7)
            with self.assertRaises(sc.InvalidSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_mul_mat_with_has_ids_set_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(PLAIN_MUL_MAT, flags=1 << 3)  # MUL_MAT should never have HAS_IDS
            with self.assertRaises(sc.InvalidSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_mul_mat_id_without_has_ids_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(MUL_MAT_ID, flags=0)
            with self.assertRaises(sc.InvalidSignatureDomain):
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
            with self.assertRaises(sc.UnauditedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_glu_bias_fusion_with_nonzero_glu_op_is_invalid(self):
        # adversarial-review follow-up: fusion=BIAS means no gate tensor at
        # all, so it can never legitimately carry a nonzero glu_op (the real
        # producer only sets glu_op when fusion->gate is non-null) -- this
        # combination cannot come from a real dispatch and must be INVALID,
        # not merely unaudited (it was previously misclassified as
        # UnauditedSignatureDomain, eligible for HI136 quarantine).
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, fusion=1)  # BIAS, not GATE, but glu_op still 2
            with self.assertRaises(sc.InvalidSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_glu_unknown_fusion_kind_is_invalid(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, fusion=99)
            with self.assertRaises(sc.InvalidSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_glu_unfusable_glu_op_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, glu_op=0)  # REGLU, never fused
            with self.assertRaises(sc.InvalidSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_glu_x_scale_flag_set_is_unsupported_pending_hi120(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, flags=GLU_ALL_ZERO["flags"] | (1 << 9))
            with self.assertRaises(sc.UnauditedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_glu_gate_scale_flag_set_is_unsupported_pending_hi120(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, flags=GLU_ALL_ZERO["flags"] | (1 << 10))
            with self.assertRaises(sc.UnauditedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_glu_with_real_gate_bias_fusion_kind_still_requires_all_four(self):
        # ggml_hip_fusion_kind() (verified against real source) classifies a
        # GATE fusion that ALSO has a real bias tensor as GGML_HIP_FUSION_
        # GATE_BIAS (3), never GATE (2) -- this is the ACTUAL representation
        # a real biased GLU dispatch has, not fusion=2 with bias bits set.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, fusion=3, flags=GLU_ALL_ZERO["flags"] | (1 << 7) | (1 << 8))
            required = sc.hip_required_capabilities(sig, vendor_root=vendor)
            self.assertEqual(required, ALL_FOUR_CAPS)

    def test_glu_gate_bias_fusion_with_zero_bias_flags_is_self_contradictory(self):
        # ggml_hip_fusion_kind() cannot classify GATE_BIAS without a real
        # bias tensor present, and the content flags are set from the same
        # pointers -- fusion=GATE_BIAS with neither bias flag set is a
        # second impossible state, not a valid "zero means unknown" case.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, fusion=3)
            with self.assertRaises(sc.InvalidSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_glu_gate_fusion_with_a_bias_flag_set_is_self_contradictory(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, fusion=2, flags=GLU_ALL_ZERO["flags"] | (1 << 7))
            with self.assertRaises(sc.InvalidSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)


class GeneralTests(unittest.TestCase):
    def test_unknown_flag_bit_is_unsupported(self):
        # A bit outside hip-autotune-types.h's current ggml_hip_signature_flag
        # range must fail closed, never silently return CORE_SIGNATURE_V1 for
        # a signature that may carry meaning this rule set never audited.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(PLAIN_MUL_MAT, flags=1 << 11)
            with self.assertRaises(sc.InvalidSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_unknown_flag_bit_on_glu_is_also_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(GLU_ALL_ZERO, flags=GLU_ALL_ZERO["flags"] | (1 << 15))
            with self.assertRaises(sc.InvalidSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_unknown_op_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(PLAIN_MUL_MAT, op=1)  # ADD
            with self.assertRaises(sc.UnauditedSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_wrong_schema_version_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(PLAIN_MUL_MAT, schema_version=1)
            with self.assertRaises(sc.InvalidSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)

    def test_non_int_field_is_unsupported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            sig = dict(PLAIN_MUL_MAT, op="not-an-int")
            with self.assertRaises(sc.InvalidSignatureDomain):
                sc.hip_required_capabilities(sig, vendor_root=vendor)


class FusionKindEnumParityTests(unittest.TestCase):
    """HI121 review follow-up (P1): signature_capabilities.py hand-transcribes
    _FUSION_KIND_NONE/GATE/GATE_BIAS as bare ints rather than reading them
    from source, unlike hip_capabilities.py's declared-capabilities loader.
    This repo carries the real ggml_hip_fusion_kind enum directly (it is
    bigcherry's own authored instrumentation file, not fetched from an
    external vendor tree), so its values can be checked directly against the
    real source rather than only trusted by hand."""

    def test_fusion_kind_constants_match_real_source_enum(self):
        types_h = (
            Path(__file__).resolve().parents[3]
            / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-types.h"
        )
        text = types_h.read_text(encoding="utf-8")
        match = re.search(r"enum ggml_hip_fusion_kind \{(.*?)\};", text, re.DOTALL)
        self.assertIsNotNone(match, "could not find enum ggml_hip_fusion_kind in hip-autotune-types.h")
        real_values: dict[str, int] = {}
        for line in match.group(1).splitlines():
            entry = re.match(r"\s*(GGML_HIP_FUSION_\w+)\s*=\s*(\d+)", line)
            if entry:
                real_values[entry.group(1)] = int(entry.group(2))
        self.assertEqual(real_values["GGML_HIP_FUSION_NONE"], sc._FUSION_KIND_NONE)
        self.assertEqual(real_values["GGML_HIP_FUSION_GATE"], sc._FUSION_KIND_GATE)
        self.assertEqual(real_values["GGML_HIP_FUSION_GATE_BIAS"], sc._FUSION_KIND_GATE_BIAS)


if __name__ == "__main__":
    unittest.main()
