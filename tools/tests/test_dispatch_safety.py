"""Source-level safety contracts for dispatch paths with upstream assertions."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DISPATCH = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-dispatch.cu"
RECORD = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-record.cpp"
RECORD_HEADER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-record.h"
TUNER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"
SMI = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-smi.cpp"
SMI_HEADER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-smi.h"


class TestDispatchSafetyContracts(unittest.TestCase):
    def test_mmvq_native_moe_guard_precedes_native_return(self):
        source = DISPATCH.read_text(encoding="utf-8")
        start = source.index("bool ggml_hip_mmvq_can_execute(")
        end = source.index("void ggml_hip_mmvq_launch(", start)
        function = source[start:end]

        guard = "sig.ned[2] > MMVQ_MAX_BATCH_SIZE"
        native_return = "if (self->source_class == GGML_HIP_SOURCE_NATIVE_WRAPPER)"
        self.assertIn(guard, function)
        self.assertIn(native_return, function)
        self.assertLess(function.index(guard), function.index(native_return))

    def test_mmvq_forced_multi_token_ids_are_not_eligible(self):
        source = DISPATCH.read_text(encoding="utf-8")
        start = source.index("bool ggml_hip_mmvq_can_execute(")
        end = source.index("void ggml_hip_mmvq_launch(", start)
        function = source[start:end]

        self.assertIn(
            "sig.ned[2] > 1) {\n        return false;",
            function,
        )

    def test_tuner_numeric_environment_parsers_reject_empty_and_whitespace(self):
        source = TUNER.read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("std::isspace((unsigned char) *v)"), 3)
        self.assertGreaterEqual(source.count("*v == '\\0'"), 3)
        self.assertIn("*end != '\\0'", source)

    def test_blas_observation_telemetry_uses_existing_workspace_hook(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        record = RECORD.read_text(encoding="utf-8")
        header = RECORD_HEADER.read_text(encoding="utf-8")

        self.assertIn("native.candidate->family == GGML_HIP_FAMILY_BLAS", dispatch)
        self.assertIn("ggml_hip_blas_workspace(native.candidate, sig)", dispatch)
        self.assertIn('"ggml_cuda_mul_mat_cublas"', dispatch)
        self.assertIn('effective_api', record)
        self.assertIn('workspace_bytes', record)
        self.assertIn("const char * effective_api", header)
        self.assertIn("size_t workspace_bytes", header)

        record_block = dispatch[dispatch.index("if (mode == GGML_HIP_DISPATCH_MODE_RECORD)"):]
        self.assertLess(
            record_block.index("ggml_hip_blas_workspace"),
            record_block.index("ggml_hip_record_observation"),
        )

    def test_blas_telemetry_is_not_part_of_dispatch_identity(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        self.assertNotIn("effective_api", dispatch[:dispatch.index("// --------------------------------------------------------------------- mode")])

    def test_device_evidence_has_identity_and_explicit_drift_states(self):
        tuner = TUNER.read_text(encoding="utf-8")
        smi = SMI.read_text(encoding="utf-8")
        header = SMI_HEADER.read_text(encoding="utf-8")

        self.assertIn('\\"pci_bdf\\"', tuner)
        self.assertIn('\\"device_clock_drift\\"', tuner)
        self.assertIn('\\"status\\":\\"identity_mismatch\\"', tuner)
        self.assertIn('\\"status\\":\\"clock_unavailable\\"', tuner)
        self.assertIn("device_clock_drift_json(", tuner)
        self.assertIn("result.device_state_pre", tuner)
        self.assertIn("result.device_state_post", tuner)
        self.assertIn("identity_valid", header)
        self.assertIn("out.identity_valid = true", smi)
        self.assertIn("out.pci_bus", smi)

    def test_device_capture_does_not_emit_zero_state_when_unresolved(self):
        tuner = TUNER.read_text(encoding="utf-8")
        self.assertIn('if (!s.valid) return "{}";', tuner)
        self.assertIn('return "{\\"status\\":\\"unavailable\\"}";', tuner)


if __name__ == "__main__":
    unittest.main()
