"""Source-level safety contracts for dispatch paths with upstream assertions."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DISPATCH = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-dispatch.cu"


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


if __name__ == "__main__":
    unittest.main()
