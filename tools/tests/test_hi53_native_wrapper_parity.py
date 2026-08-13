"""HI53 offline parity contracts for MMVQ/MUL_MAT_ID safety.

These tests deliberately inspect both the pinned upstream source and the
BigCherry dispatch source.  They are not a substitute for GPU execution; they
make it impossible to claim the hardware matrix without first proving that the
two sides still describe the same safety boundary.
"""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_MMVQ = (
    ROOT / "vendor" / "llama.cpp" / "ggml" / "src" / "ggml-cuda" / "mmvq.cu"
)
UPSTREAM_MMVQ_HEADER = UPSTREAM_MMVQ.with_suffix(".cuh")
DISPATCH = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-dispatch.cu"
# HI53 is closed; keep the acceptance contract attached to its completed
# evidence rather than a path that disappears when the item transitions.
PLAN = ROOT / "docs" / "planning" / "completed" / "hip-autotune" / "HI53.md"


def _function(source: str, signature: str, end_marker: str) -> str:
    start = source.index(signature)
    end = source.index(end_marker, start)
    return source[start:end]


class HI53NativeWrapperParityTests(unittest.TestCase):
    def test_pinned_upstream_defines_the_asserted_boundary_and_architecture_limits(self):
        source = UPSTREAM_MMVQ.read_text(encoding="utf-8")
        header = UPSTREAM_MMVQ_HEADER.read_text(encoding="utf-8")
        self.assertIn("#define MMVQ_MAX_BATCH_SIZE 8", header)
        self.assertIn("GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);", source)
        self.assertIn("int get_mmvq_mmid_max_batch(ggml_type type, int cc)", source)
        for selector in (
            "GGML_CUDA_CC_IS_RDNA4",
            "GGML_CUDA_CC_IS_RDNA3",
            "GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)",
        ):
            self.assertIn(selector, source)

    def test_native_guard_precedes_native_wrapper_return_and_uses_expert_batch(self):
        source = DISPATCH.read_text(encoding="utf-8")
        function = _function(
            source,
            "bool ggml_hip_mmvq_can_execute(",
            "void ggml_hip_mmvq_launch(",
        )
        ids_guard = "(sig.flags & GGML_HIP_SIG_HAS_IDS) != 0"
        oversized_guard = "sig.ned[2] > mmvq_mmid_max"
        native_return = "if (self->source_class == GGML_HIP_SOURCE_NATIVE_WRAPPER)"
        self.assertIn(ids_guard, function)
        self.assertIn(oversized_guard, function)
        self.assertIn("get_mmvq_mmid_max_batch(", function)
        self.assertIn("sig.src0_type", function)
        self.assertIn("ggml_cuda_info().devices[device].cc", function)
        self.assertIn(native_return, function)
        self.assertLess(function.index(oversized_guard), function.index(native_return))
        # `ned[1]` is the dense output width; using it here would not protect
        # the upstream MUL_MAT_ID assertion, whose `ne12` is the expert batch.
        self.assertNotIn("sig.ned[1] > MMVQ_MAX_BATCH_SIZE", function)

    def test_forced_multi_token_ids_are_rejected_before_variant_checks(self):
        source = DISPATCH.read_text(encoding="utf-8")
        function = _function(
            source,
            "bool ggml_hip_mmvq_can_execute(",
            "void ggml_hip_mmvq_launch(",
        )
        multi_token = "sig.ned[2] > 1"
        self.assertIn(multi_token, function)
        self.assertIn("return false;", function[function.index(multi_token):])
        # A forced candidate must not reach geometry/type validation or launch
        # after the semantic MUL_MAT_ID rejection.
        self.assertLess(
            function.index(multi_token),
            function.index("self->variant.src0_type"),
        )

    def test_native_selection_carries_architecture_and_shape_coverage_inputs(self):
        source = DISPATCH.read_text(encoding="utf-8")
        selector = _function(
            source,
            "ggml_hip_native_selection ggml_hip_native_select(",
            "// ------------------------------------------------------------ process cache",
        )
        self.assertIn("const int cc", selector)
        self.assertIn("const int64_t mmid_batch = dst->ne[2]", selector)
        self.assertIn("get_mmvq_mmid_max_batch(src0->type, cc)", selector)
        self.assertIn("mmid_batch <= MMVQ_MAX_BATCH_SIZE", selector)
        self.assertIn("mmid_batch <= mmvq_mmid_max", selector)

    def test_plan_requires_all_architectures_and_boundary_shapes(self):
        plan = PLAN.read_text(encoding="utf-8")
        for architecture in ("gfx1030", "gfx1100", "gfx1201"):
            self.assertIn(architecture, plan)
        for shape in ("ned[2]=8", "ned[2]=9", "MMVQ_MAX_BATCH_SIZE", "above the MMVQ boundary"):
            self.assertIn(shape, plan)
        for evidence in ("selected family", "forced eligibility", "measurement status", "assertion"):
            self.assertIn(evidence, plan)


if __name__ == "__main__":
    unittest.main()
