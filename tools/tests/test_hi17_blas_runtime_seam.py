"""Runtime seam contracts for the first HI17 BLAS-1 slice."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DISPATCH = (ROOT / "src/ggml/src/ggml-cuda/hip-autotune-dispatch.cu").read_text(
    encoding="utf-8")
HEADER = (ROOT / "src/ggml/src/ggml-cuda/hip-autotune-dispatch.cuh").read_text(
    encoding="utf-8")
TYPES = (ROOT / "src/ggml/src/ggml-cuda/hip-autotune-types.h").read_text(
    encoding="utf-8")
REPLAY = (ROOT / "src/ggml/src/ggml-cuda/hip-autotune-replay.cpp").read_text(
    encoding="utf-8")
PATCH = (ROOT / "patches/0200_dispatch_hook.py").read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    left = text.index(start)
    right = text.index(end, left)
    return text[left:right]


def test_seam_resolves_applies_and_executes_in_one_ordered_path():
    assert DISPATCH.index("ggml_hip_resolve_native_blas_call(") < DISPATCH.index(
        "ggml_hip_apply_blas_plan(") < DISPATCH.index(
        "ggml_hip_execute_blas_call(")
    launch = _between(DISPATCH, "void ggml_hip_blas_launch(",
                      "size_t ggml_hip_blas_workspace(")
    assert "ggml_hip_resolve_native_blas_call" in launch
    assert "ggml_hip_apply_blas_plan" in launch
    assert "ggml_hip_execute_blas_call" in launch
    assert "ggml_hip_blas_plan_for_candidate" in launch


def test_call_contract_carries_api_shapes_precision_conversion_and_workspace():
    for field in (
        "operand_type", "accumulation_type", "output_type", "api",
        "source_a_conversion", "source_b_conversion", "output_conversion",
        "numerical_class", "strict_precision", "m", "n", "k",
        "batch_count", "lda", "ldb", "ldc", "stride_a", "stride_b",
        "stride_c", "source_a_temp_bytes", "source_b_temp_bytes",
        "output_temp_bytes", "workspace_bytes",
    ):
        assert field in HEADER
    assert "GGML_HIP_BLAS_API_SGEMM" in DISPATCH
    assert "GGML_HIP_BLAS_API_GEMMEX" in DISPATCH
    assert "GGML_HIP_BLAS_API_STRIDED_BATCHED" in DISPATCH
    assert "GGML_HIP_BLAS_API_POINTER_BATCHED" in DISPATCH


def test_native_only_plan_is_fail_closed_without_changing_variant_or_replay():
    apply = _between(DISPATCH, "bool ggml_hip_apply_blas_plan(",
                     "void ggml_hip_execute_blas_call(")
    matcher = _between(DISPATCH, "bool ggml_hip_blas_plan_matches_call(",
                       "bool ggml_hip_apply_blas_plan(")
    assert "GGML_HIP_BLAS_OPERAND_NATIVE" in matcher
    assert "GGML_HIP_BLAS_NUMERICAL_EXACT_BASELINE" in matcher
    assert "call->has_ids" in matcher
    assert "ggml_hip_variant_params" not in TYPES[TYPES.index(
        "struct ggml_hip_blas_plan_v1"):TYPES.index(
        "struct ggml_hip_launch_context")]
    assert "constexpr size_t ENT_SIZE      = ENT_SRC0_TYPE + 1" in REPLAY
    assert "ggml_hip_blas_plan_matches_call" in DISPATCH
    assert "call->plan = nullptr" in apply
    assert "call->numerical_class" in matcher
    assert "call->workspace_bytes" in DISPATCH


def test_plan_application_validates_resolved_call_facts_before_execution():
    matcher = _between(DISPATCH, "bool ggml_hip_blas_plan_matches_call(",
                       "bool ggml_hip_apply_blas_plan(")
    assert "ggml_hip_blas_call_is_well_formed" in matcher
    assert "call->has_ids" in matcher
    assert "call->numerical_class" in matcher
    assert "call.workspace_bytes == expected_workspace" in DISPATCH
    execute = _between(DISPATCH, "void ggml_hip_execute_blas_call(",
                       "bool ggml_hip_blas_can_execute(")
    assert "ggml_hip_blas_plan_matches_call(call.plan, &call)" in execute
    assert "GGML_UNUSED(call)" not in execute


def test_shared_executor_terminates_at_the_existing_vendor_forwarder():
    execute = _between(DISPATCH, "void ggml_hip_execute_blas_call(",
                       "bool ggml_hip_blas_can_execute(")
    assert "ggml_cuda_mul_mat_cublas_dispatch" in execute
    assert "ggml_hip_execute_blas_call" in DISPATCH
    assert "ggml_cuda_mul_mat_cublas_dispatch" in PATCH
