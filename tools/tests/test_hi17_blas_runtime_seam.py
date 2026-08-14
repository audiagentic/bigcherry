"""Runtime seam contracts for the first HI17 BLAS-1 slice."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

from bigcherry.patcher import apply_patch


ROOT = Path(__file__).resolve().parents[2]
DISPATCH = (ROOT / "src/ggml/src/ggml-cuda/hip-autotune-dispatch.cu").read_text(
    encoding="utf-8")
RECORD = (ROOT / "src/ggml/src/ggml-cuda/hip-autotune-record.cpp").read_text(
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


def test_release_build_guards_do_not_reach_vendor_launcher_after_resolution_failure():
    launch = _between(DISPATCH, "void ggml_hip_blas_launch(",
                      "size_t ggml_hip_blas_workspace(")
    execute = _between(DISPATCH, "void ggml_hip_execute_blas_call(",
                       "bool ggml_hip_blas_can_execute(")
    assert "const bool resolved = ggml_hip_resolve_native_blas_call" in launch
    assert "if (!resolved)" in launch
    assert "const bool applied = ggml_hip_apply_blas_plan" in launch
    assert "if (!applied)" in launch
    assert "const bool call_is_valid = ggml_hip_blas_plan_matches_call" in execute
    assert "if (!call_is_valid)" in execute
    assert launch.index("if (!resolved)") < launch.index(
        "const ggml_hip_blas_plan_v1 * plan")
    assert launch.index("if (!applied)") < launch.index(
        "ggml_hip_execute_blas_call(*lc.ctx, call, lc)")
    assert execute.index("if (!call_is_valid)") < execute.index(
        "ggml_cuda_mul_mat_cublas_dispatch(")


def test_shared_executor_terminates_at_the_existing_vendor_forwarder():
    execute = _between(DISPATCH, "void ggml_hip_execute_blas_call(",
                       "bool ggml_hip_blas_can_execute(")
    assert "ggml_cuda_mul_mat_cublas_dispatch" in execute
    assert "ggml_hip_execute_blas_call" in DISPATCH
    assert "ggml_cuda_mul_mat_cublas_dispatch" in PATCH


def test_forced_record_binds_resolved_candidate_and_clears_metadata_tls():
    resolve = _between(
        DISPATCH,
        "ggml_hip_resolved_dispatch ggml_hip_dispatch_resolve(",
        "void ggml_hip_dispatch_launch(",
    )
    launch = _between(
        DISPATCH,
        "void ggml_hip_dispatch_launch(",
        "// ------------------------------------------------------------- entry point",
    )
    assert "bool forced_selected = false;" in resolve
    assert "if (mode != GGML_HIP_DISPATCH_MODE_RECORD)" in resolve
    assert 'resolved.candidate, "forced"' in resolve
    assert 'resolved.candidate, "native"' in resolve
    assert "const ggml_hip_candidate_descriptor * resolved_candidate" in RECORD
    assert '\\"resolved_candidate\\":\\"%s\\"' in RECORD
    assert '\\"resolution_source\\":\\"%s\\"' in RECORD
    assert '"forced"' in resolve
    assert "resolved.from_cache = false;" in resolve
    assert "} else {" in resolve
    assert "ggml_hip_record_end_observation();" in launch
    assert launch.index("effective.launch(&effective, lc);") < launch.index(
        "ggml_hip_record_end_observation();")
    assert "g_has_active_key = false;" in RECORD


def test_runtime_experiment_options_are_per_call_and_non_persistent():
    options = _between(
        HEADER,
        "enum ggml_hip_blas_experiment_compute_v1",
        "struct ggml_hip_blas_call_v1",
    )
    assert "ggml_hip_blas_execution_options_v1" in options
    assert "GGML_HIP_BLAS_EXPERIMENT_COMPUTE_F16" in options
    assert "GGML_HIP_BLAS_EXPERIMENT_OUTPUT_TEMPORARY_TO_F32" in options

    execute = _between(DISPATCH, "void ggml_hip_execute_blas_call(",
                       "bool ggml_hip_blas_can_execute(")
    experiment = _between(DISPATCH, "bool ggml_hip_blas_experiment_options(",
                          "void ggml_hip_execute_blas_call(")
    assert "GGML_HIP_BLAS_EXPERIMENT" in experiment
    assert "options_ptr" in execute
    assert "ggml_cuda_mul_mat_cublas_dispatch(" in execute
    assert "call.has_ids" in experiment
    assert "call.strict_precision" in experiment
    assert "GGML_HIP_BLAS_API_SGEMM" in experiment
    assert "call.workspace_bytes" in experiment

    # Runtime experiment controls must not leak into the serialized plan or
    # replay identity.
    plan = TYPES[TYPES.index("struct ggml_hip_blas_plan_v1"):TYPES.index(
        "struct ggml_hip_launch_context")]
    assert "execution_options" not in plan
    assert "GGML_HIP_BLAS_EXPERIMENT" not in REPLAY
    assert "const ggml_hip_blas_execution_options_v1 * options" in PATCH
    assert "const void * execution_options" in PATCH


def test_already_applied_vendor_hooks_have_idempotent_migrations():
    assert "blas-experiment-f16-output-route-migrate" in PATCH
    assert "blas-metadata-state-migrate-malformed-execution-options" in PATCH
    assert "blas-metadata-state-migrate-execution-options" in PATCH
    assert "blas-metadata-emission-migrate-execution-options" in PATCH
    assert "options_valid" in PATCH
    assert 'bigcherry_execution_options = "native"' in PATCH
    assert "bigcherry_execution_options = \"f16_temp\"" in PATCH


def test_metadata_state_migrations_handle_old_malformed_and_repeat_shapes():
    spec = importlib.util.spec_from_file_location(
        "bigcherry_hi17_patch", ROOT / "patches/0200_dispatch_hook.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    migration_ids = {
        "blas-metadata-state-migrate-malformed-execution-options",
        "blas-metadata-state-migrate-execution-options",
    }
    migration_patch = replace(
        module.PATCH,
        edits=tuple(edit for edit in module.PATCH.edits if edit.id in migration_ids),
    )
    fresh_patch = replace(
        module.PATCH,
        edits=tuple(edit for edit in module.PATCH.edits if edit.id in {
            "blas-metadata-state",
            *migration_ids,
        }),
    )
    old = (
        "#ifdef GGML_HIP_AUTOTUNE_RECORD\n"
        "    const char * bigcherry_effective_call_api = \"unknown\";\n"
        "#endif\n"
    )
    repaired = (
        "#ifdef GGML_HIP_AUTOTUNE_RECORD\n"
        "    const char * bigcherry_effective_call_api    const char * "
        "bigcherry_execution_options = \"native\";\n"
        " = \"unknown\";\n"
        "#endif\n"
    )
    for source in (old, repaired):
        view = {migration_patch.path: source}
        result = apply_patch(migration_patch, ROOT, dry_run=True, texts=view)
        assert result.ok
        assert view[migration_patch.path].count(
            'const char * bigcherry_execution_options = "native";') == 1
        assert view[migration_patch.path].index(
            'bigcherry_effective_call_api = "unknown";') < view[
                migration_patch.path].index(
                    'bigcherry_execution_options = "native";')
        before = view[migration_patch.path]
        repeated = apply_patch(migration_patch, ROOT, dry_run=True, texts=view)
        assert repeated.ok
        assert view[migration_patch.path] == before

    fresh = (
        "    GGML_ASSERT(ggml_is_contiguous(dst));\n"
    )
    view = {fresh_patch.path: fresh}
    result = apply_patch(fresh_patch, ROOT, dry_run=True, texts=view)
    assert result.ok
    assert view[fresh_patch.path].count(
        'const char * bigcherry_execution_options = "native";') == 1
    before = view[fresh_patch.path]
    repeated = apply_patch(fresh_patch, ROOT, dry_run=True, texts=view)
    assert repeated.ok
    assert view[fresh_patch.path] == before
