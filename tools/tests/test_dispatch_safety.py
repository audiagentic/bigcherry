"""Source-level safety contracts for dispatch paths with upstream assertions."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DISPATCH = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-dispatch.cu"
RECORD = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-record.cpp"
RECORD_HEADER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-record.h"
TUNER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"
TUNER_HEADER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cuh"
DISPATCH_PATCH = ROOT / "patches" / "0200_dispatch_hook.py"
MMQ_PATCH = ROOT / "patches" / "0300_mmq_forced_j.py"
SMI = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-smi.cpp"
SMI_HEADER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-smi.h"


class TestDispatchSafetyContracts(unittest.TestCase):
    def test_workspace_accounting_uses_requested_size_and_preserves_pool_size(self):
        patch = (ROOT / "patches" / "0900_pool_workspace_metrics.py").read_text(
            encoding="utf-8"
        )

        requested_member = patch.index("size_t bc_requested_size = 0;")
        requested_assignment = patch.index(
            "this->bc_requested_size = size * sizeof(T);"
        )
        note_alloc = patch.index(
            "pool->bc_workspace_note_alloc(this->bc_requested_size);"
        )
        note_free = patch.index(
            "pool->bc_workspace_note_free(this->bc_requested_size);"
        )
        pool_free = patch.index(r"pool->free\(ptr, actual_size\);")

        self.assertLess(requested_member, requested_assignment)
        self.assertLess(requested_assignment, note_alloc)
        # Accounting follows the request, while the real allocator continues
        # to release the size returned by its best-fit allocation.
        self.assertNotEqual(note_free, pool_free)
        self.assertIn("bc_requested_size", patch)
        self.assertIn("actual_size", patch)

    def test_workspace_clear_rebase_isolated_before_warmup_only(self):
        tuner = TUNER.read_text(encoding="utf-8")
        start = tuner.index("bool time_candidate(")
        end = tuner.index("// HI34:", start)
        function = tuner[start:end]

        clear = function.index("bc_workspace_clear_cache();")
        warmup = function.index("for (int i = 0; i < warmup; ++i)")
        sync = function.index("hipStreamSynchronize(lc.stream)")
        rebase = function.index("bc_workspace_reset_peak();")
        self.assertLess(clear, warmup)
        self.assertLess(warmup, sync)
        self.assertLess(sync, rebase)
        self.assertIn("if (workspace_ctx != nullptr && isolate_workspace)", function)
        self.assertIn("peak >= workspace_baseline", function)

        # Final/confirmation timing is intentionally interleaved and must not
        # opt into the isolated cache-clear/rebase protocol.
        final_calls = tuner[
            tuner.index("for (int round = 0; round < config.final_samples;") :
        ]
        self.assertNotIn("&ctx, true", final_calls)
        confirmation = tuner[
            tuner.index("const int rounds = std::max(config.confirmation_samples") :
        ]
        self.assertNotIn("&ctx, true", confirmation)

    def test_workspace_protocol_trace_is_gated_and_labels_confirmation(self):
        tuner = TUNER.read_text(encoding="utf-8")
        self.assertIn("GGML_HIP_WORKSPACE_TRACE", tuner)
        self.assertIn('trace_workspace_event(stage, "clear_cache"', tuner)
        metrics = tuner[tuner.index("#ifdef GGML_HIP_WORKSPACE_METRICS") :]
        self.assertIn('trace_workspace_event(stage, "rebase_peak"', metrics)
        self.assertIn("run_counterbalanced_round(", tuner)
        # Slice B0: the confirmation round states its flush mode explicitly
        # (config read, never a hidden default).
        self.assertIn(
            'result.launches_per_sample, config.pre_sample_mode, "confirmation")', tuner
        )

    def test_mmvq_native_moe_guard_precedes_native_return(self):
        source = DISPATCH.read_text(encoding="utf-8")
        start = source.index("bool ggml_hip_mmvq_can_execute(")
        end = source.index("void ggml_hip_mmvq_launch(", start)
        function = source[start:end]

        guard = "sig.ned[2] > mmvq_mmid_max"
        native_return = "if (self->source_class == GGML_HIP_SOURCE_NATIVE_WRAPPER)"
        self.assertIn(guard, function)
        self.assertIn("get_mmvq_mmid_max_batch(", function)
        self.assertIn("(ggml_type) sig.src0_type", function)
        self.assertIn("ggml_cuda_info().devices[device].cc", function)
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

    def test_mmq_native_wrapper_dense_safety_precedes_native_return(self):
        # HI71: the MMQ native-wrapper branch must compute its own natural J
        # and run it through the same dense-safety check as a forced
        # candidate, not exempt itself with an unconditional `return true`.
        source = DISPATCH.read_text(encoding="utf-8")
        start = source.index("bool ggml_hip_mmq_can_execute(")
        end = source.index("void ggml_hip_mmq_launch(", start)
        function = source[start:end]

        native_branch = "if (self->source_class == GGML_HIP_SOURCE_NATIVE_WRAPPER)"
        native_j_call = "ggml_cuda_mmq_native_j_best("
        self.assertIn(native_branch, function)
        self.assertIn(native_j_call, function)
        self.assertIn("if (native_j == 0)", function)
        self.assertIn("return false;", function)

        # the native-J lookup and its safety check must be inside the
        # native-wrapper branch, i.e. after native_branch starts.
        self.assertGreater(function.index(native_j_call), function.index(native_branch))
        native_slice = function[function.index(native_branch):]
        self.assertIn(
            "return ggml_cuda_mmq_variant_is_eligible(",
            native_slice,
        )

    def test_mmq_type_is_supported_uses_config_check_not_shape_aware_entry(self):
        # HI71: ggml_cuda_mmq_type_is_supported has no real dispatch signature
        # to test, so it must call the config-only ggml_cuda_mmq_config_is_eligible
        # directly rather than passing a sentinel ncols_max into the
        # shape-aware ggml_cuda_mmq_variant_is_eligible.
        source = MMQ_PATCH.read_text(encoding="utf-8")
        start = source.index("bool ggml_cuda_mmq_type_is_supported(")
        end = source.index("\n}", start)
        function = source[start:end]

        self.assertIn("ggml_cuda_mmq_config_is_eligible(", function)
        self.assertNotIn("ggml_cuda_mmq_variant_is_eligible(", function)

    def test_mmq_variant_is_eligible_checks_tail_padding(self):
        # HI71: dense-shape-aware eligibility must compare the launch's
        # required tail-padding columns against what upstream's own
        # J-selection search (ggml_cuda_mmq_get_J_max) would have allocated.
        source = MMQ_PATCH.read_text(encoding="utf-8")
        start = source.index("bool ggml_cuda_mmq_variant_is_eligible(")
        end = source.index("\n}", start)
        function = source[start:end]

        self.assertIn("ncols_max > 0", function)
        self.assertIn("required_tail", function)
        self.assertIn("available_tail", function)
        self.assertIn("ggml_cuda_mmq_get_J_max(", function)
        self.assertIn("required_tail <= available_tail", function)

    def test_tuner_numeric_environment_parsers_reject_empty_and_whitespace(self):
        source = TUNER.read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("std::isspace((unsigned char) *v)"), 3)
        self.assertGreaterEqual(source.count("*v == '\\0'"), 3)
        self.assertIn("*end != '\\0'", source)

    def test_tuner_event_destruction_has_one_checked_cleanup_seam(self):
        """Every created event must use the checked, non-short-circuit seam."""
        source = TUNER.read_text(encoding="utf-8")
        helper_start = source.index("static bool hip_event_destroy_checked(")
        helper_end = source.index("bool time_candidate(", helper_start)
        helper = source[helper_start:helper_end]
        managed = source[:helper_start] + source[helper_end:]

        raw_destroy = re.compile(r"""(?<!["'])\bhipEventDestroy\s*\(""")
        self.assertEqual(len(raw_destroy.findall(helper)), 1)
        self.assertEqual(raw_destroy.findall(managed), [])
        self.assertRegex(
            source,
            r"const bool start_ok = hip_event_destroy_checked\(start,.*?;\s*"
            r"const bool stop_ok = hip_event_destroy_checked\(stop,.*?;\s*"
            r"return start_ok && stop_ok;",
        )
        self.assertRegex(
            source,
            r"const bool a_ok = hip_event_destroy_checked\(a,.*?;\s*"
            r"const bool b_ok = hip_event_destroy_checked\(b,.*?;\s*"
            r"return a_ok && b_ok;",
        )
        self.assertNotRegex(
            source,
            r"hip_event_destroy_checked\([^;]*\)\s*&&",
        )

    def test_host_sync_calibration_never_caches_incomplete_lifecycle(self):
        source = TUNER.read_text(encoding="utf-8")
        start = source.index("double host_sync_overhead_us(")
        end = source.index("// E3: max()", start)
        function = source[start:end]

        # HI64: cache storage moved from two module-global doubles to a
        # per-device PerDeviceState<HostSyncOverheadCache> (see the class's
        # own comment) -- the reset-on-incomplete-lifecycle contract this
        # test checks is unchanged, just spelled as a .store() call now.
        self.assertIn("g_host_sync_overhead.store(HostSyncOverheadCache{});", function)
        self.assertIn("if (!destroy_ok || samples.empty())", function)
        self.assertIn("if (!std::isfinite(overhead) || overhead < 0.0)", function)
        self.assertLess(
            function.index("const bool destroy_ok = destroy_events();"),
            function.index("g_host_sync_overhead.store(HostSyncOverheadCache{overhead, true});"),
        )

    def test_blas_observation_telemetry_uses_existing_workspace_hook(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        record = RECORD.read_text(encoding="utf-8")
        header = RECORD_HEADER.read_text(encoding="utf-8")

        self.assertIn("native.candidate->family == GGML_HIP_FAMILY_BLAS", dispatch)
        self.assertIn("ggml_hip_blas_workspace(native.candidate, sig)", dispatch)
        self.assertIn('"ggml_cuda_mul_mat_cublas"', dispatch)
        self.assertIn("effective_api", record)
        self.assertIn("effective_call_api", record)
        self.assertIn("workspace_bytes", record)
        self.assertIn("const char * effective_api", header)
        self.assertIn("size_t workspace_bytes", header)

        record_block = dispatch[
            dispatch.index("if (mode == GGML_HIP_DISPATCH_MODE_RECORD)") :
        ]
        self.assertLess(
            record_block.index("ggml_hip_blas_workspace"),
            record_block.index("ggml_hip_record_observation"),
        )

    def test_blas_observation_payload_is_additive_and_observation_only(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        record = RECORD.read_text(encoding="utf-8")
        header = RECORD_HEADER.read_text(encoding="utf-8")
        patch = DISPATCH_PATCH.read_text(encoding="utf-8")
        for field in (
            "operand_a_type",
            "operand_b_type",
            "output_type",
            "accumulation_type",
            "source_a_conversion",
            "source_b_conversion",
            "output_conversion",
            "requested_precision",
            "effective_call_api",
            "effective_provider",
            "effective_backend",
            "source_a_temp_bytes",
            "source_b_temp_bytes",
            "output_temp_bytes",
            "execution_options",
        ):
            self.assertIn(field, header)
            self.assertIn(field, record)
        self.assertIn("ggml_hip_record_blas_metadata", record)
        self.assertIn("ggml_hip_record_blas_metadata", patch)
        self.assertNotIn(
            "blas_metadata",
            dispatch[
                : dispatch.index(
                    "// --------------------------------------------------------------------- mode"
                )
            ],
        )
        self.assertNotIn("bigcherry_blas_metadata", dispatch)

    def test_blas_effective_call_api_covers_native_branches_without_dispatch_changes(
        self,
    ):
        patch = (ROOT / "patches" / "0200_dispatch_hook.py").read_text(encoding="utf-8")
        record = RECORD.read_text(encoding="utf-8")

        for api in (
            "cublasSgemm",
            "cublasGemmEx",
            "cublasGemmStridedBatchedEx",
            "cublasGemmBatchedEx",
        ):
            self.assertIn(f'_record_api("{api}")', patch)
        self.assertNotIn('mode="insert_before",\n            text=_record_api', patch)
        self.assertIn(
            'rationale="the completed native single-matrix F32 BLAS call"', patch
        )
        self.assertIn(
            'rationale="the completed native strided-batched BLAS call"', patch
        )
        self.assertIn("thread_local PairKey g_active_key", record)
        self.assertIn("effective_call_api = api", record)
        # This field is observation-only; it must not appear in the resolver.
        self.assertNotIn("effective_call_api", DISPATCH.read_text(encoding="utf-8"))

    def test_blas_telemetry_is_not_part_of_dispatch_identity(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        self.assertNotIn(
            "effective_api",
            dispatch[
                : dispatch.index(
                    "// --------------------------------------------------------------------- mode"
                )
            ],
        )

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

    def test_device_capture_does_not_emit_zero_for_a_single_failed_metric(self):
        # dev-gpt-agent deep review (2026-08-25): `valid == true` only proves
        # device identity resolved -- it does NOT prove every individual RSMI
        # metric call succeeded. Before this fix, a metric whose RSMI call
        # failed independently (e.g. power unsupported on this ASIC while
        # clocks read fine) serialized as a plain numeric 0, indistinguishable
        # from a genuinely idle/cold reading -- the exact ambiguity this
        # struct's own header docstring claims never happens. Each metric now
        # carries its own *_valid flag and prints JSON `null`, not 0, when
        # that flag is false.
        header = SMI_HEADER.read_text(encoding="utf-8")
        smi = SMI.read_text(encoding="utf-8")
        tuner = TUNER.read_text(encoding="utf-8")
        for field in ("sclk_valid", "mclk_valid", "edge_temp_valid",
                      "junction_temp_valid", "power_valid", "busy_valid"):
            self.assertIn(field, header)
            self.assertIn(field, smi)
        self.assertIn("device_state_metric_u64(", tuner)
        self.assertIn("device_state_metric_u32(", tuner)
        self.assertIn('if (!metric_valid) return "null";', tuner)
        # The clock-drift falsification consumer must also treat a failed
        # clock read as unavailable, not silently proceed with a stale 0.
        self.assertIn("!pre.sclk_valid || !post.sclk_valid", tuner)

    def test_hi60_counterbalanced_round_is_observation_only_and_fail_closed(self):
        tuner = TUNER.read_text(encoding="utf-8")
        helper_start = tuner.index("CounterbalancedRound run_counterbalanced_round(")
        helper_end = tuner.index("void merge_retime_status(", helper_start)
        helper = tuner[helper_start:helper_end]

        self.assertIn("const bool reverse", tuner)
        self.assertIn(
            "(offset + candidates.size() - 1 - position) % candidates.size()", helper
        )
        self.assertIn("const bool reverse_complete = run_order(!reverse", helper)
        self.assertIn("first_observation.identity_mismatch", helper)
        self.assertIn(
            "out.gpu_us.assign(candidates.size(), std::numeric_limits<double>::quiet_NaN())",
            helper,
        )
        self.assertIn("out.status = RetimeStatus::unresolved", helper)
        self.assertIn("(void) hipGetLastError();", helper)
        self.assertIn("smi_enabled && complete", helper)
        self.assertIn("bool measurement_failure = false", tuner)
        self.assertIn(
            "if (result.measurement_failure || !smi_capture_enabled())", tuner
        )
        self.assertIn("g_smi_runtime_disabled", tuner)
        self.assertIn("disable_smi_after_measurement_failure();", tuner)
        self.assertIn("const bool smi_enabled = smi_capture_enabled();", tuner)
        self.assertNotIn("sclk_mhz", helper[helper.index("run_order") :])
        self.assertGreaterEqual(tuner.count("run_counterbalanced_round("), 4)

    def test_hi60_retime_fields_are_serialized_but_not_identity_inputs(self):
        tuner = TUNER.read_text(encoding="utf-8")
        self.assertIn('\\"clock_drift_rounds\\"', tuner)
        self.assertIn('\\"reverse_retime_attempts\\"', tuner)
        self.assertIn('\\"reverse_retime_passed\\"', tuner)
        self.assertIn('\\"retime_status\\"', tuner)
        self.assertIn('\\"measurement_failure\\"', tuner)
        self.assertIn('result.measurement_failure ? "true" : "false"', tuner)
        self.assertIn('result.retime_status == "unresolved"', tuner)
        self.assertIn("clock drift retime unresolved; run rejected", tuner)
        signature = (
            ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-signature.cpp"
        ).read_text(encoding="utf-8")
        self.assertNotIn("retime_status", signature)
        self.assertNotIn("clock_drift_rounds", signature)

    def test_tuner_first_encounters_are_single_flight_through_publish(self):
        """The source-level concurrency contract prevents duplicate winners.

        HI64 follow-up (2026-08-22): the single-flight lock moved from this
        impl function to the public ggml_hip_tuner_resolve() wrapper (see
        TestHi64CrossDevicePoisonLeak.
        test_single_flight_lock_held_across_impl_call_and_failure_readback
        for that half) -- a second thread's failure->success replacement
        landing between the impl returning and a caller reading
        measurement_failure back out would otherwise report a stale value
        for what actually happened on THIS invocation. This test now covers
        only the impl's own internal ordering, which is unaffected by where
        the lock lives."""
        tuner = TUNER.read_text(encoding="utf-8")
        resolve_start = tuner.index(
            "static const ggml_hip_candidate_descriptor * ggml_hip_tuner_resolve_impl("
        )
        # Bounded to the impl only -- NOT through ggml_hip_tuner_flush(),
        # which would also swallow the public wrapper (declared between the
        # two) and its own, deliberately-present single_flight_lock.
        resolve_end = tuner.index(
            "ggml_hip_tuner_resolution ggml_hip_tuner_resolve(", resolve_start
        )
        resolve = tuner[resolve_start:resolve_end]

        self.assertIn("std::mutex g_single_flight_mutex;", tuner)
        self.assertNotIn("single_flight_lock", resolve)
        lookup = resolve.index("g_results.find(dispatch_digest)")
        first_measurement = resolve.index("const ggml_hip_tuner_config & config")
        final_publish = resolve.rindex("record_result(dispatch_digest, result);")
        final_return = resolve.index("return result.winner;", final_publish)

        self.assertLess(lookup, first_measurement)
        self.assertLess(final_publish, final_return)
        self.assertIn("g_results.emplace(dispatch_digest, result);", tuner)

    def test_final_measurement_failures_cannot_reuse_screening_medians(self):
        """A failed final stage must reject stale screening evidence."""
        tuner = TUNER.read_text(encoding="utf-8")
        final_stage = tuner.index("// --- final measurement")
        ranking = tuner.index("// --- noise canary", final_stage)
        final_text = tuner[final_stage:ranking]
        self.assertIn("m->reason = GGML_HIP_REJECT_LAUNCH_FAILED;", final_text)
        self.assertIn("m->measured = false;", final_text)
        self.assertIn("native final measurement failed; run rejected", final_text)
        self.assertIn("!native_m->measured", final_text)

    def test_measurement_failure_poison_suppresses_later_gpu_work(self):
        tuner = TUNER.read_text(encoding="utf-8")
        # HI64: g_tuner_poisoned moved from a single std::atomic<bool> to a
        # PerDeviceState<bool> (per-device poison scoping) -- every .load()/
        # .store() call site below is unchanged by that move.
        self.assertIn("PerDeviceState<bool> g_tuner_poisoned(false);", tuner)
        failure_helper = tuner.index(
            "static void disable_smi_after_measurement_failure()"
        )
        failure_end = tuner.index("static bool smi_capture_enabled()", failure_helper)
        self.assertIn("g_tuner_poisoned.store(true", tuner[failure_helper:failure_end])
        time_start = tuner.index("bool time_candidate(")
        time_end = tuner.index("// HI34: derive", time_start)
        self.assertIn("g_tuner_poisoned.load", tuner[time_start:time_end])
        screening = tuner[
            tuner.index("// --- screening") : tuner.index("// --- final measurement")
        ]
        self.assertIn(
            "screening measurement failed; tuning experiment poisoned", screening
        )
        final_stage = tuner[
            tuner.index("// --- final measurement") : tuner.index(
                "if (result.retime_status", tuner.index("// --- final measurement")
            )
        ]
        self.assertIn("if (!measured.complete)", final_stage)
        self.assertIn("break;", final_stage)
        self.assertIn(
            "tuning experiment poisoned; later measurements suppressed", tuner
        )
        canary = tuner[tuner.index("// --- noise canary") : tuner.index("// E3: rank")]
        self.assertIn("if (!measured.complete)", canary)
        confirmation = tuner[
            tuner.index("const int rounds =") : tuner.index(
                "// Confirmation is also a promotion gate"
            )
        ]
        self.assertIn(
            "confirmation measurement failed; tuning experiment poisoned", confirmation
        )

    def test_overhead_and_correctness_copy_failures_share_terminal_poison(self):
        tuner = TUNER.read_text(encoding="utf-8")
        overhead = tuner.index("double host_sync_overhead_us(")
        overhead_end = tuner.index("double effective_us_of(", overhead)
        self.assertIn(
            "if (g_tuner_poisoned.load(std::memory_order_relaxed))",
            tuner[overhead:overhead_end],
        )
        self.assertIn(
            "disable_smi_after_measurement_failure();", tuner[overhead:overhead_end]
        )
        for operation in (
            '"hipMemcpyAsync(reference)"',
            '"hipStreamSynchronize(reference)"',
            '"hipMemcpyAsync(candidate)"',
            '"hipStreamSynchronize(candidate)"',
        ):
            self.assertIn(operation, tuner)
        self.assertIn("record_measurement_failure(native_m", tuner)
        self.assertIn("record_measurement_failure(m", tuner)

    def test_fault_injection_is_opt_in_and_traced_before_poisoning(self):
        tuner = TUNER.read_text(encoding="utf-8")
        helper = tuner.index("static bool inject_test_measurement_failure(")
        helper_end = tuner.index("static bool smi_capture_enabled()", helper)
        contract = tuner[helper:helper_end]
        self.assertIn("GGML_HIP_TUNE_TEST_FAIL_CANDIDATE", contract)
        self.assertIn("GGML_HIP_TUNE_TEST_FAIL_STAGE", contract)
        self.assertIn("std::strcmp", contract)
        self.assertIn("consumed.exchange", contract)
        attempt = tuner.index(
            "trace_launch_attempt(candidate ? candidate->stable_name : nullptr, stage);"
        )
        injection = tuner.index("inject_test_measurement_failure", attempt)
        self.assertLess(attempt, injection)
        self.assertIn("disable_smi_after_measurement_failure();", contract)

    def test_ex02_quarantine_bypass_is_explicitly_opt_in(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        # HI71: ggml_hip_mmq_can_execute's native-wrapper branch now has its
        # OWN earlier "return ggml_cuda_mmq_variant_is_eligible(" call (its
        # dense-safety check on the native candidate's own J), so the
        # quarantine slice's end boundary must be anchored to the FIRST such
        # call *after* the quarantine comment, not the first in the file.
        quarantine_start = dispatch.index("// EX02 quarantine")
        quarantine = dispatch[
            quarantine_start : dispatch.index(
                "return ggml_cuda_mmq_variant_is_eligible(", quarantine_start
            )
        ]
        self.assertIn("GGML_HIP_TUNE_TEST_DISABLE_EX02_QUARANTINE", quarantine)
        self.assertIn('std::strcmp(ex02_bypass, "1") == 0', quarantine)
        self.assertIn("ex02_candidate && !ex02_test_bypass", quarantine)
        self.assertIn(
            "EX02 quarantine bypass enabled for diagnostic testing", quarantine
        )

    def test_first_candidate_attempt_is_durable_and_identified(self):
        tuner = TUNER.read_text(encoding="utf-8")
        resolve = tuner[
            tuner.index(
                "static const ggml_hip_candidate_descriptor * ggml_hip_tuner_resolve_impl("
            ) :
        ]
        self.assertIn("open_tuning_journal_once(result.hardware_digest);", resolve)
        self.assertLess(
            resolve.index("open_tuning_journal_once(result.hardware_digest);"),
            resolve.index("const ggml_hip_dispatch_scope no_reentry;"),
        )
        self.assertIn("thread_local ggml_hip_digest g_trace_signature_digest", tuner)
        self.assertIn(
            "thread_local ggml_hip_dispatch_signature_v1 g_trace_signature", tuner
        )
        self.assertIn('\\"signature\\":\\"', tuner)
        self.assertIn("ggml_hip_digest_hex(g_trace_signature_digest)", tuner)
        self.assertIn('\\"signature_json\\":', tuner)
        self.assertIn("ggml_hip_signature_json(g_trace_signature, true)", tuner)
        self.assertIn("g_trace_signature = sig", tuner)
        self.assertIn('signature_json +\n        ",\\"stage\\":\\""', tuner)
        self.assertIn('\\"stage\\":\\"', tuner)
        self.assertIn("std::string(protocol_stage", tuner)
        self.assertIn(
            "trace_launch_attempt(candidate ? candidate->stable_name : nullptr, stage);",
            tuner,
        )

    def _tune_branch(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        # "if (mode == GGML_HIP_DISPATCH_MODE_TUNE)" also appears in the
        # earlier mode-validation function; anchor on the occurrence that
        # immediately precedes the tuner call inside the resolver.
        resolve_call = dispatch.index("ggml_hip_tuner_resolve(ctx, sig, hw")
        start = dispatch.rindex(
            "if (mode == GGML_HIP_DISPATCH_MODE_TUNE)", 0, resolve_call
        )
        end = dispatch.index("#ifdef GGML_HIP_DISPATCH_REPLAY", start)
        return dispatch[start:end]

    def test_hi67_tune_mode_never_installs_a_winner_into_the_live_binding(self):
        """F1: tune mode measures/selects/records; the workload stays native.

        A selected winner must not reach real dispatch from the tuner path:
        native-relative acceptance is a screening invariant, not production
        correctness proof (RV08). Winners arrive only via replay, after
        external promotion against CPU-reference evidence.
        """
        branch = self._tune_branch()
        # The resolver is still called (measurement/selection/recording are
        # intact) -- what is gone is the install into the binding.
        self.assertIn("ggml_hip_tuner_resolve(ctx, sig, hw", branch)
        self.assertNotIn("binding.candidate  = winner;", branch)
        self.assertNotIn("binding.variant    = winner->variant;", branch)
        self.assertNotIn("binding.from_cache = true;", branch)

    def test_hi67_tune_mode_keeps_native_and_reports_nonnative_selections(self):
        """The divergence is observable: a non-native selection logs once per
        signature that dispatch stays native, while a native selection is
        silent (no log storm on long runs)."""
        branch = self._tune_branch()
        guard = "tuning.winner != nullptr && tuning.winner != native.candidate"
        call = "ggml_hip_log_tune_kept_native("
        self.assertIn(guard, branch)
        self.assertIn(call, branch)
        self.assertLess(branch.index(guard), branch.index(call))

    def test_hi67_stays_native_log_is_once_per_signature(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        start = dispatch.index("void ggml_hip_log_tune_kept_native(")
        end = dispatch.index("#endif // GGML_HIP_AUTOTUNE", start)
        function = dispatch[start:end]
        self.assertIn("static std::set<std::string> logged;", function)
        insert = function.index("logged.insert(key).second")
        early_return = function.index("return;", insert)
        emit = function.index("GGML_LOG_INFO(", insert)
        self.assertLess(early_return, emit)

    def test_tune_journal_result_keeps_replay_identity_digests(self):
        tuner = TUNER.read_text(encoding="utf-8")
        summary = tuner[tuner.index("std::string journal_result_summary(") :]
        self.assertIn("signature_digest", summary)
        self.assertIn("hardware_digest", summary)
        self.assertLess(summary.index("signature_digest"), summary.index("winner"))


class TestHi64PerDevicePoisonScoping(unittest.TestCase):
    """HI64 (RV49 F6 scope extension, 2026-08-17 GPT adjudication): a
    transient fault on one HIP device must not poison tuning on every other
    device sharing the process. g_tuner_poisoned/g_smi_runtime_disabled/
    g_host_sync_overhead moved from process-global flags to a
    PerDeviceState<T> template keyed by the current hipGetDevice() value."""

    def test_per_device_state_class_keys_by_current_device(self):
        tuner = TUNER.read_text(encoding="utf-8")
        start = tuner.index("template <typename T>\nclass PerDeviceState {")
        end = tuner.index("};", tuner.index("std::unordered_map<int, T> values_;", start)) + 2
        cls = tuner[start:end]
        self.assertIn("static int current_device()", cls)
        self.assertIn("hipGetDevice(&device)", cls)
        self.assertIn("std::lock_guard<std::mutex> lock(mutex_);", cls)
        self.assertIn("std::unordered_map<int, T> values_;", cls)
        # store() and load() both resolve the CURRENT device internally --
        # no caller anywhere passes a device in, so every existing
        # .load()/.store() call site needed zero changes.
        load_start = cls.index("T load(")
        load_end = cls.index("void store(")
        self.assertIn("current_device()", cls[load_start:load_end])

    def test_no_time_based_or_implicit_auto_clear(self):
        # GPT requirement 2: no time-based expiry, no clear-after-one-
        # success. PerDeviceState never removes a key once set -- store()
        # only ever inserts/overwrites, load() never deletes, and grepping
        # the whole file confirms no .erase(/.clear( call exists on any of
        # the three PerDeviceState instances.
        tuner = TUNER.read_text(encoding="utf-8")
        for symbol in ("g_tuner_poisoned", "g_smi_runtime_disabled", "g_host_sync_overhead"):
            self.assertNotIn(f"{symbol}.erase(", tuner)
            self.assertNotIn(f"{symbol}.clear(", tuner)

    def test_tuner_never_calls_hip_device_reset(self):
        # GPT requirement 3: the tuner must never call hipDeviceReset()
        # itself -- could invalidate allocations owned by llama.cpp. Strip
        # line comments first -- this file's own HI64 documentation
        # mentions the symbol by name to explain why it is absent.
        tuner = TUNER.read_text(encoding="utf-8")
        code_lines = [
            line for line in tuner.splitlines() if not line.strip().startswith("//")
        ]
        self.assertNotIn("hipDeviceReset(", "\n".join(code_lines))

    def test_three_globals_are_per_device_scoped(self):
        tuner = TUNER.read_text(encoding="utf-8")
        self.assertIn("PerDeviceState<bool> g_tuner_poisoned(false);", tuner)
        self.assertIn("PerDeviceState<bool> g_smi_runtime_disabled(false);", tuner)
        self.assertIn(
            "PerDeviceState<HostSyncOverheadCache> g_host_sync_overhead(HostSyncOverheadCache{});",
            tuner,
        )

    def test_host_sync_overhead_header_report_reads_through_per_device_load(self):
        # ggml_hip_tuner_flush()'s JSON header must report the calling
        # device's cache, not a stale value some other device happened to
        # compute first under the old process-global scheme.
        tuner = TUNER.read_text(encoding="utf-8")
        flush = tuner[tuner.index("void ggml_hip_tuner_flush(") :]
        flush = flush[: flush.index("\n}\n", flush.index("kind\\\":\\\"header"))]
        self.assertIn("const HostSyncOverheadCache sync_overhead = g_host_sync_overhead.load();", flush)
        self.assertIn("sync_overhead.us, sync_overhead.valid", flush)


class TestHi29TransformRecording(unittest.TestCase):
    """HI29: transform-attempt / transform-gap recording, written to a
    separate <GGML_HIP_DISPATCH_DB>.transforms.jsonl alongside the ordinary
    measurements.jsonl, guarded by GGML_HIP_ROUTING_TRANSFORM. Kept
    structurally separate from Result -- offline agent-driven pattern
    analysis across many signatures, not per-signature dispatch identity."""

    def test_record_structs_and_accumulators_declared(self):
        tuner = TUNER.read_text(encoding="utf-8")
        self.assertIn("struct TransformAttemptRecord {", tuner)
        self.assertIn("struct TransformGapRecord {", tuner)
        self.assertIn("struct TransformTriedEntry {", tuner)
        self.assertIn("std::vector<TransformAttemptRecord> g_transform_attempts;", tuner)
        self.assertIn("std::vector<TransformGapRecord>     g_transform_gaps;", tuner)
        self.assertIn("std::mutex                          g_transform_mutex;", tuner)

    def test_record_functions_are_mutex_guarded(self):
        tuner = TUNER.read_text(encoding="utf-8")
        attempt_fn = tuner.index("void ggml_hip_record_transform_attempt(")
        attempt_end = tuner.index("}", attempt_fn)
        attempt_body = tuner[attempt_fn:attempt_end]
        self.assertIn("std::lock_guard<std::mutex> lock(g_transform_mutex);", attempt_body)
        self.assertIn("g_transform_attempts.push_back(record);", attempt_body)

        gap_fn = tuner.index("void ggml_hip_record_transform_gap(")
        gap_end = tuner.index("}", gap_fn)
        gap_body = tuner[gap_fn:gap_end]
        self.assertIn("std::lock_guard<std::mutex> lock(g_transform_mutex);", gap_body)
        self.assertIn("g_transform_gaps.push_back(record);", gap_body)

    def test_recording_machinery_is_guarded_behind_routing_transform_flag(self):
        # Everything HI29 added must be compiled out when
        # GGML_HIP_ROUTING_TRANSFORM is off -- same discipline as HI27/HI28's
        # own guard (standards 12.2: a production replay build carries no
        # symbol it does not dispatch through).
        tuner = TUNER.read_text(encoding="utf-8")
        start = tuner.index("#ifdef GGML_HIP_ROUTING_TRANSFORM\n// bigcherry (HI29)")
        end = tuner.index("#endif // GGML_HIP_ROUTING_TRANSFORM", start)
        block = tuner[start:end]
        self.assertIn("struct TransformAttemptRecord {", block)
        self.assertIn("struct TransformGapRecord {", block)
        self.assertIn("void ggml_hip_record_transform_attempt(", block)
        self.assertIn("void ggml_hip_record_transform_gap(", block)

    def test_flush_writes_a_separate_transforms_file_with_atomic_crash_safety(self):
        tuner = TUNER.read_text(encoding="utf-8")
        flush = tuner[tuner.index("void ggml_hip_tuner_flush(") :]
        transform_block_start = flush.index("#ifdef GGML_HIP_ROUTING_TRANSFORM")
        transform_block = flush[transform_block_start:]
        self.assertIn('std::string transforms_path = std::string(path) + ".transforms.jsonl";', transform_block)
        self.assertIn("ggml_hip_atomic_begin(transforms_path.c_str(), transforms_atomic)", transform_block)
        self.assertIn("ggml_hip_atomic_commit(transforms_atomic)", transform_block)
        # Separate atomic file from measurements.jsonl -- one crash-safety
        # unit each, so a transforms-file write failure cannot corrupt or
        # block the (already-committed, by this point) measurements file.
        measurements_commit = flush.index("ggml_hip_atomic_commit(measurements_atomic)")
        self.assertLess(measurements_commit, transform_block_start)

    def test_gap_record_serializes_every_tried_transform_with_its_reason(self):
        tuner = TUNER.read_text(encoding="utf-8")
        flush = tuner[tuner.index("void ggml_hip_tuner_flush(") :]
        gap_loop_start = flush.index("for (const TransformGapRecord & r : g_transform_gaps)")
        gap_loop = flush[gap_loop_start : gap_loop_start + 1200]
        self.assertIn('transformations_tried', gap_loop)
        self.assertIn("r.tried[i].transform_id", gap_loop)
        self.assertIn("r.tried[i].rejection_reason.c_str()", gap_loop)

    def test_attempt_record_carries_both_timings_and_correctness_metrics(self):
        tuner = TUNER.read_text(encoding="utf-8")
        struct_start = tuner.index("struct TransformAttemptRecord {")
        struct_end = tuner.index("\n};", struct_start)
        struct_body = tuner[struct_start:struct_end]
        for field in ("original_us", "transformed_us", "improvement_pct", "nmse", "max_abs_error"):
            self.assertIn(field, struct_body)

    def test_measurements_jsonl_result_row_carries_the_winning_transform(self):
        # HI31 prerequisite: the production measurements.jsonl artifact
        # (not just the diagnostic journal) must carry which transform (if
        # any) the FINAL winner reached its candidate through, or
        # tune_promotion.py's promoted-winners output and replay_cache.py's
        # exporter would have no way to know a v5 replay entry needs a
        # transform_id at all.
        tuner = TUNER.read_text(encoding="utf-8")
        flush = tuner[tuner.index("void ggml_hip_tuner_flush(") :]
        result_row = flush[: flush.index("for (const auto & entry : g_results)") + 4000]
        self.assertIn('\\"winner_transform\\":\\"%s\\"', result_row)
        self.assertIn('\\"winner_transform_id\\":%d', result_row)
        self.assertIn("winner_transform_name(r)", result_row)
        self.assertIn("winner_transform_id(r)", result_row)
        # And the helper itself must return "" when the feature is compiled
        # out, not merely omit the field -- the JSON schema must stay
        # identical across both build configurations.
        helper_start = tuner.index("static const char * winner_transform_name(")
        helper_end = tuner.index("\n}", helper_start)
        helper_body = tuner[helper_start:helper_end]
        self.assertIn('return "";', helper_body)


class TestHi31DispatchTransformIntegration(unittest.TestCase):
    """HI31: Binding/ggml_hip_resolved_dispatch carry a transform pointer,
    the replay resolution path re-validates a transformed entry against the
    real transformed signature (never the original), and the dispatch-time
    launch routes through ggml_hip_transform_launch() for a transformed
    binding while the ordinary path stays exactly as it was."""

    def test_resolved_dispatch_and_binding_both_carry_a_transform_pointer(self):
        header = (ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-types.h").read_text(encoding="utf-8")
        struct_start = header.index("struct ggml_hip_resolved_dispatch {")
        struct_end = header.index("\n};", struct_start)
        struct_body = header[struct_start:struct_end]
        self.assertIn("const ggml_hip_routing_transformation * transform = nullptr;", struct_body)
        self.assertIn("struct ggml_hip_routing_transformation;", header[: struct_start])

        dispatch = DISPATCH.read_text(encoding="utf-8")
        binding_start = dispatch.index("struct Binding {")
        binding_end = dispatch.index("\n};", binding_start)
        binding_body = dispatch[binding_start:binding_end]
        self.assertIn("const ggml_hip_routing_transformation * transform = nullptr;", binding_body)

    def test_replay_resolution_revalidates_against_transformed_signature(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        helper_start = dispatch.index("static bool transformed_candidate_still_valid(")
        helper_end = dispatch.index("\n}", helper_start)
        helper = dispatch[helper_start:helper_end]
        self.assertIn("transform->equivalence_verified", helper)
        self.assertIn("ggml_hip_transform_signature_is_eligible(sig)", helper)
        self.assertIn("transform->can_apply(sig)", helper)
        self.assertIn("transform->apply(lc, &ctx, &transformed_sig, &transformed_lc)", helper)
        self.assertIn("candidate->family != transform->target_family", helper)
        # The final can_execute check must run against the TRANSFORMED
        # signature, never the original -- the whole point of this helper.
        self.assertIn("candidate->can_execute(candidate, transformed_sig, hw)", helper)
        self.assertNotIn("candidate->can_execute(candidate, sig, hw)", helper)

    def test_replay_block_calls_the_shared_transform_validator(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        replay_start = dispatch.index("if (mode == GGML_HIP_DISPATCH_MODE_REPLAY) {")
        replay_end = dispatch.index("\n    }\n#endif", replay_start)
        replay_block = dispatch[replay_start:replay_end]
        self.assertIn("ggml_hip_transform_find((ggml_hip_transform_id) transform_id)", replay_block)
        self.assertIn("transformed_candidate_still_valid(winner, winner_transform, sig, lc, hw)", replay_block)
        self.assertIn("binding.transform  = winner_transform;", replay_block)

    def test_blacklist_safety_net_uses_transform_aware_revalidation(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        net_start = dispatch.index("// A stored winner that cannot run on this hardware")
        net_end = dispatch.index("g_bindings.emplace(dispatch_digest, binding);", net_start)
        net = dispatch[net_start:net_end]
        self.assertIn("if (binding.transform != nullptr) {", net)
        self.assertIn(
            "transformed_candidate_still_valid(binding.candidate, binding.transform, sig, lc, hw)",
            net,
        )
        self.assertIn("binding.transform  = nullptr;", net)

    def test_thread_cache_hit_and_final_resolved_propagate_transform(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        resolve = dispatch[dispatch.index("ggml_hip_resolved_dispatch ggml_hip_dispatch_resolve(") :]
        self.assertIn("resolved.transform  = thread_binding.transform;", resolve)
        self.assertIn("resolved.transform   = binding.transform;", resolve)

    def test_dispatch_launch_routes_a_transformed_binding_through_transform_launch(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        launch = dispatch[dispatch.index("void ggml_hip_dispatch_launch(") :]
        launch = launch[: launch.index("// ------------------------------------------------------------- entry point")]
        self.assertIn("if (bound.transform != nullptr) {", launch)
        self.assertIn(
            "bound.transform->apply(lc, &ctx, /*out_sig=*/nullptr, &transformed_lc)",
            launch,
        )
        self.assertIn(
            "ggml_hip_transform_launch(bound.transform, bound.candidate,\n"
            "                                      bound.variant, &ctx, transformed_lc);",
            launch,
        )
        self.assertIn("launch_native_fallback_after_transform_failure(bound.transform, lc);", launch)
        # Ordinary path must still be reachable and unchanged in shape.
        self.assertIn("effective.launch(&effective, lc);", launch)

    def test_apply_failure_fallback_never_uses_ggml_assert(self):
        # GGML_ASSERT always aborts in this codebase (not compiled out under
        # NDEBUG) -- using it for "transform apply() unexpectedly failed"
        # would turn one bad transform into a crashed inference process,
        # exactly what the fail-closed-to-native design exists to avoid.
        dispatch = DISPATCH.read_text(encoding="utf-8")
        fn_start = dispatch.index("static void launch_native_fallback_after_transform_failure(")
        fn_end = dispatch.index("\n}", fn_start)
        fn_body = dispatch[fn_start:fn_end]
        self.assertNotIn("GGML_ASSERT", fn_body)
        self.assertIn("ggml_hip_native_select(*lc.ctx, lc.src0, lc.src1, lc.ids, lc.dst)", fn_body)
        self.assertIn("logged.insert(transform->name).second", fn_body)

    def test_tune_mode_never_installs_a_transform_into_the_live_binding(self):
        # HI67 slice 1's contract (tune mode stays native) must still hold
        # after HI31: the TUNE branch never writes into `binding` at all,
        # transformed or otherwise -- it only logs what it would have won.
        dispatch = DISPATCH.read_text(encoding="utf-8")
        resolve = dispatch[dispatch.index("ggml_hip_resolved_dispatch ggml_hip_dispatch_resolve(") :]
        tune_start = resolve.index("if (mode == GGML_HIP_DISPATCH_MODE_TUNE) {")
        tune_end = resolve.index("\n    }\n#endif", tune_start)
        tune_block = resolve[tune_start:tune_end]
        self.assertNotIn("binding.candidate", tune_block)
        self.assertNotIn("binding.transform", tune_block)
        self.assertIn("ggml_hip_tuner_resolve(", tune_block)


class TestHi67StrictForcedCandidate(unittest.TestCase):
    """HI67: a CPU-reference correctness producer that asks for a specific
    candidate via GGML_HIP_FORCE_CANDIDATE must be able to PROVE it actually
    executed. Without GGML_HIP_FORCE_CANDIDATE_STRICT=1, an unregistered or
    ineligible forced candidate silently falls back to normal resolution --
    the process exit code and stderr give no signal, so a correctness
    producer could record a valid-looking comparison for one that never
    happened. Strict mode fails closed instead (GGML_ABORT) and, on success,
    emits a per-dispatch machine-readable marker."""

    def _forced_candidate_struct(self, dispatch: str) -> str:
        start = dispatch.index("struct ForcedCandidate {")
        end = dispatch.index("\n};\n", start)
        return dispatch[start:end]

    def _resolve_forced_block(self, dispatch: str) -> str:
        resolve = dispatch[dispatch.index("ggml_hip_resolved_dispatch ggml_hip_dispatch_resolve(") :]
        start = resolve.index("if (const auto & forced = ForcedCandidate::instance();")
        end = resolve.index("\n    }\n", start) + len("\n    }\n")
        return resolve[start:end]

    def test_strict_flag_read_from_env_alongside_stable_name(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        struct = self._forced_candidate_struct(dispatch)
        self.assertIn("bool strict = false;", struct)
        self.assertIn('getenv("GGML_HIP_FORCE_CANDIDATE_STRICT") != nullptr', struct)

    def test_unregistered_candidate_aborts_in_strict_mode(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        struct = self._forced_candidate_struct(dispatch)
        not_found = struct[struct.index("if (!fc.candidate) {") :]
        not_found = not_found[: not_found.index("} else {")]
        self.assertIn("if (fc.strict) {", not_found)
        self.assertIn("GGML_ABORT(", not_found)
        # The non-strict warning-and-disable path must still exist, unchanged,
        # for ordinary (non-correctness-evidence) manual force-candidate use.
        self.assertIn("GGML_LOG_WARN(", not_found)

    def test_ineligible_candidate_aborts_in_strict_mode_not_falls_through(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        block = self._resolve_forced_block(dispatch)
        strict_abort = block.index("} else if (forced.strict) {")
        fallback_warn = block.index("} else {\n            // Not eligible")
        self.assertLess(strict_abort, fallback_warn)
        strict_branch = block[strict_abort:fallback_warn]
        self.assertIn("GGML_ABORT(", strict_branch)
        # The strict-mode abort branch must be checked BEFORE the silent
        # fallback branch, not after -- else the fallback would still run
        # first for a strict-mode caller.
        fallback_branch = block[fallback_warn:]
        self.assertIn("using normal resolution instead", fallback_branch)

    def test_successful_strict_selection_emits_per_dispatch_marker(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        block = self._resolve_forced_block(dispatch)
        can_execute = block.index("if (forced.candidate->can_execute(forced.candidate, sig, hw)) {")
        strict_marker = block.index("if (forced.strict) {", can_execute)
        marker_end = block.index("}", strict_marker)
        marker_body = block[strict_marker:marker_end]
        self.assertIn(
            'GGML_LOG_INFO("BIGCHERRY_FORCE_CANDIDATE_EXECUTED stable_name=%s\\n",',
            marker_body,
        )
        self.assertIn("forced.stable_name)", marker_body)
        # Marker emission must happen before the mode-dependent early return,
        # not be skippable by it.
        early_return = block.index("if (mode != GGML_HIP_DISPATCH_MODE_RECORD) {", can_execute)
        self.assertLess(strict_marker, early_return)


class TestHi64CrossDevicePoisonLeak(unittest.TestCase):
    """HI64 (2026-08-22, real dual-XTX hardware finding): a fatal
    measurement failure on one device was silently blocking tuning on a
    second, healthy, identical GPU. Root cause was NOT PerDeviceState --
    it was the process-global, device-ordinal-free g_bindings/g_results
    caches (deliberately shared so identical GPUs can reuse a winner,
    standards 10.2) treating a device-local failure as if it were a
    portable resolution. Fix: only a non-failed result may populate those
    shared caches; a failed one is retained solely as fallback evidence."""

    def test_resolution_struct_carries_measurement_failure(self):
        header = TUNER_HEADER.read_text(encoding="utf-8")
        struct_start = header.index("struct ggml_hip_tuner_resolution {")
        struct_end = header.index("};", struct_start)
        struct_body = header[struct_start:struct_end]
        self.assertIn("const ggml_hip_candidate_descriptor * winner = nullptr;", struct_body)
        self.assertIn("bool measurement_failure = false;", struct_body)

    def test_public_resolve_returns_the_resolution_struct(self):
        header = TUNER_HEADER.read_text(encoding="utf-8")
        self.assertIn("ggml_hip_tuner_resolution ggml_hip_tuner_resolve(", header)
        # The old declaration returning a bare pointer must be gone from the
        # header -- only the internal .cu-file impl keeps that signature now.
        self.assertNotIn(
            "const ggml_hip_candidate_descriptor * ggml_hip_tuner_resolve(", header
        )

    def test_impl_cache_hit_excludes_failed_results(self):
        tuner = TUNER.read_text(encoding="utf-8")
        impl_start = tuner.index(
            "static const ggml_hip_candidate_descriptor * ggml_hip_tuner_resolve_impl("
        )
        impl_end = tuner.index("ggml_hip_tuner_resolution ggml_hip_tuner_resolve(", impl_start)
        impl = tuner[impl_start:impl_end]
        lookup = impl.index("g_results.find(dispatch_digest)")
        cache_hit_guard = impl.index(
            "found != g_results.end() && !found->second.measurement_failure", lookup
        )
        cache_hit_return = impl.index("return found->second.winner;", cache_hit_guard)
        self.assertLess(lookup, cache_hit_guard)
        self.assertLess(cache_hit_guard, cache_hit_return)

    def test_public_wrapper_reports_measurement_failure_from_g_results(self):
        tuner = TUNER.read_text(encoding="utf-8")
        wrapper_start = tuner.index("ggml_hip_tuner_resolution ggml_hip_tuner_resolve(")
        wrapper_end = tuner.index("void ggml_hip_tuner_flush(", wrapper_start)
        wrapper = tuner[wrapper_start:wrapper_end]
        self.assertIn("ggml_hip_tuner_resolve_impl(ctx, sig, hw, dispatch_digest, native, lc)", wrapper)
        self.assertIn("g_results.find(dispatch_digest)", wrapper)
        self.assertIn(
            "found != g_results.end() && found->second.measurement_failure", wrapper
        )

    def test_record_result_replaces_failure_with_success_never_the_reverse(self):
        tuner = TUNER.read_text(encoding="utf-8")
        start = tuner.index("void record_result(")
        end = tuner.index("const char * reason_name(", start)
        record = tuner[start:end]
        self.assertIn("const auto found = g_results.find(dispatch_digest);", record)
        self.assertIn("if (found == g_results.end()) {", record)
        self.assertIn("g_results.emplace(dispatch_digest, result);", record)
        replace_guard = record.index(
            "found->second.measurement_failure && !result.measurement_failure"
        )
        replace_assign = record.index("found->second = result;", replace_guard)
        self.assertGreater(replace_assign, replace_guard)
        # The inverse condition (a good result getting clobbered by a later
        # failure) must never appear as a guard in this function.
        self.assertNotIn("!found->second.measurement_failure && result.measurement_failure", record)

    def test_dispatcher_derives_cacheable_flag_from_measurement_failure(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        decl = dispatch.index("bool process_binding_cacheable = true;")
        tune_start = dispatch.index("if (mode == GGML_HIP_DISPATCH_MODE_TUNE) {", decl)
        emplace_guard = dispatch.index("if (process_binding_cacheable) {", tune_start)
        emplace_call = dispatch.index("g_bindings.emplace(dispatch_digest, binding);", emplace_guard)
        self.assertLess(decl, tune_start)
        self.assertLess(tune_start, emplace_guard)
        self.assertLess(emplace_guard, emplace_call)

        tune_block = dispatch[tune_start:emplace_guard]
        self.assertIn("process_binding_cacheable = false;", tune_block)  # capture-skip site
        self.assertIn("process_binding_cacheable = !tuning.measurement_failure;", tune_block)

    def test_capture_skip_marks_binding_not_cacheable(self):
        dispatch = DISPATCH.read_text(encoding="utf-8")
        tune_start = dispatch.index("if (mode == GGML_HIP_DISPATCH_MODE_TUNE) {")
        capture_branch_start = dispatch.index("ggml_hip_stream_is_capturing(lc.stream)", tune_start)
        capture_branch_end = dispatch.index("} else {", capture_branch_start)
        capture_branch = dispatch[capture_branch_start:capture_branch_end]
        self.assertIn("ggml_hip_warn_tuning_skipped_under_capture();", capture_branch)
        self.assertIn("process_binding_cacheable = false;", capture_branch)

    def test_thread_local_binding_insert_is_never_gated_by_process_flag(self):
        # A device-local measurement FAILURE must still populate the
        # per-device/thread cache (so THIS device/thread doesn't keep
        # retrying a signature it already knows is poisoned) -- only the
        # PROCESS-GLOBAL, cross-device g_bindings publish is gated on
        # process_binding_cacheable. The thread cache has its own,
        # narrower gate (thread_binding_cacheable) for a different case --
        # see test_capture_skip_also_clears_thread_binding_cacheable.
        dispatch = DISPATCH.read_text(encoding="utf-8")
        insert_call = dispatch.index("g_thread_bindings.insert(ctx.device, sig, binding);")
        guard_start = dispatch.rindex("if (", 0, insert_call)
        guard_line = dispatch[guard_start:insert_call]
        self.assertIn("mode != GGML_HIP_DISPATCH_MODE_RECORD", guard_line)
        self.assertIn("thread_binding_cacheable", guard_line)
        self.assertNotIn("process_binding_cacheable", guard_line)

    def test_capture_skip_also_clears_thread_binding_cacheable(self):
        # GPT follow-up review (2026-08-22): the process-global fix alone
        # left a real gap -- caching native into g_thread_bindings during a
        # capture-time skip would permanently starve that exact (thread,
        # device, signature) triple of ever being measured, even long after
        # capture ends, since capture is not a permanent condition the way
        # a measurement failure is.
        dispatch = DISPATCH.read_text(encoding="utf-8")
        tune_start = dispatch.index("if (mode == GGML_HIP_DISPATCH_MODE_TUNE) {")
        capture_branch_start = dispatch.index("ggml_hip_stream_is_capturing(lc.stream)", tune_start)
        capture_branch_end = dispatch.index("} else {", capture_branch_start)
        capture_branch = dispatch[capture_branch_start:capture_branch_end]
        self.assertIn("process_binding_cacheable = false;", capture_branch)
        self.assertIn("thread_binding_cacheable  = false;", capture_branch)

    def test_measurement_failure_does_not_clear_thread_binding_cacheable(self):
        # The opposite of the capture case: a fatal measurement failure is
        # a PERMANENT condition for this device/process, so it must leave
        # thread_binding_cacheable alone (still true) -- unlike the
        # capture-skip branch, nothing in the non-capturing tune branch may
        # assign thread_binding_cacheable at all.
        dispatch = DISPATCH.read_text(encoding="utf-8")
        tune_start = dispatch.index("if (mode == GGML_HIP_DISPATCH_MODE_TUNE) {")
        capture_check = dispatch.index("ggml_hip_stream_is_capturing(lc.stream)", tune_start)
        else_start = dispatch.index("} else {", capture_check)
        else_end = dispatch.index("#ifdef GGML_HIP_DISPATCH_REPLAY", else_start)
        non_capture_branch = dispatch[else_start:else_end]
        self.assertIn("process_binding_cacheable = !tuning.measurement_failure;", non_capture_branch)
        self.assertNotIn("thread_binding_cacheable", non_capture_branch)

    def test_single_flight_lock_held_across_impl_call_and_failure_readback(self):
        # GPT follow-up review (2026-08-22): a race existed where the impl's
        # own single_flight_lock released before the public wrapper's
        # g_results re-check ran, letting a second thread's failure->success
        # replacement land in between -- making the wrapper report
        # measurement_failure=false for an invocation that actually failed.
        # Fix: the lock moves to the wrapper and covers both the impl call
        # and the re-check; the impl itself must not acquire it at all.
        tuner = TUNER.read_text(encoding="utf-8")
        impl_start = tuner.index(
            "static const ggml_hip_candidate_descriptor * ggml_hip_tuner_resolve_impl("
        )
        impl_end = tuner.index("ggml_hip_tuner_resolution ggml_hip_tuner_resolve(", impl_start)
        impl = tuner[impl_start:impl_end]
        self.assertNotIn("single_flight_lock", impl)

        wrapper = tuner[impl_end:]
        wrapper_end = wrapper.index("void ggml_hip_tuner_flush(")
        wrapper = wrapper[:wrapper_end]
        lock_decl = wrapper.index(
            "std::unique_lock<std::mutex> single_flight_lock(g_single_flight_mutex);"
        )
        impl_call = wrapper.index("ggml_hip_tuner_resolve_impl(ctx, sig, hw, dispatch_digest, native, lc)")
        readback = wrapper.index("g_results.find(dispatch_digest)")
        self.assertLess(lock_decl, impl_call)
        self.assertLess(impl_call, readback)

    def test_hardware_and_dispatch_identity_remain_device_ordinal_free(self):
        # The correct fix keeps identical GPUs sharing a portable key --
        # adding device ordinal to the identity would silently throw away
        # standards 10.2 (a real regression path GPT's design explicitly
        # warned against) instead of fixing the actual defect.
        dispatch = DISPATCH.read_text(encoding="utf-8")
        self.assertNotIn("ggml_hip_hardware_digest(hw, ctx.device)", dispatch)
        self.assertNotIn("ggml_hip_dispatch_digest(hardware_digest, signature_digest, ctx.device", dispatch)


class TestHi64ElapsedTimeRetry(unittest.TestCase):
    """HI64 (2026-08-23, real Windows/WDDM hardware finding, RX 7900 GRE):
    an unblocked tuning sweep reproduced deterministically across two
    independent runs, poisoning the whole process on the FIRST signature
    after 11/11 candidates measured cleanly. GGML_HIP_TUNE_TRACE_ATTEMPTS
    tracing found zero hip_ok()-logged HIP API failures anywhere across all
    runs -- every checked HIP call succeeded -- narrowing the cause to one
    of the two silent numeric-sanity checks in time_candidate()'s timed-
    sample loop (a non-finite/non-positive hipEventElapsedTime() reading for
    a sub-millisecond kernel), not a driver crash/TDR event. Fix: a bounded
    retry, WITHIN one timed sample, for that specific silent-branch class
    only -- every hip_ok()-checked call keeps its original immediately-fatal
    behavior on a genuine API failure."""

    def test_config_field_declared_with_default(self):
        header = TUNER_HEADER.read_text(encoding="utf-8")
        self.assertIn(
            "int    elapsed_time_retry_max  = 2;       // GGML_HIP_TUNE_ELAPSED_RETRY",
            header,
        )

    def test_env_var_parsed_and_bounded(self):
        # HI99: this knob's env-parsing is now generated from
        # GGML_HIP_TUNER_CONFIG_FIELDS rather than a standalone hand-written
        # getenv() block -- check its row in the macro table instead.
        header = TUNER_HEADER.read_text(encoding="utf-8")
        self.assertIn('elapsed_time_retry_max,', header)
        self.assertIn('"GGML_HIP_TUNE_ELAPSED_RETRY"', header)
        row_idx = header.index('F(INT,    elapsed_time_retry_max,')
        row_end = header.index(") \\", row_idx)
        row = header[row_idx:row_end]
        self.assertIn('"elapsed_time_retry_max"', row)
        self.assertIn('"GGML_HIP_TUNE_ELAPSED_RETRY"', row)
        self.assertIn("0,", row)
        self.assertIn("10", row)

    def test_hip_ok_checked_calls_in_the_retry_loop_remain_immediately_fatal(self):
        # A genuine HIP API failure (anything hip_ok() itself catches) must
        # never be retried -- only the two silent numeric-sanity checks are.
        tuner = TUNER.read_text(encoding="utf-8")
        loop_start = tuner.index("const int max_elapsed_time_attempts =")
        loop_end = tuner.index("if (!sample_ok) {", loop_start)
        loop_body = tuner[loop_start:loop_end]
        for hip_call in (
            'hip_ok(hipEventRecord(start, lc.stream), "hipEventRecord(start)")',
            'hip_ok(hipEventRecord(stop, lc.stream), "hipEventRecord(stop)")',
            'hip_ok(hipEventSynchronize(stop), "hipEventSynchronize(stop)")',
            'hip_ok(hipEventElapsedTime(&ms, start, stop), "hipEventElapsedTime")',
        ):
            call_at = loop_body.index(hip_call)
            # Each of these sits inside its own `if (!hip_ok(...)) { ... return
            # false; }` block, unlike the two numeric-sanity checks below.
            following = loop_body[call_at : call_at + 400]
            self.assertIn("return false;", following)

    def test_numeric_sanity_checks_are_retryable_not_immediately_fatal(self):
        tuner = TUNER.read_text(encoding="utf-8")
        loop_start = tuner.index("const int max_elapsed_time_attempts =")
        loop_end = tuner.index("if (!sample_ok) {", loop_start)
        loop_body = tuner[loop_start:loop_end]
        ms_check = loop_body.index("if (!std::isfinite(ms) || ms <= 0.0f) {")
        self.assertIn(
            "continue;", loop_body[ms_check : ms_check + 120]
        )
        us_check = loop_body.index(
            "if (!std::isfinite(candidate_us) || candidate_us <= 0.0) {"
        )
        self.assertIn(
            "continue;", loop_body[us_check : us_check + 120]
        )
        # Neither retryable branch may itself call
        # disable_smi_after_measurement_failure() -- only exhausting every
        # attempt (the `if (!sample_ok)` block, outside this slice) may.
        self.assertNotIn(
            "disable_smi_after_measurement_failure",
            loop_body[ms_check : us_check + 200],
        )

    def test_retry_bound_derives_from_config_not_a_bare_constant(self):
        tuner = TUNER.read_text(encoding="utf-8")
        self.assertIn(
            "const ggml_hip_tuner_config & retry_config = ggml_hip_tuner_get_config();\n"
            "        const int max_elapsed_time_attempts =\n"
            "            1 + std::max(0, retry_config.elapsed_time_retry_max);",
            tuner,
        )

    def test_backoff_field_declared_with_default(self):
        header = TUNER_HEADER.read_text(encoding="utf-8")
        self.assertIn(
            "double elapsed_time_retry_backoff_us = 2000.0;  "
            "// GGML_HIP_TUNE_ELAPSED_RETRY_BACKOFF_US",
            header,
        )

    def test_backoff_env_var_parsed(self):
        # HI99: generated from GGML_HIP_TUNER_CONFIG_FIELDS, not a standalone
        # hand-written getenv() block.
        header = TUNER_HEADER.read_text(encoding="utf-8")
        row_idx = header.index("F(DOUBLE, elapsed_time_retry_backoff_us,")
        row_end = header.index(") \\", row_idx)
        row = header[row_idx:row_end]
        self.assertIn('"GGML_HIP_TUNE_ELAPSED_RETRY_BACKOFF_US"', row)
        self.assertIn("elapsed_time_retry_backoff_us", row)

    def test_retry_sleeps_before_reattempting_not_immediately(self):
        # HI64 (2026-08-23, third real-hardware round): the retry was
        # originally back-to-back with zero delay. A real wall-clock sleep
        # must happen for attempt > 0, scaled by attempt number (linear
        # backoff), before the next hipEventRecord(start) -- not before the
        # first attempt, and not unconditionally regardless of config.
        tuner = TUNER.read_text(encoding="utf-8")
        loop_start = tuner.index("const int max_elapsed_time_attempts =")
        loop_end = tuner.index("if (!sample_ok) {", loop_start)
        loop_body = tuner[loop_start:loop_end]
        attempt_gt_zero = loop_body.index("if (attempt > 0) {")
        sleep_call = loop_body.index(
            "std::this_thread::sleep_for(std::chrono::microseconds(", attempt_gt_zero
        )
        record_start = loop_body.index(
            'hip_ok(hipEventRecord(start, lc.stream), "hipEventRecord(start)")',
            attempt_gt_zero,
        )
        self.assertLess(attempt_gt_zero, sleep_call)
        self.assertLess(sleep_call, record_start)
        self.assertIn(
            "(int64_t) (attempt * retry_config.elapsed_time_retry_backoff_us)",
            loop_body[sleep_call : sleep_call + 200],
        )
        # Gated: 0 (or negative, though the config parser already rejects
        # that) must skip the sleep entirely rather than sleep_for(0), which
        # is a real (if tiny) syscall on every retry.
        self.assertIn(
            "retry_config.elapsed_time_retry_backoff_us > 0.0",
            loop_body[attempt_gt_zero:sleep_call],
        )

    def test_warmup_sync_checks_route_through_hip_ok_not_silent_comparison(self):
        # HI64 (2026-08-23, second real-hardware finding): the retry fix
        # alone did not recover a second reproduction of the flake -- the
        # trace showed zero elapsed_time_retry events, proving the failure
        # never reached the branches the first fix covered. Root cause:
        # these two warmup-phase checks compared a HIP status directly
        # instead of going through hip_ok(), so a real failure here was
        # invisible in the log too. Fixed the same way; NOT retried (a
        # warmup-phase synchronization failure is not the same narrow class
        # of "spurious sub-ms elapsedTime reading" the retry loop targets).
        tuner = TUNER.read_text(encoding="utf-8")
        warmup_region_start = tuner.index('"warmup_complete", candidate->stable_name);')
        warmup_region_end = tuner.index('"synchronize", candidate->stable_name);')
        warmup_region = tuner[warmup_region_start:warmup_region_end]
        self.assertIn(
            'hip_ok(hipStreamSynchronize(lc.stream), "hipStreamSynchronize(warmup)")',
            warmup_region,
        )
        self.assertIn(
            'hip_ok(hipGetLastError(), "hipGetLastError(post-warmup)")',
            warmup_region,
        )
        self.assertNotIn("hipStreamSynchronize(lc.stream) != hipSuccess", warmup_region)
        self.assertNotIn("hipGetLastError() != hipSuccess", warmup_region)

    def test_post_measurement_check_routes_through_hip_ok_not_silent_comparison(self):
        tuner = TUNER.read_text(encoding="utf-8")
        self.assertIn(
            'hip_ok(hipGetLastError(), "hipGetLastError(post-measurement)")',
            tuner,
        )
        self.assertNotIn("last_status != hipSuccess", tuner)

    def test_exhausting_retries_still_poisons_exactly_as_before(self):
        # Zero regression to the permanent-failure path: exhausting every
        # attempt still calls disable_smi_after_measurement_failure() and
        # returns false, unchanged from the pre-HI64-retry behavior.
        tuner = TUNER.read_text(encoding="utf-8")
        exhausted = tuner.index("if (!sample_ok) {")
        block = tuner[exhausted : tuner.index("}", tuner.index("return false;", exhausted)) + 1]
        self.assertIn("disable_smi_after_measurement_failure();", block)
        self.assertIn("return false;", block)


if __name__ == "__main__":
    unittest.main()
