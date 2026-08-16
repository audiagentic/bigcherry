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
DISPATCH_PATCH = ROOT / "patches" / "0200_dispatch_hook.py"
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

        self.assertIn("g_host_sync_overhead_valid = false;", function)
        self.assertIn("if (!destroy_ok || samples.empty())", function)
        self.assertIn("if (!std::isfinite(overhead) || overhead < 0.0)", function)
        self.assertLess(
            function.index("const bool destroy_ok = destroy_events();"),
            function.index("g_host_sync_overhead_valid = true;"),
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
        """The source-level concurrency contract prevents duplicate winners."""
        tuner = TUNER.read_text(encoding="utf-8")
        resolve_start = tuner.index(
            "const ggml_hip_candidate_descriptor * ggml_hip_tuner_resolve("
        )
        resolve_end = tuner.index("void ggml_hip_tuner_flush(", resolve_start)
        resolve = tuner[resolve_start:resolve_end]

        self.assertIn("std::mutex g_single_flight_mutex;", tuner)
        self.assertIn(
            "std::unique_lock<std::mutex> single_flight_lock(g_single_flight_mutex);",
            resolve,
        )
        lock = resolve.index("single_flight_lock(g_single_flight_mutex)")
        lookup = resolve.index("g_results.find(dispatch_digest)")
        first_measurement = resolve.index("const ggml_hip_tuner_config & config")
        final_publish = resolve.rindex("record_result(dispatch_digest, result);")
        final_return = resolve.index("return result.winner;", final_publish)

        # Every waiter is serialized before it can miss the cache; the lock is
        # still held through the only final publish and winner return.
        self.assertLess(lock, lookup)
        self.assertLess(lookup, first_measurement)
        self.assertLess(final_publish, final_return)
        self.assertNotIn("single_flight_lock.unlock()", resolve)
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
        self.assertIn("std::atomic<bool> g_tuner_poisoned{false};", tuner)
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
        quarantine = dispatch[
            dispatch.index("// EX02 quarantine") : dispatch.index(
                "return ggml_cuda_mmq_variant_is_eligible("
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
                "const ggml_hip_candidate_descriptor * ggml_hip_tuner_resolve("
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

    def test_tune_journal_result_keeps_replay_identity_digests(self):
        tuner = TUNER.read_text(encoding="utf-8")
        summary = tuner[tuner.index("std::string journal_result_summary(") :]
        self.assertIn("signature_digest", summary)
        self.assertIn("hardware_digest", summary)
        self.assertLess(summary.index("signature_digest"), summary.index("winner"))


if __name__ == "__main__":
    unittest.main()
