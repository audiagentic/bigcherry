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
        guard = "winner != nullptr && winner != native.candidate"
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


if __name__ == "__main__":
    unittest.main()
