"""Source-level safety contracts for dispatch paths with upstream assertions."""

from __future__ import annotations

import unittest
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
DISPATCH = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-dispatch.cu"
RECORD = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-record.cpp"
RECORD_HEADER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-record.h"
TUNER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"
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
        final_calls = tuner[tuner.index("for (int round = 0; round < config.final_samples;"):]
        self.assertNotIn("&ctx, true", final_calls)
        confirmation = tuner[tuner.index("const int rounds = std::max(config.confirmation_samples"):]
        self.assertNotIn("&ctx, true", confirmation)

    def test_workspace_protocol_trace_is_gated_and_labels_confirmation(self):
        tuner = TUNER.read_text(encoding="utf-8")
        self.assertIn('GGML_HIP_WORKSPACE_TRACE', tuner)
        self.assertIn('trace_workspace_event(stage, "clear_cache"', tuner)
        metrics = tuner[tuner.index('#ifdef GGML_HIP_WORKSPACE_METRICS'):]
        self.assertIn('trace_workspace_event(stage, "rebase_peak"', metrics)
        self.assertIn('nullptr, false, nullptr, "confirmation")', tuner)

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

    def test_tuner_event_destruction_has_one_checked_cleanup_seam(self):
        """Every created event must use the checked, non-short-circuit seam."""
        source = TUNER.read_text(encoding="utf-8")
        helper_start = source.index("static bool hip_event_destroy_checked(")
        helper_end = source.index("bool time_candidate(", helper_start)
        helper = source[helper_start:helper_end]
        managed = source[:helper_start] + source[helper_end:]

        raw_destroy = re.compile(r'''(?<!["'])\bhipEventDestroy\s*\(''')
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
        self.assertIn('effective_api', record)
        self.assertIn('effective_call_api', record)
        self.assertIn('workspace_bytes', record)
        self.assertIn("const char * effective_api", header)
        self.assertIn("size_t workspace_bytes", header)

        record_block = dispatch[dispatch.index("if (mode == GGML_HIP_DISPATCH_MODE_RECORD)"):]
        self.assertLess(
            record_block.index("ggml_hip_blas_workspace"),
            record_block.index("ggml_hip_record_observation"),
        )

    def test_blas_effective_call_api_covers_native_branches_without_dispatch_changes(self):
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
        self.assertIn('rationale="the completed native single-matrix F32 BLAS call"', patch)
        self.assertIn('rationale="the completed native strided-batched BLAS call"', patch)
        self.assertIn("thread_local PairKey g_active_key", record)
        self.assertIn("effective_call_api = api", record)
        # This field is observation-only; it must not appear in the resolver.
        self.assertNotIn("effective_call_api", DISPATCH.read_text(encoding="utf-8"))

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

    def test_tuner_first_encounters_are_single_flight_through_publish(self):
        """The source-level concurrency contract prevents duplicate winners."""
        tuner = TUNER.read_text(encoding="utf-8")
        resolve_start = tuner.index("const ggml_hip_candidate_descriptor * ggml_hip_tuner_resolve(")
        resolve_end = tuner.index("void ggml_hip_tuner_flush(", resolve_start)
        resolve = tuner[resolve_start:resolve_end]

        self.assertIn("std::mutex g_single_flight_mutex;", tuner)
        self.assertIn("std::unique_lock<std::mutex> single_flight_lock(g_single_flight_mutex);", resolve)
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

    def test_tune_journal_result_keeps_replay_identity_digests(self):
        tuner = TUNER.read_text(encoding="utf-8")
        summary = tuner[tuner.index("std::string journal_result_summary("):]
        self.assertIn('signature_digest', summary)
        self.assertIn('hardware_digest', summary)
        self.assertLess(summary.index("signature_digest"), summary.index("winner"))


if __name__ == "__main__":
    unittest.main()
