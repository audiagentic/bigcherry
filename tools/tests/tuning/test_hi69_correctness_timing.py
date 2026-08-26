"""HI69: source-contract tests for the correctness-cost timing diagnostic.

This is the real measurement HI69's closure decision is based on (see
HI69.md): a real gfx1100 sweep (1088 cold-path resolutions, 11,570
screening candidates, non-blocking) measured F_tune (aggregate avoidable
correctness wall time / aggregate cold tuner-resolution wall time) =
0.5379% -- well below the pre-registered <2% "not worth it" gate agreed
with dev-gpt-agent, so the proposed on-device correctness reduction was
declined. Kept as a permanent, opt-in diagnostic (matching HI87's
GGML_HIP_NATIVE_SELECT_TIMING precedent) rather than deleted, so the
documented reopen conditions (F_tune >= 2% reassess, >= 5% a device-
reduction prototype becomes justified) can be re-checked without
rebuilding the instrumentation from scratch.

Source-contract only, matching this repo's existing pattern for .cu files
-- no HIP compiler is assumed available offline (though one was used to
real-hardware-validate this diagnostic; see HI69.md's notes).
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TUNER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"


class Hi69CorrectnessTimingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = TUNER.read_text(encoding="utf-8")

    def test_env_gate_is_the_documented_permanent_name(self):
        self.assertIn('getenv("GGML_HIP_TUNE_CORRECTNESS_TIMING")', self.src)
        # The throwaway working name from the design conversation must not
        # survive into the permanent diagnostic.
        self.assertNotIn("CORRECTNESS_PROFILING", self.src)

    def test_disabled_by_default(self):
        idx = self.src.index("static bool hi69_correctness_timing_enabled() {")
        window = self.src[idx:idx + 300]
        self.assertIn("static std::atomic<bool> enabled{false};", window)

    def test_cold_tuner_scope_is_raii_not_per_return_site(self):
        # ggml_hip_tuner_resolve_impl has ~25 exit sites (the function's own
        # comment says so) -- instrumenting each individually would be both
        # tedious and easy to miss one on a future edit. RAII fires on every
        # exit automatically.
        self.assertIn("struct Hi69ColdTunerScope {", self.src)
        self.assertIn("const Hi69ColdTunerScope hi69_cold_tuner_scope;", self.src)

    def test_cold_tuner_scope_excludes_trivial_early_returns(self):
        # Cache-hit, poisoned-stub, and invalid-config returns are not real
        # cold-path work and must not be counted as "cold tuner wall time".
        scope_idx = self.src.index("const Hi69ColdTunerScope hi69_cold_tuner_scope;")
        fn_idx = self.src.index("static const ggml_hip_candidate_descriptor * ggml_hip_tuner_resolve_impl(")
        between = self.src[fn_idx:scope_idx]
        self.assertIn("tuning disabled after fatal measurement failure", between)
        self.assertIn("invalid tuning configuration", between)

    def test_every_correctness_stage_is_timed_not_sampled(self):
        # Unlike HI87's dispatch-hot-path diagnostic (1-in-128 sampling,
        # because that operation is itself only ~200ns and a per-call timer
        # would risk dominating it), every correctness stage here contains a
        # real D2H transfer + blocking sync + O(N) CPU traversal -- cheap to
        # time in full, per the agreed design (dev-gpt-agent,
        # req_5c8c2a476c2c4143).
        start = self.src.index("struct Hi69CorrectnessTiming {")
        end = self.src.index("static void hi69_report_correctness_timing() {")
        self.assertNotIn("% 128", self.src[start:end])

    def test_native_reference_copy_is_timed_and_byte_counted(self):
        idx = self.src.index('"hipMemcpyAsync(reference)")')
        window = self.src[idx:idx + 900]
        self.assertIn("native_ref_copy_ns_sum.fetch_add", window)
        self.assertIn("native_ref_copy_bytes_sum.fetch_add", window)

    def test_candidate_copy_and_compare_are_both_timed_and_byte_counted(self):
        idx = self.src.index('"hipMemcpyAsync(candidate)")')
        window = self.src[idx:idx + 1700]
        self.assertIn("compare_outputs(reference_host, candidate_host", window)
        self.assertIn("candidate_copy_bytes_sum.fetch_add", window)

    def test_report_computes_aggregate_f_tune_not_mean_of_ratios(self):
        # The agreed primary go/no-go metric is
        # (sum of correctness time) / (sum of cold tuner time), NOT an
        # average of per-candidate ratios -- averaging would let a handful
        # of tiny, cheap candidates with proportionally large fixed
        # overhead dominate the decision despite contributing negligible
        # absolute wall time (this exact gap -- 0.54% aggregate vs 2.08%
        # median-of-ratios -- was observed and explained in the real run).
        idx = self.src.index("const double f_tune =")
        window = self.src[max(0, idx - 400):idx]
        self.assertIn("correctness_sum / (double) cold_ns", self.src[idx:idx + 80])
        self.assertIn("NOT a mean of per-candidate", window)

    def test_report_includes_bytes_moved(self):
        # Distinguishes "the workload now has larger outputs" from "this
        # driver/platform made equivalent copies more expensive" if a
        # future sweep reports a different F_tune (dev-gpt-agent review).
        self.assertIn("candidate_copy_bytes_sum", self.src)
        self.assertIn("native_ref_copy_bytes_sum", self.src)
        self.assertIn("d2h_bytes=%llu", self.src)

    def test_report_wired_to_flush_independent_of_dispatch_db(self):
        flush_idx = self.src.index("void ggml_hip_tuner_flush() {")
        window = self.src[flush_idx:flush_idx + 700]
        self.assertIn("hi69_report_correctness_timing();", window)
        # Must run before the GGML_HIP_DISPATCH_DB-unset early return --
        # this is a profiling summary, not tuning evidence.
        report_idx = window.index("hi69_report_correctness_timing();")
        db_check_idx = window.index("GGML_HIP_DISPATCH_DB")
        self.assertLess(report_idx, db_check_idx)


if __name__ == "__main__":
    unittest.main()
