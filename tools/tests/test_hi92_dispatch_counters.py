"""HI92: source-contract tests for opt-in hot-path dispatch counters.

Pure diagnostic instrumentation (dispatch entries, native-selector calls,
hardware-key/signature-digest construction, L1/L2/L3 cache hit rates) --
zero behavior change, zero cost when GGML_HIP_DISPATCH_COUNTERS is unset.
Source-contract only, matching this repo's existing testing pattern for
.cu files -- no HIP compiler is assumed available offline (though one was
used to real-hardware-validate this change; see HI92.md's notes).
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DISPATCH = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-dispatch.cu"


class Hi92DispatchCountersContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = DISPATCH.read_text(encoding="utf-8")

    def test_env_gate_is_the_documented_name(self):
        self.assertIn('getenv("GGML_HIP_DISPATCH_COUNTERS")', self.src)

    def test_disabled_by_default(self):
        self.assertIn("static std::atomic<bool> enabled{false};", self.src)

    def test_every_counter_increment_is_guarded_not_unconditional(self):
        # Every fetch_add must be reached only through a dispatch_counters_
        # enabled() check -- never a bare, unconditional increment, which
        # would defeat the "zero cost when disabled" requirement.
        import re

        for match in re.finditer(r"g_dispatch_counters\.\w+\.fetch_add", self.src):
            window = self.src[max(0, match.start() - 200):match.start()]
            self.assertIn(
                "dispatch_counters_enabled()", window,
                f"unguarded fetch_add near offset {match.start()}",
            )

    def test_report_is_wired_to_the_real_end_of_run_hook_not_atexit(self):
        # std::atexit does not reliably fire in this codebase's real
        # shutdown path (confirmed on real hardware -- see the comment at
        # the call site). ggml_hip_autotune_flush() is the same explicit
        # hook the coverage report already uses successfully.
        self.assertNotIn("std::atexit(report_dispatch_counters)", self.src)
        flush_idx = self.src.index("void ggml_hip_autotune_flush(void) {")
        end_idx = self.src.index("\n}", flush_idx)
        body = self.src[flush_idx:end_idx]
        self.assertIn("ggml_hip_coverage_report();", body)
        self.assertIn("report_dispatch_counters();", body)
        # Must come after coverage_report(), same relative order as every
        # other subsystem's flush in this function.
        self.assertLess(
            body.index("ggml_hip_coverage_report()"),
            body.index("report_dispatch_counters()"),
        )

    def test_dispatch_entry_counted_once_per_resolve_call(self):
        self.assertIn(
            "g_dispatch_counters.dispatch_entries.fetch_add(1, std::memory_order_relaxed);",
            self.src,
        )

    def test_native_select_counted_at_the_function_itself_not_each_caller(self):
        # Instrumenting inside ggml_hip_native_select() (not at its call
        # sites) captures every invocation from one place regardless of
        # which of the three public dispatch entry points called it.
        idx = self.src.index("ggml_hip_native_selection ggml_hip_native_select(")
        body_start = self.src.index("{", idx)
        window = self.src[body_start:body_start + 400]
        self.assertIn("native_select_calls.fetch_add", window)

    def test_l1_hit_and_miss_are_distinguished(self):
        self.assertIn("g_dispatch_counters.l1_hits.fetch_add", self.src)
        self.assertIn("g_dispatch_counters.l1_misses.fetch_add", self.src)

    def test_l1_miss_only_counted_when_lookup_was_actually_attempted(self):
        # l1_attempted excludes RECORD mode, which never calls
        # g_thread_bindings.find() at all -- a miss must not be counted for
        # a lookup that never happened.
        self.assertIn(
            "const bool l1_attempted = mode != GGML_HIP_DISPATCH_MODE_RECORD;",
            self.src,
        )
        self.assertIn("if (l1_attempted && dispatch_counters_enabled())", self.src)

    def test_l2_hit_and_miss_are_distinguished(self):
        self.assertIn("g_dispatch_counters.l2_hits.fetch_add", self.src)
        self.assertIn("g_dispatch_counters.l2_misses.fetch_add", self.src)

    def test_l3_lookups_and_hits_are_distinguished(self):
        self.assertIn("g_dispatch_counters.l3_lookups.fetch_add", self.src)
        self.assertIn("g_dispatch_counters.l3_hits.fetch_add", self.src)

    def test_l3_hit_counted_on_usable_not_merely_exact(self):
        # A replay entry can be an EXACT digest match yet still be rejected
        # by transform re-validation (HI31) -- l3_hits must reflect the
        # binding actually being usable, not the weaker "exact" signal.
        idx = self.src.index("g_dispatch_counters.l3_hits.fetch_add")
        window = self.src[max(0, idx - 120):idx]
        self.assertIn("if (usable) {", window)

    def test_hardware_key_builds_counted_at_both_call_sites(self):
        self.assertEqual(
            self.src.count("g_dispatch_counters.hardware_key_builds.fetch_add"), 2,
            "expected one count site for the forced-candidate hardware key "
            "build and one for the main resolution path",
        )

    def test_report_computes_hit_rate_percentages(self):
        self.assertIn("L1 hit=%llu miss=%llu (%.1f%%)", self.src)
        self.assertIn("L2 hit=%llu miss=%llu (%.1f%%)", self.src)
        self.assertIn('"L3 lookups=%llu hits=%llu (%.1f%%)', self.src)


if __name__ == "__main__":
    unittest.main()
