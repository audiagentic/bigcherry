"""HI87: source-contract tests for the sampled native_select()-vs-L1-lookup
timing diagnostic.

This is the real measurement HI87's closure decision is based on (see
HI87.md): a 110,020-dispatch real-hardware sweep on gfx1100 measured
native_select() mean=193.6ns (n=860) against L1-hit-lookup mean=52.6ns
(n=860), scaling to an estimated ~0.1-0.3% of measured production decode
latency -- below this project's intervention threshold, so the warm-cache-
first reorder (RP2/RP3) was declined. Kept as a permanent, opt-in,
zero-cost-when-disabled diagnostic (matching GGML_HIP_DISPATCH_COUNTERS'
pattern, HI92) rather than deleted, so HI87's own documented reopen
condition -- a future workload where this path measurably exceeds ~1% of
real end-to-end decode wall time -- can be re-checked without rebuilding
the instrumentation from scratch.

Source-contract only, matching this repo's existing pattern for .cu files
-- no HIP compiler is assumed available offline (though one was used to
real-hardware-validate this diagnostic; see HI87.md's notes).
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DISPATCH = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-dispatch.cu"


class NativeSelectTimingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = DISPATCH.read_text(encoding="utf-8")

    def test_env_gate_is_the_documented_name(self):
        self.assertIn('getenv("GGML_HIP_NATIVE_SELECT_TIMING")', self.src)

    def test_disabled_by_default(self):
        idx = self.src.index("static bool native_select_timing_enabled() {")
        window = self.src[idx:idx + 300]
        self.assertIn("static std::atomic<bool> enabled{false};", window)

    def test_two_independent_sample_counters_not_one_shared(self):
        # A shared counter alternates parity between the two call sites (one
        # always ticks first, the other second, every dispatch) -- an even
        # modulus check can then never fire for the odd-parity site. This
        # was a real bug caught on the first run (native_select sampled
        # n=1720, L1 sampled n=0, exactly). Two independent atomics fix it.
        self.assertIn("std::atomic<uint64_t> native_select_tick{0};", self.src)
        self.assertIn("std::atomic<uint64_t> l1_hit_tick{0};", self.src)
        self.assertNotIn("std::atomic<uint64_t> sample_tick{0};", self.src)

    def test_every_sample_site_is_gated_by_the_enabled_check(self):
        import re

        for match in re.finditer(
            r"native_select_timing_should_sample_\w+\(\)", self.src
        ):
            # Skip the function's own definition (`static bool name() {`) --
            # only actual call sites need the enabled-check guard.
            after = self.src[match.end():match.end() + 5]
            if after.lstrip().startswith("{"):
                continue
            window = self.src[max(0, match.start() - 120):match.start()]
            self.assertIn(
                "native_select_timing_enabled()", window,
                f"unguarded sample-check near offset {match.start()}",
            )

    def test_native_select_call_site_is_timed(self):
        idx = self.src.index("ggml_hip_native_select(ctx, src0, src1, ids, dst);")
        window = self.src[max(0, idx - 500):idx + 500]
        self.assertIn("sample_native_select", window)
        self.assertIn("std::chrono::steady_clock::now()", window)

    def test_l1_lookup_call_site_is_timed(self):
        idx = self.src.index("g_thread_bindings.find(ctx.device, sig, &thread_binding)")
        window = self.src[max(0, idx - 500):idx + 500]
        self.assertIn("sample_l1", window)
        self.assertIn("std::chrono::steady_clock::now()", window)

    def test_l1_timing_wraps_the_lookup_regardless_of_hit_or_miss(self):
        # The comparison HI87 needed is against the lookup's OWN cost, not
        # just the hit path -- timing must wrap g_thread_bindings.find()
        # itself, not be nested inside the l1_found branch.
        find_idx = self.src.index(
            "const bool l1_found = l1_attempted && g_thread_bindings.find"
        )
        hit_branch_idx = self.src.index("if (l1_found) {", find_idx)
        between = self.src[find_idx:hit_branch_idx]
        self.assertIn("l1_hit_ns_sum.fetch_add", between)

    def test_report_is_wired_to_the_real_end_of_run_hook(self):
        flush_idx = self.src.index("void ggml_hip_autotune_flush(void) {")
        end_idx = self.src.index("\n}", flush_idx)
        body = self.src[flush_idx:end_idx]
        self.assertIn("native_select_timing_enabled()", body)
        self.assertIn("report_native_select_timing();", body)

    def test_report_computes_both_means(self):
        self.assertIn("native_select() mean=%.1fns (n=%llu)", self.src)
        self.assertIn("L1-hit-lookup mean=%.1fns (n=%llu)", self.src)


if __name__ == "__main__":
    unittest.main()
