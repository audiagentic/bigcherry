"""HI24 step 8: source-contract test for the end-of-run canary summary.

canary_pct/canary_retries are recorded per signature in the measurements
JSONL but nothing aggregated them across a run, so the noise-canary
mechanism (HI24 steps 1-4, already landed) had no way to say whether a
sweep's numbers could be trusted as a whole. This adds a one-line aggregate
log emitted once at the end of ggml_hip_tuner_flush(), after the
measurements file is committed.

Source-contract only, matching this file's existing testing pattern
(test_tuner_artifact_json.py) -- no HIP compiler is available in this
dev environment, so the test asserts against the real .cu source text
rather than compiling and running it.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TUNER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"


class Hi24CanarySummaryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuner = TUNER.read_text(encoding="utf-8")

    def test_summary_runs_after_the_file_is_committed(self):
        # Must not run on a partially-written/uncommitted file: the summary
        # block appears after the "wrote N tuning result(s)" log line, which
        # itself only runs after ggml_hip_atomic_commit() succeeded.
        commit_index = self.tuner.index("ggml_hip_atomic_commit(measurements_atomic)")
        wrote_index = self.tuner.index('"bigcherry: wrote %zu tuning result(s)')
        summary_index = self.tuner.index('"bigcherry: canary --')
        self.assertLess(commit_index, wrote_index)
        self.assertLess(wrote_index, summary_index)

    def test_excludes_unchecked_signatures_rather_than_treating_as_zero(self):
        # canary_pct == -1.0 means no same-kernel pair existed for that
        # signature (e.g. non-MMQ native with double_native disabled) --
        # must be skipped from the aggregate, not counted as 0% divergence.
        self.assertIn("if (r.canary_pct < 0.0) {\n                continue;", self.tuner)

    def test_flagged_uses_the_configured_noise_threshold(self):
        # The flagged count must compare against the same env-tunable
        # threshold the per-signature canary itself uses
        # (GGML_HIP_TUNE_NOISE_PCT -> config.noise_canary_pct), not a new
        # hardcoded constant.
        self.assertIn("r.canary_pct > config.noise_canary_pct", self.tuner)

    def test_retried_counts_signatures_not_probes(self):
        self.assertIn("if (r.canary_retries > 0) {\n                ++retried;", self.tuner)

    def test_zero_checked_signatures_logs_rather_than_dividing_or_crashing(self):
        # worst/worst_dispatch stay at their initial values when nothing was
        # checked -- must not be reported as "worst 0.00% (<empty digest>)"
        # as if a real signature scored zero.
        self.assertIn('"bigcherry: canary -- 0/%zu signature(s) had a same-kernel pair', self.tuner)


if __name__ == "__main__":
    unittest.main()
