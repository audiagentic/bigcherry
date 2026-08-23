"""HI24 steps 5-6: source-contract tests for hot-list-based screening.

A signature within the top `hot_share_pct` of a workload's cumulative
call-weighted impact (an operator-supplied hot list, GGML_HIP_TUNE_HOT_
SIGNATURES) skips screening's noise-driven elimination -- every measured,
correct candidate reaches the final stage instead of only the top few by a
screening median RV21 showed carries ~14% error.

Source-contract only, matching this file's existing testing pattern
(test_tuner_artifact_json.py, test_hi24_canary_summary.py) -- no HIP
compiler is available in this dev environment.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TUNER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"
TUNER_CUH = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cuh"


class Hi24HotListContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuner = TUNER.read_text(encoding="utf-8")
        cls.cuh = TUNER_CUH.read_text(encoding="utf-8")

    def test_config_has_hot_share_pct_with_conservative_default(self):
        self.assertIn("double hot_share_pct         = 80.0;", self.cuh)

    def test_hot_share_env_override_is_bounded(self):
        self.assertIn(
            'double_env("GGML_HIP_TUNE_HOT_SHARE", 0.0, 100.0, c.hot_share_pct);',
            self.tuner,
        )

    def test_no_env_var_means_no_signature_is_hot(self):
        # hot_list() returns early (empty map) when the env var is unset;
        # is_hot_signature() must treat an empty list as "never hot", not
        # attempt a lookup against nothing.
        self.assertIn(
            "if (list.cum_share_pct.empty()) {\n        return false;", self.tuner
        )

    def test_signature_absent_from_a_loaded_list_is_not_silently_hot(self):
        # test-backend-ops sweeps produce far more signatures than any real
        # hot list; an unlisted signature must fall through to "not hot",
        # not be promoted by the absence of a match.
        self.assertIn(
            "if (found == list.cum_share_pct.end()) {\n        return false;",
            self.tuner,
        )

    def test_hot_threshold_is_cumulative_share_at_or_below_config(self):
        self.assertIn("return found->second <= config.hot_share_pct;", self.tuner)

    def test_result_computes_hot_signature_from_the_real_signature_digest(self):
        self.assertIn(
            "result.hot_signature    = is_hot_signature(result.signature_digest, config);",
            self.tuner,
        )

    def test_retention_loop_retains_every_survivor_of_a_hot_signature(self):
        self.assertIn(
            "if (is_native || is_twin || in_top || near_best || result.hot_signature) {",
            self.tuner,
        )

    def test_hot_signature_is_emitted_in_the_measurements_jsonl(self):
        self.assertIn('"\\"hot_signature\\":%s,"', self.tuner)
        self.assertIn('r.hot_signature ? "true" : "false",', self.tuner)

    def test_hot_list_file_format_is_not_json(self):
        # No JSON parser exists anywhere in this overlay (every other file
        # here writes JSON by hand and reads none) -- the loader must use
        # sscanf against a flat text format, not attempt to parse JSON.
        self.assertIn('sscanf(line, "%63s %lf %lf %d", hex, &share_pct,', self.tuner)


if __name__ == "__main__":
    unittest.main()
