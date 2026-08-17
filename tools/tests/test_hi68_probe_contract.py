"""HI68 source contract tests: canary retry is a stability probe, never a
ranking-data replacement (RV49/F2).

The behavioral guarantee has three parts, each pinned here against the
tuner sources:

1. The transition between measurement blocks is the pure header
   hip-autotune-canary.h -- GPU-free by construction (no hip/ggml include),
   which is what makes it host-unit-testable at all.
2. A failed initial canary runs at most ONE pair-only stability probe whose
   statistics are DISCARDED: the old code overwrote pair[i]->median_us and
   friends with exactly this self-selected fresh draw; that write pattern
   must not exist anywhere in the tuner.
3. Only if the probe passes is ONE complete finalist block re-measured via
   the same extracted unit the normal final stage uses (measure_finalist_block),
   and that fresh block gets the identical post-block scrutiny (E4 noisy +
   native-baseline checks) as the original one. Its canary is evaluated once,
   terminal: no retry branch exists in the FRESH stage of the state machine.
"""

import os
import re
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
TUNER = os.path.join(REPO_ROOT, "src", "ggml", "src", "ggml-cuda",
                     "hip-autotune-tuner.cu")
TUNER_CUH = os.path.join(REPO_ROOT, "src", "ggml", "src", "ggml-cuda",
                         "hip-autotune-tuner.cuh")
CANARY_H = os.path.join(REPO_ROOT, "src", "ggml", "src", "ggml-cuda",
                        "hip-autotune-canary.h")
HOST_TEST_CPP = os.path.join(REPO_ROOT, "tools", "tests",
                             "canary_decision_host_test.cpp")
DRIVER_PY = os.path.join(REPO_ROOT, "tools", "tests",
                         "test_hi68_canary_decision.py")


def _read(path):
    return Path(path).read_text(encoding="utf-8")


def _group(pattern: str, text: str, name: str) -> str:
    """re.search group(1), failing the test with a named message if absent."""
    match = re.search(pattern, text, re.S)
    assert match is not None, f"{name}: pattern not found in source"
    return match.group(1)


class TestHi68CanaryContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuner = _read(TUNER)
        cls.cuh = _read(TUNER_CUH)
        cls.canary_h = _read(CANARY_H)

    # -- 1. the transition is a GPU-free pure header -----------------------
    def test_canary_header_exists_and_is_gpu_free(self):
        for include in ("hip/", "ggml/", "hip-autotune-tuner",
                        "hip-autotune-dispatch"):
            self.assertNotIn(
                include, self.canary_h,
                f"hip-autotune-canary.h must stay GPU-free (found {include!r})")

    def test_canary_header_states_the_three_stages(self):
        for stage in ("GGML_HIP_CANARY_STAGE_INITIAL",
                      "GGML_HIP_CANARY_STAGE_PROBE",
                      "GGML_HIP_CANARY_STAGE_FRESH"):
            self.assertIn(stage, self.canary_h)

    def test_fresh_stage_has_no_retry_branch(self):
        # The FRESH case may only end in RANK or STOP_UNRESOLVED: requesting
        # another probe or fresh block would re-open the self-selected
        # baseline (F2).
        body = _group(r"case GGML_HIP_CANARY_STAGE_FRESH:(.*?)break;",
                      self.canary_h, "FRESH stage branch")
        self.assertNotIn("RUN_PROBE", body)
        self.assertNotIn("RUN_FRESH", body)
        self.assertIn("STOP_UNRESOLVED", body)
        self.assertIn("RANK", body)

    def test_retry_budget_is_initial_stage_only(self):
        self.assertIn("retries_allowed",
                      _group(r"case GGML_HIP_CANARY_STAGE_INITIAL:(.*?)break;",
                             self.canary_h, "INITIAL stage branch"))
        for stage in ("PROBE", "FRESH"):
            body = _group(rf"case GGML_HIP_CANARY_STAGE_{stage}:(.*?)break;",
                          self.canary_h, f"{stage} stage branch")
            self.assertNotIn("retries_allowed", body,
                             f"{stage} stage must not consult the retry budget")

    # -- 2. the probe is a probe: no statistics write-back -----------------
    def test_probe_does_not_overwrite_measurement_statistics(self):
        # The F2 defect, verbatim: the pair re-measurement writing back into
        # the ranked Measurements' median/mad/p95/host/samples.
        self.assertNotIn("pair[i]->median_us", self.tuner)
        self.assertNotIn("pair[0]->median_us", self.tuner)
        for field in ("mad_us", "p95_us", "host_median_us"):
            self.assertNotRegex(
                self.tuner, rf"pair\[\d\]->{field}\s*=")

    def test_old_attempt_retry_loop_is_gone(self):
        self.assertNotIn("for (int attempt = 0; twin != nullptr", self.tuner)
        self.assertNotIn("attempt >= config.noise_canary_retries", self.tuner)

    # -- 3. the fresh block is a complete, equally-scrutinized block -------
    def test_final_stage_is_an_extracted_block_unit(self):
        self.assertIn("auto measure_finalist_block = [&]", self.tuner)
        # Two call sites: the normal final stage and the canary fresh path.
        self.assertGreaterEqual(
            self.tuner.count("measure_finalist_block();"), 2)

    def test_fresh_path_remeasures_via_the_same_unit(self):
        body = _group(r"GGML_HIP_CANARY_RUN_FRESH\)(.*?)\n            }",
                      self.tuner, "RUN_FRESH dispatch branch")
        self.assertIn("measure_finalist_block();", body)

    def test_post_block_rejections_apply_to_every_ranked_block(self):
        # E4 noisy + native-baseline checks, extracted so the fresh block
        # gets identical scrutiny to the original.
        self.assertIn("auto post_block_reject_reason = [&]", self.tuner)
        self.assertGreaterEqual(
            self.tuner.count("post_block_reject_reason()"), 2)

    def test_fresh_canary_is_evaluated_once_with_zero_budget(self):
        # Evaluated exactly once, with a zero budget, and judged ONLY on
        # medians from the fresh block itself: a finalist that failed to
        # launch there must not contribute its stale original-block median.
        self.assertIn(
            "GGML_HIP_CANARY_STAGE_FRESH,\n"
            "                        native_m->measured ? native_m->median_us : -1.0,\n"
            "                        twin->measured ? twin->median_us : -1.0,\n"
            "                        config.noise_canary_pct, 0",
            self.tuner)

    def test_probe_uses_zero_budget_transition(self):
        self.assertIn(
            "GGML_HIP_CANARY_STAGE_PROBE,", self.tuner)

    def test_fresh_block_flag_is_recorded_and_serialized(self):
        self.assertIn("bool canary_fresh_block = false;", self.tuner)
        # Serialized into the result JSON: the format string AND the
        # argument must both carry it (a format slot without an argument is a
        # classic row-corruption bug in this file.
        # The format string carries the field as a C-escaped JSON slot:
        self.assertIn('\\"canary_fresh_block\\":%s,', self.tuner)
        self.assertIn('r.canary_fresh_block ? "true" : "false"', self.tuner)
        # The invariant is documented at the field so future readers of
        # old/new rows know which measurement window covers the ranked
        # medians: a retried pass always comes from a fresh block, and a
        # fresh block may also end unresolved (terminal, native retained).
        self.assertIn(
            "canary_state == retried_pass implies\n"
            "    // canary_fresh_block", self.tuner)

    # -- host testability wiring ------------------------------------------
    def test_host_test_and_driver_exist(self):
        cpp = _read(HOST_TEST_CPP)
        driver = _read(DRIVER_PY)
        self.assertIn("#include \"hip-autotune-canary.h\"", cpp)
        self.assertIn("CANARY_DECISION_HOST_TEST_OK", cpp)
        self.assertIn("CANARY_DECISION_HOST_TEST_OK", driver)
        self.assertIn("hip-autotune-canary.h", driver)

    def test_config_comment_states_probe_semantics(self):
        # noise_canary_retries documents its HI68 meaning (probe allowance),
        # so a reader of the config cannot mistake it for a retry-until-quiet
        # budget again.
        idx = self.cuh.index("noise_canary_retries")
        window = self.cuh[max(0, idx - 700):idx]
        self.assertIn("HI68", window)
        self.assertIn("stability", window.lower())


if __name__ == "__main__":
    unittest.main()
