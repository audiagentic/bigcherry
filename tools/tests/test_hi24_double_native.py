"""HI24 step 4: double-native noise-canary twin, source contract tests.

These are source-contract tests: they pin the structural invariants that
make the design safe independently of GPU runtime testing -- most
importantly the pointer-lifetime ordering (twin push_back before the
screening pointer vector is built), the measurement-instance identity
rules (schedule/flush emit "#twin", the funnel counts stay registry-sized),
and the naming-authority contract (every emission site goes through
measurement_name(), the runtime ON/OFF artifacts do not exercise the
policy/ranking twin branches).
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TUNER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"
TUNER_H = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cuh"


class DoubleNativeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuner = TUNER.read_text(encoding="utf-8")
        cls.tuner_h = TUNER_H.read_text(encoding="utf-8")

    def test_double_native_defaults_on_and_has_boolean_env_override(self):
        self.assertIn("int    double_native         = 1;", self.tuner_h)
        self.assertIn(
            'int_env("GGML_HIP_TUNE_DOUBLE_NATIVE", 0, 1, c.double_native);', self.tuner
        )

    def test_twin_is_created_before_screening_pointer_vector(self):
        # The twin copies a Measurement and push_backs it into
        # result.measurements. `screening` holds raw pointers into that
        # vector, so the creation must precede the pointer capture or a
        # later reallocation dangles every screening entry.
        create = self.tuner.find("result.measurements.push_back(std::move(twin))")
        capture = self.tuner.find("std::vector<Measurement *> screening;")
        self.assertGreaterEqual(create, 0)
        self.assertGreater(capture, create)

    def test_native_lookup_explicitly_excludes_twin(self):
        # Role identity, not descriptor identity: the twin shares native's
        # descriptor and must never be mistaken for the baseline.
        self.assertIn(
            "if (!m->is_native_twin && m->candidate == native.candidate) {\n"
            "            native_m = m; break;",
            self.tuner,
        )

    def test_screening_skips_only_primary_native(self):
        self.assertIn(
            "if (m == native_m) {\n            continue;\n        }", self.tuner
        )
        # The old descriptor-equality skip must be gone from the screening
        # loop: it would also skip the twin and starve the canary.
        self.assertNotIn(
            "for (Measurement * m : screening) {\n"
            "        if (m->candidate == native.candidate) {",
            self.tuner,
        )

    def test_twin_does_not_increment_candidate_measured_count(self):
        # generated >= applicable >= eligible >= measured must stay a funnel
        # over the registry; the twin is a synthetic measurement role.
        self.assertIn(
            "if (!m->is_native_twin) {\n            ++result.measured;\n        }",
            self.tuner,
        )

    def test_twin_nan_inf_rejects_the_signature(self):
        # The NaN/Inf branch must reject the signature for a twin, not just
        # drop it like an ordinary bad challenger -- compare_outputs fails
        # before any tolerance is computed, so it needs its own twin check.
        self.assertIn(
            "if (m->is_native_twin) {\n"
            "                m->reason = GGML_HIP_REJECT_NAN_INF;\n"
            '                result.reason = "double-native twin failed '
            'correctness; signature rejected";',
            self.tuner,
        )

    def test_twin_correctness_mismatch_rejects_the_signature(self):
        # The same descriptor producing incompatible output on its second
        # run is a measurement-context integrity failure, not an ordinary
        # challenger rejection. The reject reason must be recorded on the
        # twin's Measurement BEFORE the return so flush never serializes a
        # signature-killing failure as status "ok".
        self.assertIn(
            "m->reason = GGML_HIP_REJECT_TOLERANCE;\n"
            "            if (m->is_native_twin) {\n"
            '                result.reason = "double-native twin failed '
            'correctness; signature rejected";',
            self.tuner,
        )
        # The twin message must not overstate the action scope (other
        # unrelated paths legitimately use "run rejected").
        self.assertNotIn(
            "double-native twin failed correctness; run rejected", self.tuner
        )

    def test_native_and_twin_are_both_retained_as_finalists(self):
        self.assertIn("const bool is_twin        = m->is_native_twin;", self.tuner)
        self.assertIn("if (is_native || is_twin || in_top || near_best)", self.tuner)
        # Descriptor equality must no longer play the native role here.
        self.assertNotIn(
            "const bool is_native = m->candidate == native.candidate;", self.tuner
        )

    def test_final_schedule_disambiguates_and_orders_twin(self):
        # Equivalent strcmp keys (native and twin share a stable name) must
        # get an explicit tie-break, and the schedule must use the
        # measurement-instance identity or it carries duplicate names.
        self.assertIn("return a->is_native_twin < b->is_native_twin;", self.tuner)
        self.assertIn(
            "result.schedule_candidates.push_back(measurement_name(*m));", self.tuner
        )

    def test_jbest_canary_precedes_double_native_fallback(self):
        jbest = self.tuner.find("ggml_cuda_mmq_native_j_best")
        fallback = self.tuner.find("twin = native_twin;")
        self.assertGreaterEqual(jbest, 0)
        self.assertGreater(fallback, 0)
        self.assertGreater(fallback, jbest)

    def test_canary_fallback_uses_measured_native_twin(self):
        self.assertIn(
            "if (m->is_native_twin && m->measured) {\n                native_twin = m;",
            self.tuner,
        )
        self.assertIn("result.canary_pair = measurement_name(*twin);", self.tuner)

    def test_flush_uses_twin_measurement_identity_and_emits_setting(self):
        # Duplicate raw stable names in the JSONL would break offline
        # consumers; the header must record whether the twin was active so
        # ON/OFF artifacts are not configuration-equivalent.
        self.assertIn("measurement_name(m).c_str()", self.tuner)
        # The header also records the Slice B0 flush configuration: a
        # flush=0 artifact is not measurement-equivalent to a flush=1 one.
        self.assertIn('"\\\"alpha\\\":%.4f,\\\"double_native\\\":%d,', self.tuner)

    def test_measurement_name_is_the_single_naming_authority(self):
        # One definition of measurement-instance identity: the "#twin"
        # suffix is constructed exactly once (inside the helper) and every
        # emission site goes through it. The runtime ON artifact cannot
        # exercise the policy/ranking twin branches, so this guard must be
        # at source level.
        self.assertEqual(self.tuner.count('"#twin"'), 1)
        for use in (
            "const std::string emitted_name = measurement_name(*rv.m);",
            "const std::string predicted_name = measurement_name(*picked);",
            "result.schedule_candidates.push_back(measurement_name(*m));",
            "result.canary_pair = measurement_name(*twin);",
            "measurement_name(m).c_str()",
        ):
            self.assertIn(use, self.tuner)

    def test_jbest_search_excludes_the_synthetic_twin(self):
        # "J-best preferred, synthetic twin fallback" must be encoded in the
        # scan itself rather than emergent from the native wrapper's
        # zero-variant descriptor never matching.
        self.assertIn(
            "if (m == native_m || m->is_native_twin ||\n"
            "                            !m->measured || m->candidate == nullptr) {",
            self.tuner,
        )


if __name__ == "__main__":
    unittest.main()
