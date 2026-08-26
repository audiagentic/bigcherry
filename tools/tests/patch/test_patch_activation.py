"""tools/bigcherry/patch_activation.py had zero test coverage before this
(found while fixing a real bug GPT's review caught: verdict() collapsed
executed+correctness=None into "validated", when it should mean "activation
proven, correctness genuinely unknown")."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import activation as pa # noqa: E402


class VerdictTests(unittest.TestCase):
    def test_not_applicable_is_gate_verified_blocked(self):
        ev = pa.ActivationEvidence(status="not_applicable", mechanism="m", detail="d")
        self.assertEqual(pa.verdict(ev, correctness_passed=None), "gate-verified-blocked")
        self.assertEqual(pa.verdict(ev, correctness_passed=True), "gate-verified-blocked")

    def test_unobservable_is_unobservable(self):
        ev = pa.ActivationEvidence(status="unobservable", mechanism="m", detail="d")
        self.assertEqual(pa.verdict(ev, correctness_passed=None), "unobservable")

    def test_executed_plus_correctness_true_is_validated(self):
        ev = pa.ActivationEvidence(status="executed", mechanism="m", detail="d")
        self.assertEqual(pa.verdict(ev, correctness_passed=True), "validated")

    def test_executed_plus_correctness_false_is_failed_correctness(self):
        ev = pa.ActivationEvidence(status="executed", mechanism="m", detail="d")
        self.assertEqual(pa.verdict(ev, correctness_passed=False), "failed-correctness")

    def test_executed_plus_correctness_unknown_is_activation_verified_not_validated(self):
        # Regression guard for the real bug: this must NOT be "validated" --
        # correctness is genuinely unproven at this point.
        ev = pa.ActivationEvidence(status="executed", mechanism="m", detail="d")
        result = pa.verdict(ev, correctness_passed=None)
        self.assertEqual(result, "activation-verified")
        self.assertNotEqual(result, "validated")

    def test_not_executed_is_failed_activation(self):
        ev = pa.ActivationEvidence(status="not_executed", mechanism="m", detail="d")
        self.assertEqual(pa.verdict(ev, correctness_passed=None), "failed-activation")


class EligibilityGateEvidenceTests(unittest.TestCase):
    def test_blocks_when_predicate_false_on_all_available_archs(self):
        ev = pa.eligibility_gate_evidence(
            predicate_name="is_rdna35", our_archs=["gfx1100", "gfx1201"],
            predicate_results={"gfx1100": False, "gfx1201": False, "gfx1151": True},
            required_true_for=["gfx1151"],
        )
        self.assertEqual(ev.status, "not_applicable")

    def test_raises_when_predicate_true_on_an_available_arch(self):
        with self.assertRaises(ValueError):
            pa.eligibility_gate_evidence(
                predicate_name="is_rdna35", our_archs=["gfx1100"],
                predicate_results={"gfx1100": True, "gfx1151": True},
                required_true_for=["gfx1151"],
            )

    def test_raises_when_positive_control_fails(self):
        with self.assertRaises(ValueError):
            pa.eligibility_gate_evidence(
                predicate_name="is_rdna35", our_archs=["gfx1100"],
                predicate_results={"gfx1100": False, "gfx1151": False},
                required_true_for=["gfx1151"],
            )


if __name__ == "__main__":
    unittest.main()
