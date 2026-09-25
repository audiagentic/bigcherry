"""PVPS06: distribution-level full-vocab criterion (offline, synthetic)."""

from __future__ import annotations

import math
import unittest
from array import array

from bigcherry.experiment import full_vocab as fv


def _dist(probs: list[float]) -> array:
    total = sum(probs)
    return array("d", [math.log(p / total) for p in probs])


BASE = [0.6, 0.25, 0.1, 0.04] + [0.01 / 996] * 996


def _judge(control: array, subject: array, criterion=fv.NEAR_LOSSLESS, tokens_match=True):
    m = fv.step_metrics(control, subject, criterion)
    return fv.judge(criterion, tokens_match=tokens_match, max_abs_diff=m.max_abs_diff, max_kl=m.kl,
                    max_material=m.material_diff, min_topp_mass=m.topp_mass)


class NearLosslessTests(unittest.TestCase):
    def test_identical_distributions_pass(self) -> None:
        ok, _ = _judge(_dist(BASE), _dist(BASE))
        self.assertTrue(ok)

    def test_extreme_tail_noise_passes(self) -> None:
        # The 1207 shape: a ~1e-13-probability token moves by 1.8 nats.
        control = list(BASE) + [1e-13]
        subject = list(BASE) + [1e-13 * math.exp(1.8)]
        ok, verdict = _judge(_dist(control), _dist(subject))
        self.assertTrue(ok, verdict)
        m = fv.step_metrics(_dist(control), _dist(subject), fv.NEAR_LOSSLESS)
        self.assertGreater(m.max_abs_diff, 1.0)  # the old all-vocab rule would have failed

    def test_material_token_shift_fails(self) -> None:
        subject = list(BASE)
        subject[2] *= 1.2  # 0.1 -> 0.12: ~0.18 nats on a material token
        ok, verdict = _judge(_dist(BASE), _dist(subject))
        self.assertFalse(ok)
        self.assertIn("material logprob diff", verdict)

    def test_nucleus_drift_fails(self) -> None:
        subject = [0.3, 0.25, 0.1, 0.04] + [0.31 / 996] * 996
        ok, verdict = _judge(_dist(BASE), _dist(subject))
        self.assertFalse(ok)

    def test_token_divergence_always_fails(self) -> None:
        ok, verdict = _judge(_dist(BASE), _dist(BASE), tokens_match=False)
        self.assertFalse(ok)
        self.assertIn("tokens differ", verdict)


class BitIdenticalTests(unittest.TestCase):
    def test_any_difference_fails(self) -> None:
        control = list(BASE) + [1e-13]
        subject = list(BASE) + [1.0000001e-13]
        ok, _ = _judge(_dist(control), _dist(subject), criterion=fv.BIT_IDENTICAL)
        self.assertFalse(ok)

    def test_exact_passes(self) -> None:
        ok, _ = _judge(_dist(BASE), _dist(BASE), criterion=fv.BIT_IDENTICAL)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
