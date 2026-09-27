"""lane_effect() extracts a named lane from a paired outcome.

Regression (t-1266-gfx1201-s2): a combined-invocation measurement
(bench_invocation="combined") legitimately returns MORE than one workload's
lane in one outcome; lane_effect() used to reject any outcome that was not
EXACTLY {workload}, so a producer combining decode+prefill in one process
(1266) failed extracting either lane with "must produce exactly one lane"
even though the requested lane was present.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from bigcherry.patch import producer_support as support
from bigcherry.patch import validation_producer as vp


def _run(rounds: int):
    return SimpleNamespace(stats={"paired_rounds": rounds, "geometric_effect_pct": 1.0,
                                  "ci95_low_pct": 0.5, "ci95_high_pct": 1.5, "pair_ratios": (1.01,) * rounds})


class LaneEffectTests(unittest.TestCase):
    def test_extracts_the_named_lane_from_a_combined_outcome(self) -> None:
        outcome = SimpleNamespace(runs={"decode": _run(10), "prefill": _run(10)})
        effect, run = support.lane_effect(outcome, workload="decode", metric="tg128", role="positive",
                                           rounds=10, label="test")
        self.assertIs(run, outcome.runs["decode"])

    def test_missing_lane_still_fails_closed(self) -> None:
        outcome = SimpleNamespace(runs={"prefill": _run(10)})
        with self.assertRaisesRegex(vp.ValidationProducerError, "missing its decode lane"):
            support.lane_effect(outcome, workload="decode", metric="tg128", role="positive",
                                 rounds=10, label="test")

    def test_wrong_round_count_still_fails_closed(self) -> None:
        outcome = SimpleNamespace(runs={"decode": _run(5)})
        with self.assertRaisesRegex(vp.ValidationProducerError, "paired rounds"):
            support.lane_effect(outcome, workload="decode", metric="tg128", role="positive",
                                 rounds=10, label="test")


if __name__ == "__main__":
    unittest.main()
