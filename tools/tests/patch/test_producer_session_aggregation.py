"""The campaign dispatcher folds prior sessions in for both session-unit
effect policies, so improvement_no_regression_v1 contracts can be evaluated."""

from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import contract as experiment_contract  # noqa: E402
from bigcherry.patch import evidence as patch_evidence  # noqa: E402
from bigcherry.patch.campaign import producer as campaign_producer  # noqa: E402
from bigcherry.patch.validation_producer import FatTargetPlan  # noqa: E402


def _effect(ratios):
    return experiment_contract.LaneEffect(
        role="positive", metric="tg128", geometric_effect_pct=1.0, ci95_low_pct=0.5,
        ci95_high_pct=1.5, paired_rounds=len(ratios), pair_ratios=tuple(ratios),
    )


def _record(contract, ratios, arch="gfx1100"):
    return {
        "contracts": [{"id": "C", "hash": contract.contract_hash}],
        "gpu_architectures": [arch],
        "lane_effects": [{"role": "positive", "metric": "tg128", "pair_ratios": list(ratios)}],
    }


class SessionAggregationTests(unittest.TestCase):
    def _contract(self, policy):
        acceptance = SimpleNamespace(
            effect_evidence_policy=policy, end_to_end_gain_pct=None, target_kernel_gain_pct=0.0
        )
        return SimpleNamespace(acceptance=acceptance, contract_hash="h" * 32)

    def _aggregate(self, policy, prior):
        contract = self._contract(policy)
        with mock.patch.object(patch_evidence, "load_records", lambda patch: prior(contract)):
            return campaign_producer._aggregate_producer_session_effects(
                Namespace(patch="p"),
                contract=contract,
                contract_id="C",
                lane_effects=(_effect([1.01, 1.02, 1.01]),),
                target_metric="tg128",
                aggregated={},
                fat_targets=FatTargetPlan(targets=("gfx1100",)),
            )

    def test_improvement_policy_counts_prior_sessions_of_the_same_hardware(self):
        def prior(contract):
            return [
                _record(contract, [1.02, 1.01, 1.02]),
                _record(contract, [1.01, 1.01, 1.02]),
                _record(contract, [1.02, 1.02, 1.01]),
                _record(contract, [1.05, 1.05, 1.05], arch="gfx1201"),
            ]

        aggregated = self._aggregate("improvement_no_regression_v1", prior)
        self.assertEqual(aggregated["target_kernel_gain_pct_sessions"], 4)
        self.assertGreater(aggregated["target_kernel_gain_pct_ci95_low"], 0.0)

    def test_single_run_policy_is_not_aggregated(self):
        aggregated = self._aggregate("ci95_threshold_bound_v1", lambda contract: [])
        self.assertNotIn("target_kernel_gain_pct_sessions", aggregated)


if __name__ == "__main__":
    unittest.main()
