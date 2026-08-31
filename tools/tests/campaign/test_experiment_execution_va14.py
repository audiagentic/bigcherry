"""VA14 tests: tools/bigcherry/experiment/execution.py -- the real
per-contract-lane paired execution primitive, unit-tested hardware-free
via an injected fake runner per GPT's design (session
ses_5bbee8ce5c9a4265, req_2072dae840434295).
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import contract as ec # noqa: E402
from bigcherry.experiment import execution as ex # noqa: E402

METRIC_PATTERN = re.compile(r"tg128\s*\|\s*([0-9.]+)")


def _fake_runner(fixed_values: dict[str, float]):
    """Returns a Runner mapping a command's first token (the binary path)
    to a fixed metric value -- deterministic, no subprocess, no hardware."""
    def runner(command: list[str]) -> str:
        key = command[0]
        value = fixed_values[key]
        return f"model | build | 99 | tg128 | {value} t/s\n"
    return runner


class MetricForWorkloadTests(unittest.TestCase):
    def test_decode_maps_to_tg128(self):
        self.assertEqual(ex.metric_for_workload("decode"), "tg128")

    def test_prefill_maps_to_pp512(self):
        self.assertEqual(ex.metric_for_workload("prefill"), "pp512")

    def test_unmapped_workload_raises(self):
        with self.assertRaises(ec.ExperimentContractError):
            ex.metric_for_workload("mtp_verify")


class RunPairedLaneTests(unittest.TestCase):
    def test_subject_faster_than_control_produces_positive_effect(self):
        runner = _fake_runner({"control_bin": 100.0, "subject_bin": 110.0})
        result = ex.run_paired_lane(
            role="positive", metric="tg128", control_command=["control_bin"],
            subject_command=["subject_bin"], pattern=METRIC_PATTERN, pairs=3, runner=runner,
        )
        self.assertGreater(result.stats["geometric_effect_pct"], 0)
        self.assertEqual(result.stats["paired_rounds"], 3)
        self.assertEqual(len(result.runs), 6) # 3 pairs x 2 arms

    def test_subject_slower_than_control_produces_negative_effect(self):
        runner = _fake_runner({"control_bin": 100.0, "subject_bin": 90.0})
        result = ex.run_paired_lane(
            role="control", metric="tg128", control_command=["control_bin"],
            subject_command=["subject_bin"], pattern=METRIC_PATTERN, pairs=3, runner=runner,
        )
        self.assertLess(result.stats["geometric_effect_pct"], 0)

    def test_identical_values_produce_zero_effect(self):
        runner = _fake_runner({"control_bin": 100.0, "subject_bin": 100.0})
        result = ex.run_paired_lane(
            role="positive", metric="tg128", control_command=["control_bin"],
            subject_command=["subject_bin"], pattern=METRIC_PATTERN, pairs=3, runner=runner,
        )
        self.assertAlmostEqual(result.stats["geometric_effect_pct"], 0.0, places=6)

    def test_order_alternates_across_pairs(self):
        seen_orders = []

        def recording_runner(command: list[str]) -> str:
            seen_orders.append(command[0])
            return "tg128 | 100.0 t/s\n"

        ex.run_paired_lane(
            role="positive", metric="tg128", control_command=["control_bin"],
            subject_command=["subject_bin"], pattern=METRIC_PATTERN, pairs=2,
            runner=recording_runner,
        )
        # pair 0: control, subject ; pair 1: subject, control
        self.assertEqual(seen_orders, ["control_bin", "subject_bin", "subject_bin", "control_bin"])

    def test_zero_pairs_rejected(self):
        with self.assertRaises(ValueError):
            ex.run_paired_lane(
                role="positive", metric="tg128", control_command=["c"],
                subject_command=["s"], pattern=METRIC_PATTERN, pairs=0,
                runner=_fake_runner({"c": 1, "s": 1}),
            )

    def test_lane_effect_from_run_produces_real_lane_effect(self):
        runner = _fake_runner({"control_bin": 100.0, "subject_bin": 110.0})
        result = ex.run_paired_lane(
            role="positive", metric="tg128", control_command=["control_bin"],
            subject_command=["subject_bin"], pattern=METRIC_PATTERN, pairs=3, runner=runner,
        )
        effect = ex.lane_effect_from_run("positive", "tg128", result)
        self.assertIsInstance(effect, ec.LaneEffect)
        self.assertEqual(effect.role, "positive")
        self.assertEqual(effect.metric, "tg128")
        self.assertGreater(effect.geometric_effect_pct, 0)


class TriggerEvidenceFromMarkerProbeTests(unittest.TestCase):
    def test_positive_hit_records_launch_observed(self):
        evidence = ex.trigger_evidence_from_marker_probe(
            lane_id="rd08-decode", role="positive", positive_hit=True,
        )
        self.assertEqual(evidence.candidate_launches, 1)

    def test_no_hit_records_zero_launches(self):
        evidence = ex.trigger_evidence_from_marker_probe(
            lane_id="rd08-decode", role="positive", positive_hit=False,
        )
        self.assertEqual(evidence.candidate_launches, 0)

    def test_feeds_real_evaluate_trigger_proof(self):
        hit = ex.trigger_evidence_from_marker_probe(
            lane_id="rd08-decode", role="positive", positive_hit=True,
        )
        gate = ec.evaluate_trigger_proof([hit])
        self.assertTrue(gate["passed"])

        miss = ex.trigger_evidence_from_marker_probe(
            lane_id="rd08-decode", role="positive", positive_hit=False,
        )
        gate2 = ec.evaluate_trigger_proof([miss])
        self.assertFalse(gate2["passed"])


class EndToEndRD08StyleCompositionTests(unittest.TestCase):
    """Proves the real composition path: run_paired_lane() ->
    lane_effect_from_run() -> aggregate_contract_effects() against RD08's
    ACTUAL real contract loaded from config/experiment-contracts.toml."""

    def test_positive_and_control_lanes_compose_into_real_aggregated_effects(self):
        contracts = ec.load_contracts(
            Path(__file__).resolve().parents[3] / "config" / "experiment-contracts.toml"
        )
        contract = contracts.contracts["RD08-Q6K-MMVQ-VDR2"]

        # aggregate_contract_effects() filters control-role effects by
        # target_metric (tg128 here) -- the control LANE is a different
        # workload (prefill), but its recorded regression-check effect
        # must be tagged with the same target_metric being aggregated, per
        # the real function's own documented contract (it computes
        # max_control_regression_pct from control-role effects AT
        # target_metric, never from a separately-named metric).
        decode_runner = _fake_runner({"control_bin": 100.0, "subject_bin": 100.5})
        decode_result = ex.run_paired_lane(
            role="positive", metric="tg128", control_command=["control_bin"],
            subject_command=["subject_bin"], pattern=METRIC_PATTERN, pairs=3,
            runner=decode_runner,
        )
        prefill_control_runner = _fake_runner({"control_bin": 100.0, "subject_bin": 99.0})
        prefill_result = ex.run_paired_lane(
            role="control", metric="tg128", control_command=["control_bin"],
            subject_command=["subject_bin"], pattern=METRIC_PATTERN, pairs=3,
            runner=prefill_control_runner,
        )

        effects = [
            ex.lane_effect_from_run("positive", "tg128", decode_result),
            ex.lane_effect_from_run("control", "tg128", prefill_result),
        ]
        aggregated = ec.aggregate_contract_effects(
            contract, effects, target_metric="tg128",
        )
        self.assertIn("target_kernel_gain_pct", aggregated)
        self.assertIn("max_control_regression_pct", aggregated)
        self.assertGreater(aggregated["target_kernel_gain_pct"], 0)


if __name__ == "__main__":
    unittest.main()
