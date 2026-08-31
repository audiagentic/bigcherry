"""VA12 tests: structured acceptance.resource_limits, ResourceResult,
evaluate_resource_gate(), and its fail-closed integration into
evaluate_promotion_gate() -- RD73's real requirement (a graph-cache
memory/resource-cost budget, not just the 5-rep timing comparison).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import contract as ec # noqa: E402


def _base_doc(**overrides: object) -> dict:
    doc = {
        "title": "VA12 fixture",
        "source": {
            "source_id": "mrlordcat-rdna-lab",
            "commits": ["deadbeefcafef00d"],
            "atomic_part": "va12-fixture",
        },
        "hypothesis": {
            "family": "mmq",
            "expected_effect": "performance",
            "rationale": "resource-cost fixture",
        },
        "prerequisites": [],
        "scope": {
            "backend": "hip",
            "architectures": ["gfx1100"],
            "weight_types": ["q8_0"],
        },
        "positive": {"models": ["recipe/model"], "workloads": ["mtp_verify"]},
        "controls": {"models": ["control-recipe"], "workloads": ["decode"]},
        "boundary": {"dimensions": {}},
        "correctness": {},
        "acceptance": {"max_control_regression_pct": 1},
    }
    doc.update(overrides)
    return doc


def _with_resource_limits(*limits: dict) -> dict:
    doc = _base_doc()
    doc["acceptance"] = dict(doc["acceptance"], resource_limits=list(limits))
    return doc


class ResourceLimitParsingTests(unittest.TestCase):
    def test_no_resource_limits_is_the_default(self):
        contract = ec.parse_contract(_base_doc(), contract_id="X")
        self.assertEqual(contract.acceptance.resource_limits, ())

    def test_max_value_limit_parses(self):
        doc = _with_resource_limits({"metric": "graph_cache_entries", "unit": "count", "max_value": 32})
        contract = ec.parse_contract(doc, contract_id="X")
        self.assertEqual(len(contract.acceptance.resource_limits), 1)
        limit = contract.acceptance.resource_limits[0]
        self.assertEqual(limit.metric, "graph_cache_entries")
        self.assertEqual(limit.unit, "count")
        self.assertEqual(limit.max_value, 32)
        self.assertIsNone(limit.max_increase_pct)

    def test_max_increase_pct_limit_parses(self):
        doc = _with_resource_limits(
            {"metric": "graph_cache_resident_bytes", "unit": "bytes", "max_increase_pct": 5.0}
        )
        contract = ec.parse_contract(doc, contract_id="X")
        limit = contract.acceptance.resource_limits[0]
        self.assertEqual(limit.max_increase_pct, 5.0)
        self.assertIsNone(limit.max_value)

    def test_limit_with_neither_bound_rejected(self):
        doc = _with_resource_limits({"metric": "x", "unit": "count"})
        with self.assertRaisesRegex(ec.ExperimentContractError, "max_value or max_increase_pct"):
            ec.parse_contract(doc, contract_id="X")

    def test_unknown_field_rejected(self):
        doc = _with_resource_limits({"metric": "x", "unit": "count", "max_value": 1, "bogus": True})
        with self.assertRaisesRegex(ec.ExperimentContractError, "bogus"):
            ec.parse_contract(doc, contract_id="X")

    def test_unknown_unit_rejected(self):
        doc = _with_resource_limits({"metric": "x", "unit": "kilobytes", "max_value": 1})
        with self.assertRaises(ec.ExperimentContractError):
            ec.parse_contract(doc, contract_id="X")

    def test_duplicate_metric_rejected(self):
        doc = _with_resource_limits(
            {"metric": "x", "unit": "count", "max_value": 1},
            {"metric": "x", "unit": "count", "max_value": 2},
        )
        with self.assertRaisesRegex(ec.ExperimentContractError, "duplicate metric"):
            ec.parse_contract(doc, contract_id="X")

    def test_negative_max_value_rejected(self):
        doc = _with_resource_limits({"metric": "x", "unit": "count", "max_value": -1})
        with self.assertRaises(ec.ExperimentContractError):
            ec.parse_contract(doc, contract_id="X")

    def test_multiple_limits_sorted_by_metric(self):
        doc = _with_resource_limits(
            {"metric": "z_metric", "unit": "count", "max_value": 1},
            {"metric": "a_metric", "unit": "count", "max_value": 1},
        )
        contract = ec.parse_contract(doc, contract_id="X")
        self.assertEqual(
            [limit.metric for limit in contract.acceptance.resource_limits],
            ["a_metric", "z_metric"],
        )

    def test_both_bounds_together_are_allowed(self):
        doc = _with_resource_limits(
            {"metric": "x", "unit": "count", "max_value": 32, "max_increase_pct": 5.0}
        )
        contract = ec.parse_contract(doc, contract_id="X")
        limit = contract.acceptance.resource_limits[0]
        self.assertEqual(limit.max_value, 32)
        self.assertEqual(limit.max_increase_pct, 5.0)


class ResourceResultValidationTests(unittest.TestCase):
    def test_nan_subject_value_rejected(self):
        with self.assertRaises(ec.ExperimentContractError):
            ec.ResourceResult(metric="x", unit="count", subject_value=float("nan"))

    def test_positive_infinity_subject_value_rejected(self):
        with self.assertRaises(ec.ExperimentContractError):
            ec.ResourceResult(metric="x", unit="count", subject_value=float("inf"))

    def test_negative_subject_value_rejected(self):
        with self.assertRaises(ec.ExperimentContractError):
            ec.ResourceResult(metric="x", unit="count", subject_value=-1)

    def test_bool_subject_value_rejected(self):
        with self.assertRaises(ec.ExperimentContractError):
            ec.ResourceResult(metric="x", unit="count", subject_value=True)

    def test_nan_control_value_rejected(self):
        with self.assertRaises(ec.ExperimentContractError):
            ec.ResourceResult(metric="x", unit="count", subject_value=1, control_value=float("nan"))

    def test_negative_control_value_rejected(self):
        with self.assertRaises(ec.ExperimentContractError):
            ec.ResourceResult(metric="x", unit="count", subject_value=1, control_value=-1)

    def test_invalid_unit_rejected(self):
        with self.assertRaises(ec.ExperimentContractError):
            ec.ResourceResult(metric="x", unit="kilobytes", subject_value=1)

    def test_empty_metric_rejected(self):
        with self.assertRaises(ec.ExperimentContractError):
            ec.ResourceResult(metric="", unit="count", subject_value=1)

    def test_valid_result_constructs_cleanly(self):
        result = ec.ResourceResult(metric="x", unit="count", subject_value=1, control_value=0)
        self.assertEqual(result.subject_value, 1)


class ContractHashCompatibilityTests(unittest.TestCase):
    def test_empty_resource_limits_does_not_change_the_hash(self):
        """The whole point of the additive identity-payload change: a
        contract that never declares resource_limits must keep its exact
        pre-VA12 hash."""
        without_field = ec.parse_contract(_base_doc(), contract_id="X")
        doc_with_empty_list = _base_doc()
        doc_with_empty_list["acceptance"] = dict(
            doc_with_empty_list["acceptance"], resource_limits=[]
        )
        with_empty_list = ec.parse_contract(doc_with_empty_list, contract_id="X")
        self.assertEqual(without_field.contract_hash, with_empty_list.contract_hash)

    def test_nonempty_resource_limits_changes_the_hash(self):
        without = ec.parse_contract(_base_doc(), contract_id="X")
        doc = _with_resource_limits({"metric": "x", "unit": "count", "max_value": 1})
        withit = ec.parse_contract(doc, contract_id="X")
        self.assertNotEqual(without.contract_hash, withit.contract_hash)


class EvaluateResourceGateTests(unittest.TestCase):
    def _contract(self):
        doc = _with_resource_limits(
            {"metric": "graph_cache_entries", "unit": "count", "max_value": 32}
        )
        return ec.parse_contract(doc, contract_id="X")

    def test_missing_evidence_fails(self):
        contract = self._contract()
        gate = ec.evaluate_resource_gate(contract, {})
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["missing_metrics"], ["graph_cache_entries"])

    def test_within_max_value_passes(self):
        contract = self._contract()
        result = ec.ResourceResult(metric="graph_cache_entries", unit="count", subject_value=15)
        gate = ec.evaluate_resource_gate(contract, {"graph_cache_entries": result})
        self.assertTrue(gate["passed"])

    def test_dict_key_metric_mismatch_fails(self):
        """A ResourceResult stored under the wrong dict key (its own
        .metric field doesn't match the key it was filed under) must not
        silently satisfy the limit -- the lookup key alone is not proof of
        which metric was actually measured."""
        contract = self._contract()
        wrong = ec.ResourceResult(metric="a_totally_different_metric", unit="count", subject_value=1)
        gate = ec.evaluate_resource_gate(contract, {"graph_cache_entries": wrong})
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["failed_metrics"], ["graph_cache_entries"])
        self.assertIn("metric mismatch", gate["detail"]["graph_cache_entries"])

    def test_exceeding_max_value_fails(self):
        contract = self._contract()
        result = ec.ResourceResult(metric="graph_cache_entries", unit="count", subject_value=64)
        gate = ec.evaluate_resource_gate(contract, {"graph_cache_entries": result})
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["failed_metrics"], ["graph_cache_entries"])

    def test_unit_mismatch_fails(self):
        contract = self._contract()
        result = ec.ResourceResult(metric="graph_cache_entries", unit="bytes", subject_value=1)
        gate = ec.evaluate_resource_gate(contract, {"graph_cache_entries": result})
        self.assertFalse(gate["passed"])
        self.assertIn("unit mismatch", gate["detail"]["graph_cache_entries"])

    def test_max_increase_pct_within_budget_passes(self):
        doc = _with_resource_limits(
            {"metric": "graph_cache_resident_bytes", "unit": "bytes", "max_increase_pct": 10.0}
        )
        contract = ec.parse_contract(doc, contract_id="X")
        result = ec.ResourceResult(
            metric="graph_cache_resident_bytes", unit="bytes", subject_value=105, control_value=100
        )
        gate = ec.evaluate_resource_gate(contract, {"graph_cache_resident_bytes": result})
        self.assertTrue(gate["passed"])

    def test_max_increase_pct_exceeded_fails(self):
        doc = _with_resource_limits(
            {"metric": "graph_cache_resident_bytes", "unit": "bytes", "max_increase_pct": 10.0}
        )
        contract = ec.parse_contract(doc, contract_id="X")
        result = ec.ResourceResult(
            metric="graph_cache_resident_bytes", unit="bytes", subject_value=200, control_value=100
        )
        gate = ec.evaluate_resource_gate(contract, {"graph_cache_resident_bytes": result})
        self.assertFalse(gate["passed"])

    def test_max_increase_pct_with_no_control_value_fails(self):
        doc = _with_resource_limits(
            {"metric": "graph_cache_resident_bytes", "unit": "bytes", "max_increase_pct": 10.0}
        )
        contract = ec.parse_contract(doc, contract_id="X")
        result = ec.ResourceResult(metric="graph_cache_resident_bytes", unit="bytes", subject_value=105)
        gate = ec.evaluate_resource_gate(contract, {"graph_cache_resident_bytes": result})
        self.assertFalse(gate["passed"])

    def test_zero_control_with_positive_subject_is_unbounded_growth_and_fails(self):
        doc = _with_resource_limits(
            {"metric": "graph_cache_resident_bytes", "unit": "bytes", "max_increase_pct": 10.0}
        )
        contract = ec.parse_contract(doc, contract_id="X")
        result = ec.ResourceResult(
            metric="graph_cache_resident_bytes", unit="bytes", subject_value=1, control_value=0
        )
        gate = ec.evaluate_resource_gate(contract, {"graph_cache_resident_bytes": result})
        self.assertFalse(gate["passed"])

    def test_zero_control_and_zero_subject_passes(self):
        doc = _with_resource_limits(
            {"metric": "graph_cache_resident_bytes", "unit": "bytes", "max_increase_pct": 10.0}
        )
        contract = ec.parse_contract(doc, contract_id="X")
        result = ec.ResourceResult(
            metric="graph_cache_resident_bytes", unit="bytes", subject_value=0, control_value=0
        )
        gate = ec.evaluate_resource_gate(contract, {"graph_cache_resident_bytes": result})
        self.assertTrue(gate["passed"])


class PromotionGateFailClosedIntegrationTests(unittest.TestCase):
    def _contract_with_limit(self):
        doc = _with_resource_limits({"metric": "graph_cache_entries", "unit": "count", "max_value": 32})
        return ec.parse_contract(doc, contract_id="X")

    def _passing_correctness_and_effects(self):
        correctness_gate = {"passed": True, "missing_checks": [], "failed_checks": []}
        aggregated_effects = {
            "target_kernel_gain_pct": 10.0, "end_to_end_gain_pct": 5.0,
            "max_control_regression_pct": 0.5,
        }
        return correctness_gate, aggregated_effects

    def test_contract_with_resource_limits_and_no_resource_gate_blocks_promotion(self):
        contract = self._contract_with_limit()
        correctness_gate, aggregated_effects = self._passing_correctness_and_effects()
        result = ec.evaluate_promotion_gate(
            contract, correctness_gate=correctness_gate, aggregated_effects=aggregated_effects,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("resource_gate" in r for r in result["reasons"]))

    def test_contract_with_resource_limits_and_failing_resource_gate_blocks_promotion(self):
        contract = self._contract_with_limit()
        correctness_gate, aggregated_effects = self._passing_correctness_and_effects()
        resource_gate = {"passed": False, "missing_metrics": [], "failed_metrics": ["graph_cache_entries"]}
        result = ec.evaluate_promotion_gate(
            contract, correctness_gate=correctness_gate, aggregated_effects=aggregated_effects,
            resource_gate=resource_gate,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("resource gate failed" in r for r in result["reasons"]))

    def test_contract_with_resource_limits_and_passing_resource_gate_promotes(self):
        contract = self._contract_with_limit()
        correctness_gate, aggregated_effects = self._passing_correctness_and_effects()
        resource_gate = {"passed": True, "missing_metrics": [], "failed_metrics": []}
        result = ec.evaluate_promotion_gate(
            contract, correctness_gate=correctness_gate, aggregated_effects=aggregated_effects,
            resource_gate=resource_gate,
        )
        self.assertTrue(result["passed"])

    def test_contract_without_resource_limits_is_unaffected_by_missing_resource_gate(self):
        contract = ec.parse_contract(_base_doc(), contract_id="X")
        correctness_gate, aggregated_effects = self._passing_correctness_and_effects()
        result = ec.evaluate_promotion_gate(
            contract, correctness_gate=correctness_gate, aggregated_effects=aggregated_effects,
        )
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
