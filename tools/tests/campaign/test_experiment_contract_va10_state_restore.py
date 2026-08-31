"""VA10 tests: the state_restore workload tag and state_restore_integrity
correctness token (RD58's real claim -- repeated multi-GPU state-restore
integrity, distinct from the generic multi_gpu_copy transfer workload and
from an absence-of-fault claim). No parser/gate logic changes -- these are
pure closed-vocabulary additions, exercised through the real, unmodified
parser and evaluate_correctness_gate().
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import contract as ec # noqa: E402


def _base_doc(**overrides: object) -> dict:
    doc = {
        "title": "VA10 fixture",
        "source": {
            "source_id": "stew675-rdna-boosts",
            "commits": ["deadbeefcafef00d"],
            "atomic_part": "va10-fixture",
        },
        "hypothesis": {
            "family": "mmq",
            "expected_effect": "correctness",
            "rationale": "state-restore integrity fixture",
        },
        "prerequisites": [],
        "scope": {
            "backend": "hip",
            "architectures": ["gfx1100"],
            "weight_types": ["q8_0"],
        },
        "positive": {"models": ["recipe/model"], "workloads": ["state_restore"]},
        "controls": {"models": ["control-recipe"], "workloads": ["decode"]},
        "boundary": {"dimensions": {}},
        "correctness": {"state_restore_integrity": "required"},
        "acceptance": {"max_control_regression_pct": 1},
    }
    doc.update(overrides)
    return doc


class StateRestoreVocabularyTests(unittest.TestCase):
    def test_state_restore_workload_accepted(self):
        contract = ec.parse_contract(_base_doc(), contract_id="VA10-FIXTURE")
        self.assertEqual(contract.positive.workloads, ("state_restore",))

    def test_state_restore_integrity_correctness_check_accepted(self):
        contract = ec.parse_contract(_base_doc(), contract_id="VA10-FIXTURE")
        self.assertEqual(contract.correctness.required_checks, ("state_restore_integrity",))

    def test_misspelled_workload_rejected(self):
        doc = _base_doc()
        doc["positive"] = {"models": ["recipe/model"], "workloads": ["state_restor"]}
        with self.assertRaisesRegex(ec.ExperimentContractError, "state_restor"):
            ec.parse_contract(doc, contract_id="X")

    def test_misspelled_correctness_token_rejected(self):
        doc = _base_doc()
        doc["correctness"] = {"state_restore_integryt": "required"}
        with self.assertRaises(ec.ExperimentContractError):
            ec.parse_contract(doc, contract_id="X")

    def test_multi_gpu_copy_remains_a_distinct_separate_tag(self):
        # Sanity: adding state_restore must not remove or alias the
        # existing generic transfer-workload tag.
        doc = _base_doc()
        doc["positive"] = {"models": ["recipe/model"], "workloads": ["multi_gpu_copy"]}
        contract = ec.parse_contract(doc, contract_id="X")
        self.assertEqual(contract.positive.workloads, ("multi_gpu_copy",))


class StateRestoreCorrectnessGateTests(unittest.TestCase):
    def _contract(self) -> object:
        return ec.parse_contract(_base_doc(), contract_id="VA10-FIXTURE")

    def test_required_check_with_matching_pass_passes(self):
        contract = self._contract()
        result = ec.CorrectnessResult(check="state_restore_integrity", passed=True, detail="ok")
        gate = ec.evaluate_correctness_gate(contract, {"state_restore_integrity": result})
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["missing_checks"], [])
        self.assertEqual(gate["failed_checks"], [])

    def test_required_check_missing_entirely_raises(self):
        contract = self._contract()
        # evaluate_correctness_gate() raises when required checks exist but
        # NO results were supplied at all -- fail-closed, not a silent pass.
        with self.assertRaises(ec.ExperimentContractError):
            ec.evaluate_correctness_gate(contract, {})

    def test_required_check_missing_from_a_nonempty_results_dict_fails(self):
        contract = self._contract()
        unrelated = ec.CorrectnessResult(check="some_other_check", passed=True, detail="")
        gate = ec.evaluate_correctness_gate(contract, {"some_other_check": unrelated})
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["missing_checks"], ["state_restore_integrity"])

    def test_matching_result_with_passed_false_fails(self):
        contract = self._contract()
        result = ec.CorrectnessResult(check="state_restore_integrity", passed=False, detail="SDMA fault observed")
        gate = ec.evaluate_correctness_gate(contract, {"state_restore_integrity": result})
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["failed_checks"], ["state_restore_integrity"])


if __name__ == "__main__":
    unittest.main()
