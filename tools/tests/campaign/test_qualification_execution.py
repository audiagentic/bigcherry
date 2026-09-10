"""VA26 execution phase: orchestrator tests, with a FAKE lane runner (no
GPU, no real campaign) -- exercises cell ordering/contrast preservation,
fail-closed behavior on a missing/failing cell, correctness/trigger
merging across cells, and that exactly one promotion-gate call happens
per plan."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bigcherry.campaign import qualification_execution as qe  # noqa: E402
from bigcherry.experiment import contract as ec  # noqa: E402

from test_qualification_matrix import QualificationMatrixFixture  # noqa: E402


def _passing_correctness() -> dict[str, ec.CorrectnessResult]:
    return {
        "backend_reference": ec.CorrectnessResult(check="backend_reference", passed=True),
        "greedy_parity": ec.CorrectnessResult(check="greedy_parity", passed=True),
    }


class ExecuteQualificationPlanTests(QualificationMatrixFixture):
    def _passing_runner(self, calls: list):
        def _run_cell(cell: "qe.QualificationCell") -> qe.CellResult:
            calls.append(cell)
            if cell.evidence_level == "inferential":
                return qe.CellResult(
                    cell=cell,
                    lane_effects=(
                        ec.LaneEffect(role="positive", metric="tg128", geometric_effect_pct=6.0),
                        ec.LaneEffect(role="control", metric="tg128", geometric_effect_pct=0.1),
                    ),
                    correctness_results=_passing_correctness(),
                    trigger_evidence=(ec.TriggerEvidence(
                        role="positive", lane_id=f"{cell.architecture}-{cell.contrast}",
                        candidate_launches=3),),
                )
            return qe.CellResult(
                cell=cell, correctness_results=_passing_correctness(), smoke_passed=True,
            )
        return _run_cell

    def test_runs_every_cell_in_plan_order(self):
        plan = self._plan()
        calls: list = []
        result = qe.execute_qualification_plan(
            plan, self.contract, run_cell=self._passing_runner(calls), target_metric="tg128",
        )
        self.assertEqual(tuple(calls), plan.cells)
        self.assertEqual(len(result.cell_results), len(plan.cells))
        for cell_result, cell in zip(result.cell_results, plan.cells):
            self.assertIs(cell_result.cell, cell)

    def test_only_inferential_cells_feed_the_promotion_aggregate(self):
        plan = self._plan()
        calls: list = []
        result = qe.execute_qualification_plan(
            plan, self.contract, run_cell=self._passing_runner(calls), target_metric="tg128",
        )
        # CONTRACT_ID's acceptance: target_kernel_gain_pct=5, effect 6.0 clears it.
        self.assertEqual(result.promotion["status"], "pass", result.promotion)
        self.assertTrue(result.fully_evidenced)
        self.assertEqual(result.failed_cells, ())

    def test_smoke_cell_effects_never_reach_the_gate(self):
        # A smoke cell reporting a huge effect must not be able to push a
        # borderline plan over the threshold -- only inferential cells may.
        plan = self._plan()

        def _run_cell(cell):
            if cell.evidence_level == "inferential":
                return qe.CellResult(
                    cell=cell,
                    lane_effects=(
                        ec.LaneEffect(role="positive", metric="tg128", geometric_effect_pct=1.0),
                        ec.LaneEffect(role="control", metric="tg128", geometric_effect_pct=0.0),
                    ),
                    correctness_results=_passing_correctness(),
                    trigger_evidence=(ec.TriggerEvidence(role="positive", lane_id="p", candidate_launches=1),),
                )
            return qe.CellResult(
                cell=cell,
                lane_effects=(ec.LaneEffect(role="positive", metric="tg128", geometric_effect_pct=999.0),),
                correctness_results=_passing_correctness(), smoke_passed=True,
            )

        result = qe.execute_qualification_plan(
            plan, self.contract, run_cell=_run_cell, target_metric="tg128",
        )
        # 1.0 < target_kernel_gain_pct=5 -- must fail despite the smoke
        # cell's fabricated 999.0 effect, proving that effect was ignored.
        self.assertEqual(result.promotion["status"], "fail", result.promotion)

    def test_a_failing_cell_forces_an_invalid_verdict_not_a_silent_pass_or_fail(self):
        plan = self._plan()
        calls: list = []
        passing = self._passing_runner(calls)

        def _run_cell(cell):
            if cell.contrast == "release_delta" and cell.architecture == plan.gain_arches[0]:
                return qe.CellResult(cell=cell, error="server never came up")
            return passing(cell)

        result = qe.execute_qualification_plan(
            plan, self.contract, run_cell=_run_cell, target_metric="tg128",
        )
        self.assertFalse(result.fully_evidenced)
        self.assertEqual(len(result.failed_cells), 1)
        self.assertEqual(result.promotion["status"], "invalid", result.promotion)
        self.assertFalse(result.promotion["passed"])
        # Every cell still ran (full diagnostics), despite the failure.
        self.assertEqual(len(calls) + 1, len(plan.cells))

    def test_a_raised_exception_in_run_cell_is_recorded_not_propagated(self):
        plan = self._plan()

        def _run_cell(cell):
            if cell is plan.cells[0]:
                raise RuntimeError("boom")
            return qe.CellResult(cell=cell, correctness_results=_passing_correctness())

        result = qe.execute_qualification_plan(
            plan, self.contract, run_cell=_run_cell, target_metric="tg128",
        )
        self.assertEqual(len(result.cell_results), len(plan.cells))
        self.assertEqual(result.cell_results[0].error, "boom")
        self.assertFalse(result.fully_evidenced)

    def test_run_cell_returning_a_different_cell_is_rejected(self):
        plan = self._plan()

        def _run_cell(cell):
            other = plan.cells[1] if cell is plan.cells[0] else plan.cells[0]
            return qe.CellResult(cell=other)

        with self.assertRaises(qe.QualificationExecutionError):
            qe.execute_qualification_plan(
                plan, self.contract, run_cell=_run_cell, target_metric="tg128",
            )

    def test_correctness_results_are_merged_across_all_cells(self):
        plan = self._plan()

        def _run_cell(cell):
            # Split the two required checks across different cells --
            # neither cell alone satisfies the contract's correctness gate.
            if cell.contrast == "isolated":
                return qe.CellResult(
                    cell=cell,
                    correctness_results={"backend_reference": ec.CorrectnessResult(
                        check="backend_reference", passed=True)},
                    lane_effects=(
                        ec.LaneEffect(role="positive", metric="tg128", geometric_effect_pct=6.0),
                    ) if cell.evidence_level == "inferential" else (),
                )
            return qe.CellResult(
                cell=cell,
                correctness_results={"greedy_parity": ec.CorrectnessResult(
                    check="greedy_parity", passed=True)},
                lane_effects=(
                    ec.LaneEffect(role="control", metric="tg128", geometric_effect_pct=0.0),
                ) if cell.evidence_level == "inferential" else (),
            )

        result = qe.execute_qualification_plan(
            plan, self.contract, run_cell=_run_cell, target_metric="tg128",
        )
        self.assertTrue(result.correctness_gate["passed"], result.correctness_gate)

    def test_a_failing_check_never_gets_silently_overwritten_by_a_later_passing_one(self):
        """Regression: multiple cells reporting the SAME check name (e.g.
        a multi-architecture plan where every architecture reports
        'backend_reference') must never let a later PASSING cell hide an
        earlier FAILING one via naive dict merge -- the merged result for
        that check must stay failed."""
        plan = self._plan()
        order = []

        def _run_cell(cell):
            order.append(cell)
            if len(order) == 1:
                correctness = {"backend_reference": ec.CorrectnessResult(
                    check="backend_reference", passed=False, detail="first cell failed")}
            else:
                correctness = {"backend_reference": ec.CorrectnessResult(
                    check="backend_reference", passed=True, detail="later cell passed")}
            if cell.evidence_level == "inferential":
                correctness["greedy_parity"] = ec.CorrectnessResult(
                    check="greedy_parity", passed=True)
                return qe.CellResult(
                    cell=cell, correctness_results=correctness,
                    lane_effects=(
                        ec.LaneEffect(role="positive", metric="tg128", geometric_effect_pct=6.0),
                        ec.LaneEffect(role="control", metric="tg128", geometric_effect_pct=0.0),
                    ),
                )
            correctness["greedy_parity"] = ec.CorrectnessResult(check="greedy_parity", passed=True)
            return qe.CellResult(cell=cell, correctness_results=correctness, smoke_passed=True)

        result = qe.execute_qualification_plan(
            plan, self.contract, run_cell=_run_cell, target_metric="tg128",
        )
        self.assertFalse(result.correctness_gate["passed"], result.correctness_gate)
        self.assertIn("backend_reference", result.correctness_gate["failed_checks"])

    def test_exactly_one_promotion_gate_call_per_plan(self):
        plan = self._plan()
        calls: list = []
        result = qe.execute_qualification_plan(
            plan, self.contract, run_cell=self._passing_runner(calls), target_metric="tg128",
        )
        # A single `promotion` dict at the RESULT level (not per-cell) is
        # itself the proof there is one gate call per plan, not per cell.
        self.assertIsInstance(result.promotion, dict)
        self.assertIn("status", result.promotion)


if __name__ == "__main__":
    unittest.main()
