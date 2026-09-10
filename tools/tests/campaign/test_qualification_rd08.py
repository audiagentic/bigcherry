"""VA26 execution phase: RD08 adapter tests, with every real build/measure
producer faked (no GPU, no compiler) -- exercises build-cache reuse across
shared compositions, per-architecture (not per-cell) correctness caching,
and that a real correctness FAILURE reaches the plan as evidence, not an
execute_qualification_plan()-level error."""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bigcherry.campaign import qualification_execution as qe  # noqa: E402
from bigcherry.campaign import qualification_rd08 as rd08  # noqa: E402
from bigcherry.experiment import contract as ec  # noqa: E402

from test_qualification_matrix import QualificationMatrixFixture  # noqa: E402


class _FakeBuildEvidence:
    def __init__(self, architecture: str, patch_set_id: str):
        self.effective_configure = ("fixed-configure", architecture)
        self._patch_set_id = patch_set_id

    def campaign_identity(self):
        return {"patch_set_id": self._patch_set_id}


class MakeRd08RunCellTests(QualificationMatrixFixture):
    def setUp(self) -> None:
        super().setUp()
        self._tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp2.cleanup)
        self.work_root = Path(self._tmp2.name)

        self.build_tree_calls: list[dict] = []
        self.materialize_calls: list[dict] = []
        self.correctness_calls: list[dict] = []
        self.lane_calls: list[dict] = []
        self.trigger_calls: list[dict] = []

        def _fake_materialize_composition(**kwargs):
            self.materialize_calls.append(kwargs)
            path = self.work_root / "src" / f"src-{len(self.materialize_calls)}"
            path.mkdir(parents=True, exist_ok=True)
            return path

        def _fake_build_tree(**kwargs):
            self.build_tree_calls.append(kwargs)
            bin_dir = self.work_root / "bin" / f"bin-{len(self.build_tree_calls)}"
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / "llama-bench").write_text("fake", encoding="utf-8")
            return bin_dir

        def _fake_capture_build_evidence(*args, architecture=None, **kwargs):
            return _FakeBuildEvidence(architecture, f"composition-{len(self.build_tree_calls)}")

        def _fake_assert_parity(control_evidence, subject_evidence, *, patch_id):
            pass

        def _fake_generate_registry(**kwargs):
            pass

        self.correctness_result_by_arch: dict[str, ec.CorrectnessResult] = {}

        def _fake_run_rd08_contract_correctness(*, amdgpu_targets, **kwargs):
            self.correctness_calls.append({"amdgpu_targets": amdgpu_targets, **kwargs})
            result = self.correctness_result_by_arch.get(
                amdgpu_targets,
                ec.CorrectnessResult(check="bit_identical", passed=True, detail="fake pass"),
            )
            return {"results": {"bit_identical": result}}

        def _fake_run_rd08_validation_lanes(**kwargs):
            self.lane_calls.append(kwargs)
            return {"effects": (
                ec.LaneEffect(role="positive", metric="tg128", geometric_effect_pct=2.0),
                ec.LaneEffect(role="control", metric="pp512", geometric_effect_pct=0.0),
            )}

        def _fake_run_rd08_contract_trigger(**kwargs):
            self.trigger_calls.append(kwargs)
            return {"evidence": (
                ec.TriggerEvidence(role="positive", lane_id="rd08-decode-subject", candidate_launches=1),
                ec.TriggerEvidence(role="control", lane_id="rd08-decode-control", candidate_launches=0),
            )}

        patches = [
            mock.patch.object(rd08.patch_source, "materialize_composition", side_effect=_fake_materialize_composition),
            mock.patch.object(rd08, "build_tree", side_effect=_fake_build_tree),
            mock.patch.object(rd08, "capture_completed_build_evidence", side_effect=_fake_capture_build_evidence),
            mock.patch.object(rd08, "assert_validation_subject_parity", side_effect=_fake_assert_parity),
            mock.patch.object(rd08, "generate_registry", side_effect=_fake_generate_registry),
            mock.patch.object(rd08, "run_rd08_contract_correctness", side_effect=_fake_run_rd08_contract_correctness),
            mock.patch.object(rd08, "run_rd08_validation_lanes", side_effect=_fake_run_rd08_validation_lanes),
            mock.patch.object(rd08, "run_rd08_contract_trigger", side_effect=_fake_run_rd08_contract_trigger),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])

    def _make_run_cell(self):
        return rd08.make_rd08_run_cell(
            contract=self.contract, base_revision="deadbeef" * 5,
            hip_path=self.work_root / "hip", model=self.work_root / "model.gguf",
            model_ref="test-model", marker_regex="BIGCHERRY_PATCH_HIT",
            worktree_root=self.work_root / "worktrees", build_root=self.work_root / "builds",
            run_root=self.work_root / "runs", build_env={},
        )

    def test_runs_all_cells_and_produces_full_evidence_on_a_passing_plan(self):
        plan = self._plan()
        run_cell = self._make_run_cell()
        result = qe.execute_qualification_plan(
            plan, self.contract, run_cell=run_cell, target_metric="tg128",
        )
        self.assertTrue(result.fully_evidenced)
        self.assertEqual(len(result.cell_results), len(plan.cells))

    def test_correctness_is_cached_per_architecture_not_per_cell(self):
        plan = self._plan()
        run_cell = self._make_run_cell()
        qe.execute_qualification_plan(plan, self.contract, run_cell=run_cell, target_metric="tg128")
        architectures = {c["amdgpu_targets"] for c in self.correctness_calls}
        # One correctness call per DISTINCT architecture in the plan, even
        # though each architecture has 2 contrasts (isolated/release_delta).
        self.assertEqual(len(self.correctness_calls), len(architectures))
        self.assertEqual(architectures, set(plan.gain_arches) | set(plan.safety_arches))

    def test_shared_compositions_are_built_only_once(self):
        plan = self._plan()
        run_cell = self._make_run_cell()
        qe.execute_qualification_plan(plan, self.contract, run_cell=run_cell, target_metric="tg128")
        # 4 distinct compositions (native, native+candidate, release,
        # release+candidate) x 2 architectures = 8 builds, not one per cell
        # x 2 sides (which would be 4 cells x 2 = 8 anyway here, but the
        # real assertion is that a composition shared across contrasts --
        # e.g. isolated control == native, reused by nothing else in THIS
        # fixture, but release_delta control == release, likewise -- is not
        # rebuilt if requested twice). Force that by calling run_cell twice
        # on the SAME cell and confirming no additional build happens.
        before = len(self.build_tree_calls)
        run_cell(plan.cells[0])
        after = len(self.build_tree_calls)
        self.assertEqual(before, after)

    def test_a_real_correctness_failure_is_evidence_not_an_error(self):
        plan = self._plan()
        self.correctness_result_by_arch[plan.gain_arches[0]] = ec.CorrectnessResult(
            check="bit_identical", passed=False,
            detail="RD08 correctness evidence failed for shape='ffn' seed=1",
        )
        # RD08's real contract requires "bit_identical" specifically (not
        # this fixture's generic backend_reference/greedy_parity checks).
        rd08_contract = dataclasses.replace(
            self.contract,
            correctness=ec.CorrectnessRequirements(required_checks=("bit_identical",)),
        )
        run_cell = self._make_run_cell()
        result = qe.execute_qualification_plan(
            plan, rd08_contract, run_cell=run_cell, target_metric="tg128",
        )
        self.assertTrue(result.fully_evidenced)
        self.assertFalse(result.correctness_gate["passed"])
        self.assertIn("bit_identical", result.correctness_gate["failed_checks"])
        self.assertEqual(result.promotion["status"], "fail")
        self.assertFalse(result.promotion["passed"])

    def test_a_real_build_failure_forces_an_invalid_verdict(self):
        plan = self._plan()

        def _boom(**kwargs):
            raise RuntimeError("compiler crashed")

        with mock.patch.object(rd08, "build_tree", side_effect=_boom):
            run_cell = self._make_run_cell()
            result = qe.execute_qualification_plan(
                plan, self.contract, run_cell=run_cell, target_metric="tg128",
            )
        self.assertFalse(result.fully_evidenced)
        self.assertEqual(result.promotion["status"], "invalid")

    def test_lane_and_trigger_run_once_per_cell_not_cached(self):
        plan = self._plan()
        run_cell = self._make_run_cell()
        qe.execute_qualification_plan(plan, self.contract, run_cell=run_cell, target_metric="tg128")
        self.assertEqual(len(self.lane_calls), len(plan.cells))
        self.assertEqual(len(self.trigger_calls), len(plan.cells))


if __name__ == "__main__":
    unittest.main()
