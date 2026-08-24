from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry import patch_validation as pv


class UniversalPlanIntegrationTests(unittest.TestCase):
    def test_control_subject_apply_and_build_can_reach_pass(self) -> None:
        plan = pv.build_validation_plan(
            "9999_example",
            [
                pv.CheckSpec("apply", "apply", "apply", True),
                pv.CheckSpec("build", "build", "build", True),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            artifact = {"path": "evidence.json", "sha256": "a" * 64}
            context = pv.ValidationContext(
                descriptor=None, base_revision="b" * 40,
                control_source=run_dir, subject_source=run_dir,
                control_tree="c" * 40, subject_tree="d" * 40,
                build_identities={"control": "control-id", "subject": "subject-id"},
                build_evidence={
                    role: {
                        "build_id": f"{role}-id", "source_tree": tree,
                        "architecture": "gfx1100", "options": {"mode": role},
                        "compile_commands": artifact, "runtime_bundle": artifact,
                    }
                    for role, tree in (("control", "c" * 40), ("subject", "d" * 40))
                },
                apply_evidence={
                    "control": {"verified": True, "idempotent": True, "artifact": artifact},
                    "subject": {"verified": True, "idempotent": True, "artifact": artifact},
                },
                run_dir=run_dir,
            )
            with patch.object(pv, "_verified_source_tree", return_value=True), \
                 patch.object(pv, "_artifact_is_bound", return_value=True):
                results = {spec.check_id: pv.evaluate_check(spec, context) for spec in plan.checks}
        verdict = pv.compute_verdict(plan, results)
        self.assertTrue(verdict.eligible)
        self.assertTrue(all(result.status == pv.PASS for result in verdict.results))
