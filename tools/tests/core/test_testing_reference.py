"""Guard the testing-reference authority boundaries against documentation drift."""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TESTING_ROOT = REPO_ROOT / "docs" / "reference" / "testing"


class TestingReferenceTests(unittest.TestCase):
    def test_reference_index_lists_all_testing_authorities(self) -> None:
        index = (REPO_ROOT / "docs" / "reference" / "README.md").read_text(
            encoding="utf-8"
        )
        for name in (
            "testing/README.md",
            "testing/TEST.md",
            "testing/PATCH_VALIDATION.md",
            "testing/MULTI_GPU_LARGE_MODEL_VALIDATION.md",
            "testing/RCCL_HETEROGENEOUS_RUNBOOK.md",
            "testing/COVERAGE_AUDIT.md",
        ):
            self.assertIn(name, index)

    def test_hardware_free_gate_does_not_prescribe_unscoped_apply(self) -> None:
        document = (TESTING_ROOT / "TEST.md").read_text(encoding="utf-8")
        start = document.index("## Hardware-free repository gate")
        end = document.index("### Running a validation campaign", start)
        gate = document[start:end]

        self.assertIn("PYTHONPATH=tools python -m bigcherry check", gate)
        code = "\n".join(re.findall(r"```bash\n(.*?)```", gate, re.DOTALL))
        for line in code.splitlines():
            if "bigcherry apply" in line:
                self.assertIn("--source", line)
                self.assertIn("--dry-run", line)

    def test_campaign_capabilities_and_status_boundary_are_documented(self) -> None:
        document = (TESTING_ROOT / "PATCH_VALIDATION.md").read_text(
            encoding="utf-8"
        )
        for flag in (
            "--run-rd08-lanes",
            "--run-rd08-contract",
            "--run-rd04-benchmark",
            "--run-rd58-state-restore",
            "--run-rd73-contract",
        ):
            self.assertIn(flag, document)
        self.assertIn("eligible_for_validated_state", document)
        self.assertIn("diagnostic PASS", document)
        self.assertNotIn(
            "docs/planning/active/validation-package-standard/VA14.md", document
        )

    def test_rd73_reference_tracks_live_contract_acceptance(self) -> None:
        contract_data = tomllib.loads(
            (REPO_ROOT / "config" / "experiment-contracts.toml").read_text(
                encoding="utf-8"
            )
        )
        acceptance = contract_data["contract"]["RD73-STABLE-GRAPH-CACHE-KEY"][
            "acceptance"
        ]
        self.assertEqual(acceptance["end_to_end_gain_pct"], 1.0)
        self.assertEqual(acceptance["max_control_regression_pct"], 1.0)
        self.assertEqual(acceptance["min_paired_rounds"], 10)
        test_doc = (TESTING_ROOT / "TEST.md").read_text(encoding="utf-8")
        self.assertIn("at least 1.0%", test_doc)
        self.assertIn("at most 1.0%", test_doc)
        self.assertIn("at least 10", test_doc)
        self.assertNotIn("3.0% promotion bar", test_doc)

    def test_empirical_and_historical_documents_are_scoped(self) -> None:
        coverage = (TESTING_ROOT / "COVERAGE_AUDIT.md").read_text(encoding="utf-8")
        multi_gpu = (TESTING_ROOT / "MULTI_GPU_LARGE_MODEL_VALIDATION.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Historical coverage snapshot", coverage)
        self.assertIn("universal prohibition", multi_gpu)
        self.assertIn("canonical contract/evidence record", multi_gpu)

    def test_testing_markdown_has_no_known_dead_va14_reference(self) -> None:
        for path in TESTING_ROOT.glob("*.md"):
            document = path.read_text(encoding="utf-8")
            self.assertIsNone(
                re.search(
                    r"docs/planning/active/validation-package-standard/VA14\.md",
                    document,
                ),
                path.name,
            )


if __name__ == "__main__":
    unittest.main()
