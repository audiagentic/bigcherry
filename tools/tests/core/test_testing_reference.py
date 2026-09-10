"""Guard the testing-reference authority boundaries against documentation drift."""

from __future__ import annotations

import re
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
        self.assertIn("Diagnostic PASS", document)
        self.assertNotIn(
            "docs/planning/active/validation-package-standard/VA14.md", document
        )

    def test_validation_policy_separates_lifecycle_axes(self) -> None:
        document = (TESTING_ROOT / "PATCH_VALIDATION.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Package state", document)
        self.assertIn("Source status", document)
        self.assertIn("`validated`, `rejected`, `superseded`", document)
        self.assertIn("`ported-benched`, `ported-validated`", document)
        self.assertIn("does not mean `ported-validated` is a package state", document)

    def test_rccl_runbook_matches_completed_gp07_tooling(self) -> None:
        document = (TESTING_ROOT / "RCCL_HETEROGENEOUS_RUNBOOK.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("GP07 is implemented", document)
        self.assertIn("--repetitions", document)
        self.assertIn("rechecks the homogeneous control", document)
        self.assertNotIn("until GP07 lands", document)
        self.assertNotIn("Current known gap", document)

    def test_patch_validation_pointer_does_not_duplicate_policy(self) -> None:
        document = (REPO_ROOT / "docs" / "reference" / "patches" / "PATCH_VALIDATION.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("sole canonical", document)
        self.assertIn("validation authority", document)
        self.assertIn("../testing/PATCH_VALIDATION.md", document)
        self.assertNotIn("--run-rd08-contract", document)
        self.assertNotIn("end_to_end_gain_pct", document)
        self.assertNotIn("experiment plan", document.lower())

    def test_reference_defers_to_live_contract_acceptance(self) -> None:
        test_doc = (TESTING_ROOT / "TEST.md").read_text(encoding="utf-8")
        policy_doc = (TESTING_ROOT / "PATCH_VALIDATION.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("experiment-contracts.toml", test_doc)
        self.assertIn("experiment-contracts.toml", policy_doc)
        self.assertNotIn("at least 1.0%", test_doc)
        self.assertNotIn("at most 1.0%", test_doc)
        self.assertNotIn("at least 10", test_doc)
        self.assertNotIn("3.0% promotion bar", test_doc)

    def test_empirical_and_historical_documents_are_scoped(self) -> None:
        coverage = (REPO_ROOT / "docs" / "archive" / "COVERAGE_AUDIT.md").read_text(
            encoding="utf-8"
        )
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

    def test_maintained_reference_links_resolve(self) -> None:
        documents = sorted((REPO_ROOT / "docs" / "reference").rglob("*.md"))
        link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
        heading_pattern = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)

        def heading_anchors(document: str) -> set[str]:
            anchors = set()
            for heading in heading_pattern.findall(document):
                slug = re.sub(r"<[^>]+>", "", heading.lower())
                slug = re.sub(r"[^a-z0-9 _-]", "", slug)
                anchors.add(re.sub(r"\s+", "-", slug).strip("-"))
            return anchors

        for source in documents:
            document = source.read_text(encoding="utf-8")
            for raw_target in link_pattern.findall(document):
                clean_target = raw_target.strip().strip("<>")
                target, _, anchor = clean_target.partition("#")
                if not target:
                    if anchor:
                        self.assertIn(
                            anchor,
                            heading_anchors(document),
                            f"{source.relative_to(REPO_ROOT)} -> #{anchor}",
                        )
                    continue
                if target.startswith(("http:", "https:", "mailto:")):
                    continue
                resolved = (source.parent / target).resolve()
                self.assertTrue(
                    resolved.exists(),
                    f"{source.relative_to(REPO_ROOT)} -> {target}",
                )
                if anchor and resolved.suffix.lower() == ".md":
                    target_doc = resolved.read_text(encoding="utf-8")
                    self.assertIn(
                        anchor,
                        heading_anchors(target_doc),
                        f"{source.relative_to(REPO_ROOT)} -> {target}#{anchor}",
                    )


if __name__ == "__main__":
    unittest.main()

