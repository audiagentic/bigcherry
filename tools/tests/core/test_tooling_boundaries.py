"""TR02 static checks for the tooling architecture transition."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = REPO_ROOT / "tools"
PRODUCT_ROOT = TOOLS_ROOT / "bigcherry"

# These are intentionally transitional until TR05 classifies or removes them.
TOP_LEVEL_TOOLS_ALLOWLIST = {
    "residency_gates.py",
    "verify_slice_a.py",
}
PROTECTED_DOMAINS = {
    "release",
    "source",
    "build",
    "experiment",
    "patch",
    "campaign",
    "tuning",
}
CANONICAL_MOVED_FILES = (
    PRODUCT_ROOT / "analysis" / "candidate_report.py",
    PRODUCT_ROOT / "release" / "pin.py",
)


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.py") if "__pycache__" not in path.parts
    )


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


class ToolingBoundaryTests(unittest.TestCase):
    def test_lab_is_not_a_python_package(self) -> None:
        self.assertFalse((TOOLS_ROOT / "lab" / "__init__.py").exists())

    def test_lab_template_declares_management_fields(self) -> None:
        template = (TOOLS_ROOT / "lab" / "_template" / "README.md").read_text(
            encoding="utf-8"
        )
        for required in (
            "Plan item:",
            "Status:",
            "Owner:",
            "Question state:",
            "Canonical-state mutation:",
            "exactly one:",
            "never evidence authority",
        ):
            self.assertIn(required, template)

    def test_lab_template_runner_is_self_contained(self) -> None:
        runner = (TOOLS_ROOT / "lab" / "_template" / "run.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("bigcherry", runner)
        self.assertNotIn("tools.tests", runner)

    def test_top_level_tools_are_explicitly_allowlisted_during_transition(self) -> None:
        actual = {path.name for path in TOOLS_ROOT.glob("*.py")}
        self.assertEqual(actual, TOP_LEVEL_TOOLS_ALLOWLIST)

    def test_canonical_moved_tools_do_not_assume_parent_depth(self) -> None:
        for path in CANONICAL_MOVED_FILES:
            source = path.read_text(encoding="utf-8")
            self.assertNotRegex(source, r"\.parents\[\d+\]")

    def test_production_does_not_import_lab_or_tests(self) -> None:
        violations: list[str] = []
        for path in _python_files(PRODUCT_ROOT):
            for imported in _imports(path):
                if imported == "tools.lab" or imported.startswith("tools.lab."):
                    violations.append(f"{path}: {imported}")
                if imported == "tools.tests" or imported.startswith("tools.tests."):
                    violations.append(f"{path}: {imported}")
        self.assertEqual(violations, [])

    def test_domain_modules_do_not_import_cli_or_analysis(self) -> None:
        violations: list[str] = []
        for domain in PROTECTED_DOMAINS:
            root = PRODUCT_ROOT / domain
            if not root.exists():
                continue
            for path in _python_files(root):
                for imported in _imports(path):
                    if imported == "bigcherry.cli" or imported.startswith(
                        "bigcherry.cli."
                    ):
                        violations.append(f"{path}: {imported}")
                    if imported == "bigcherry.analysis" or imported.startswith(
                        "bigcherry.analysis."
                    ):
                        violations.append(f"{path}: {imported}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
