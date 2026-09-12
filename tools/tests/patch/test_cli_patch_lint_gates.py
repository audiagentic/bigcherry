"""CLI compatibility tests for the PA23 repository-static lint consumer."""

from __future__ import annotations

import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.cli.main import build_parser  # noqa: E402
from bigcherry.cli.patch import cmd_patch_lint  # noqa: E402
from bigcherry.patch import gates  # noqa: E402


class PatchLintCliTests(unittest.TestCase):
    def _args(self, *, json_output: bool) -> SimpleNamespace:
        return SimpleNamespace(json=json_output)

    def test_json_shape_and_clean_exit_are_unchanged(self) -> None:
        report = gates.LintGateReport(results=(), problems=(), grandfathered=("P4",))
        with (
            mock.patch("bigcherry.cli.patch.patch_catalog.cross_check", return_value=[])
            as cross_check,
            mock.patch("bigcherry.cli.patch.patch_gates.evaluate_repository_lint_gates", return_value=report)
            as evaluate,
            mock.patch("bigcherry.patch.validation_policy.check_validation_packages") as legacy_packages,
            mock.patch("bigcherry.patch.validation_policy.check_performance_evidence") as legacy_performance,
        ):
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = cmd_patch_lint(self._args(json_output=True))

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        expected = json.dumps(
            {"passed": True, "problems": [], "grandfathered": ["P4"]},
            indent=2,
            sort_keys=True,
        ) + "\n"
        self.assertEqual(stdout.getvalue(), expected)
        cross_check.assert_called_once_with(
            verify_validation_evidence=False,
            allow_legacy_grandfather=True,
            check_summaries=False,
        )
        evaluate.assert_called_once_with()
        legacy_packages.assert_not_called()
        legacy_performance.assert_not_called()

    def test_human_failure_preserves_problem_order_and_grandfather_text(self) -> None:
        report = gates.LintGateReport(
            results=(),
            problems=("SUMMARY problem", "package problem", "performance problem"),
            grandfathered=("P4",),
        )
        with (
            mock.patch("bigcherry.cli.patch.patch_catalog.cross_check", return_value=["catalog problem"]),
            mock.patch("bigcherry.cli.patch.patch_gates.evaluate_repository_lint_gates", return_value=report),
        ):
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = cmd_patch_lint(self._args(json_output=False))

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(
            stderr.getvalue(),
            "catalog problem\nSUMMARY problem\npackage problem\nperformance problem\n"
            "P4: structurally grandfathered (non-current, not failing)\n",
        )

    def test_internal_lint_intent_is_not_exposed_by_patch_gates_cli(self) -> None:
        parser = build_parser()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["patch-gates", "P1", "--intent", "lint"])


if __name__ == "__main__":
    unittest.main()
