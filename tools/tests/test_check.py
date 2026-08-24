from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from bigcherry import check


class CheckTests(TestCase):
    def test_deterministic_order_and_success(self) -> None:
        specs = (
            check.CheckSpec("a", "quick", lambda root: None),
            check.CheckSpec("b", "default", lambda root: None),
        )
        with patch.object(check, "check_specs", return_value=specs):
            report = check.run_checks(root=Path("."))
        self.assertTrue(report["passed"])
        self.assertEqual([row["id"] for row in report["checks"]], ["a", "b"])

    def test_multiple_failures_are_aggregated(self) -> None:
        specs = (
            check.CheckSpec("a", "quick", lambda root: "first"),
            check.CheckSpec("b", "quick", lambda root: "second"),
        )
        with patch.object(check, "check_specs", return_value=specs):
            report = check.run_checks(root=Path("."), tier="quick")
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["checks"]), 2)

    def test_fail_fast_stops_after_first_failure(self) -> None:
        calls: list[str] = []
        specs = (
            check.CheckSpec("a", "quick", lambda root: calls.append("a") or "bad"),
            check.CheckSpec("b", "quick", lambda root: calls.append("b") or None),
        )
        with patch.object(check, "check_specs", return_value=specs):
            report = check.run_checks(root=Path("."), tier="quick", fail_fast=True)
        self.assertFalse(report["passed"])
        self.assertEqual(calls, ["a"])

    def test_exception_is_a_failure(self) -> None:
        def broken(root: Path) -> None:
            raise RuntimeError("boom")
        with patch.object(check, "check_specs", return_value=(check.CheckSpec("x", "quick", broken),)):
            report = check.run_checks(root=Path("."), tier="quick")
        self.assertIn("exception", report["checks"][0]["detail"])

    def test_tier_filters(self) -> None:
        specs = (
            check.CheckSpec("q", "quick", lambda root: None),
            check.CheckSpec("d", "default", lambda root: None),
            check.CheckSpec("f", "full", lambda root: None),
        )
        with patch.object(check, "check_specs", return_value=specs):
            report = check.run_checks(root=Path("."), tier="quick")
        self.assertEqual([row["id"] for row in report["checks"]], ["q"])

    def test_cli_json(self) -> None:
        with patch.object(check, "run_checks", return_value={"passed": True, "checks": []}) as run:
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "check.json"
                self.assertEqual(check.main(["--quick", "--json", str(output)]), 0)
                run.assert_called_once()
                self.assertEqual(json.loads(output.read_text())["passed"], True)

    def test_invalid_tier(self) -> None:
        with self.assertRaises(ValueError):
            check.run_checks(root=Path("."), tier="nope")
