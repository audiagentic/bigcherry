from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import check


class CheckTests(TestCase):
    def test_tooling_hygiene_is_clean_for_a_minimal_valid_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            topic = root / "tools" / "lab" / "example"
            topic.mkdir(parents=True)
            (topic / "README.md").write_text("question and disposition\n", encoding="utf-8")
            self.assertEqual(check.tooling_hygiene(root), ())

    def test_tooling_hygiene_reports_bounded_findings_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            product = tools / "bigcherry"
            topic = tools / "lab" / "unknown"
            product.mkdir(parents=True)
            topic.mkdir(parents=True)
            (tools / "rogue.py").write_text("pass\n", encoding="utf-8")
            (tools / "run.ps1").write_text("Write-Output ok\n", encoding="utf-8")
            (product / "facade.py").write_text(
                "import importlib\n"
                "_CANONICAL = importlib.import_module('bigcherry.core.api')\n",
                encoding="utf-8",
            )
            (product / "bad.py").write_text(
                "from tools.lab.unknown import probe\n"
                "from pathlib import Path\n"
                "ROOT = Path(__file__).resolve().parents[2]\n",
                encoding="utf-8",
            )
            (topic / "probe.py").write_text("print('probe')\n", encoding="utf-8")
            disposition = root / "docs" / "planning" / "active" / "rationalisation"
            disposition.mkdir(parents=True)
            (disposition / "TOOL_DISPOSITION.md").write_text(
                "| `tmp/old_probe.py` | **DELETE** | obsolete |\n",
                encoding="utf-8",
            )
            (root / "tmp").mkdir()
            (root / "tmp" / "old_probe.py").write_text("pass\n", encoding="utf-8")

            findings = check.tooling_hygiene(root)
            finding_keys = [(item.code, item.path, item.message) for item in findings]
            self.assertEqual(finding_keys, sorted(finding_keys))
            codes = [item.code for item in findings]
            self.assertIn("TR14.TOP_LEVEL_SCRIPT", codes)
            self.assertIn("TR14.ENVIRONMENT_SCRIPT", codes)
            self.assertIn("TR14.ROOT_FACADE", codes)
            self.assertIn("TR14.PRODUCTION_LAB_IMPORT", codes)
            self.assertIn("TR14.FIXED_PARENT_DEPTH", codes)
            self.assertIn("TR14.LAB_METADATA", codes)
            self.assertIn("TR14.LAB_UNCLASSIFIED", codes)
            self.assertIn("TR14.DISPOSITION_DELETE_PENDING", codes)
            self.assertTrue(all(item.remediation for item in findings))

    def test_tooling_hygiene_is_registered_in_quick_and_preserves_overlay_failure(self) -> None:
        ids = [(spec.id, spec.tier) for spec in check.check_specs()]
        self.assertIn(("tooling-hygiene", "quick"), ids)
        specs = (
            check.CheckSpec("tooling-hygiene", "quick", lambda root: "TR14.ROOT_FACADE"),
            check.CheckSpec("source-audit", "default", lambda root: "overlay.vendor_sync: drift"),
        )
        with patch.object(check, "check_specs", return_value=specs):
            report = check.run_checks(root=Path("."), tier="default")
        self.assertFalse(report["passed"])
        self.assertEqual([row["id"] for row in report["checks"]], [
            "tooling-hygiene", "source-audit"
        ])
        self.assertIn("overlay.vendor_sync", report["checks"][1]["detail"])

    def test_retained_wrapper_allowlists_are_exact_and_contract_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            product = root / "tools" / "bigcherry"
            product.mkdir(parents=True)
            (product / "inventory.py").write_text(
                "import importlib\n"
                "importlib.import_module('bigcherry.tuning.inventory')\n",
                encoding="utf-8",
            )
            (product / "patcher.py").write_text(
                "import importlib\n"
                "importlib.import_module('bigcherry.unrelated')\n",
                encoding="utf-8",
            )

            findings = check.tooling_hygiene(root)

            self.assertEqual(
                [(item.code, item.path) for item in findings],
                [("TR14.ROOT_FACADE", "tools/bigcherry/patcher.py")],
            )

    def test_source_root_uses_path_authority_instead_of_fixed_parent_depth(self) -> None:
        source = (
            Path(__file__).resolve().parents[2]
            / "bigcherry"
            / "patch"
            / "source.py"
        ).read_text(encoding="utf-8")
        self.assertNotRegex(source, r"\.parents\[\d+\]")

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
