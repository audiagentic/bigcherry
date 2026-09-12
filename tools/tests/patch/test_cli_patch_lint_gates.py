"""CLI compatibility tests for the PA23 repository-static lint consumer."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.cli.main import build_parser  # noqa: E402
from bigcherry.cli.patch import cmd_patch_lint  # noqa: E402
from bigcherry.patch import gates, docs as patch_docs, registry as patch_registry  # noqa: E402
from bigcherry.patch import validation_policy  # noqa: E402


_FRAMEWORK_ADAPTER = (
    "schema = 1\n"
    '[[check]]\nid = "apply"\ncapability = "apply"\nvalidator = "apply"\n'
    '[[check]]\nid = "build"\ncapability = "build"\nvalidator = "build"\n'
)


def _write_package(
    root: Path,
    patch_id: str,
    *,
    order: int,
    state: str,
    kind: str,
    origin: str,
    tags: str = "",
    readme: str | None = None,
    adapter: str | None = None,
) -> Path:
    package = root / patch_id
    package.mkdir()
    (package / "patch.py").write_text(
        f'STATE = "{state}"\nPATCHES = []\n', encoding="utf-8",
    )
    (package / "patch.toml").write_text(
        "schema = 1\n"
        f'id = "{patch_id}"\n'
        f"order = {order}\n"
        f'state = "{state}"\n'
        f'kind = "{kind}"\n'
        f'origin = "{origin}"\n'
        'backend = "hip"\n'
        f"{tags}",
        encoding="utf-8",
    )
    if readme is not None:
        (package / "README.md").write_text(readme, encoding="utf-8")
    if adapter is not None:
        (package / "validation.toml").write_text(adapter, encoding="utf-8")
    (package / "SUMMARY.md").write_text(
        f"# {patch_id}\n\n**Status:** {state}\n**Plan item:** none\n",
        encoding="utf-8",
    )
    return package


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    def test_real_static_authorities_match_legacy_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patches = root / "patches"
            patches.mkdir()
            _write_package(
                patches, "0001_current", order=1, state="validated",
                kind="framework", origin="local", readme="# current\n",
                adapter=_FRAMEWORK_ADAPTER,
            )
            _write_package(
                patches, "0002_invalid", order=2, state="untested",
                kind="enhancement", origin="external-fork",
            )
            _write_package(
                patches, "0003_not_required", order=3, state="untested",
                kind="enhancement", origin="external-fork",
            )
            grandfathered = _write_package(
                patches, "0004_grandfathered", order=4, state="untested",
                kind="enhancement", origin="external-fork",
            )
            (patches / "0005_legacy.py").write_text(
                'STATE = "validated"\nPATCHES = []\n', encoding="utf-8",
            )
            (patches / "SUMMARY.md").write_text(
                "# 0005_legacy\n\n**Status:** validated\n**Plan item:** none\n",
                encoding="utf-8",
            )
            _write_package(
                patches, "0006_performance", order=6, state="validated",
                kind="enhancement", origin="local",
                tags='tags = ["optimization"]\n',
                readme="# performance\n\nControl versus subject only.\n",
            )

            sha = "a" * 40
            external_sources = root / "external-sources.toml"
            external_sources.write_text(
                "version = 1\n\n"
                "[[sources]]\n"
                'id = "example"\nrepo = "example/example"\nlocator = "example"\n\n'
                "[[sources.snapshots]]\n"
                'label = "head"\n'
                f'head = "{sha}"\nbase = "{sha}"\nactive = true\n\n'
                "[[sources.tracked]]\n"
                f'commit = "{sha}"\noriginal = "{sha}"\n'
                'title = "invalid"\nplan-item = "EX01"\n'
                'status = "ported-benched"\npatch = "0002_invalid"\n\n'
                "[[sources.tracked]]\n"
                f'commit = "{sha}"\noriginal = "{sha}"\n'
                'title = "grandfathered"\nplan-item = "EX02"\n'
                'status = "ported-benched"\npatch = "0004_grandfathered"\n',
                encoding="utf-8",
            )
            baseline = root / "baseline.json"
            baseline.write_text(
                json.dumps({
                    "schema_version": 1,
                    "policy_version": validation_policy.VALIDATION_PACKAGE_POLICY_VERSION,
                    "patches": {
                        "0004_grandfathered": {
                            "implementation_digest": _sha256(grandfathered / "patch.py"),
                            "patch_toml_digest": _sha256(grandfathered / "patch.toml"),
                            "tracked_statuses": ["ported-benched"],
                        },
                    },
                }),
                encoding="utf-8",
            )

            registry = patch_registry.load_registry(patches)
            summary_problems = [
                problem
                for descriptor in registry.descriptors
                for problem in patch_docs.check_summary_for_patch(descriptor, registry.root)
            ]
            package_report = validation_policy.check_validation_packages(
                root=patches,
                registry_path=patches,
                external_sources_path=external_sources,
                baseline_path=baseline,
            )
            performance_problems = tuple(
                problem
                for descriptor in registry.descriptors
                for problem in validation_policy.check_performance_evidence_for_patch(
                    descriptor, root=patches, assume_validated=False,
                )
            )
            expected = (
                tuple((*summary_problems, *package_report.problems, *performance_problems)),
                tuple(package_report.grandfathered),
            )
            report = gates.evaluate_repository_lint_gates(
                patches_dir=patches,
                external_sources_path=external_sources,
                validation_baseline_path=baseline,
            )

        self.assertEqual((report.problems, report.grandfathered), expected)
        outcomes = {
            (item.patch_id, item.result.id): item.result.status
            for item in report.results
        }
        self.assertEqual(outcomes[("0001_current", gates.GateId.G3)], gates.GateStatus.PASS)
        self.assertEqual(outcomes[("0002_invalid", gates.GateId.G3)], gates.GateStatus.FAIL)
        self.assertEqual(outcomes[("0003_not_required", gates.GateId.G3)], gates.GateStatus.NA)
        self.assertEqual(outcomes[("0004_grandfathered", gates.GateId.G3)], gates.GateStatus.PASS)
        self.assertEqual(outcomes[("0005_legacy", gates.GateId.G3)], gates.GateStatus.NA)
        self.assertEqual(outcomes[("0006_performance", gates.GateId.G3)], gates.GateStatus.FAIL)


if __name__ == "__main__":
    unittest.main()
