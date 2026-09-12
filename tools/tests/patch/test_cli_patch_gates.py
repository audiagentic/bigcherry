"""CLI contract tests for the PA22 shared patch-gates consumer."""

from __future__ import annotations

import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.cli.patch import cmd_patch_gates  # noqa: E402
from bigcherry.patch.gates import GateId, GateResult, GateStatus  # noqa: E402


def _args(**overrides: object) -> SimpleNamespace:
    values = {
        "patch_id": "P1",
        "intent": "validate",
        "source": None,
        "rebase_report": None,
        "all_report": None,
        "no_legacy_grandfather": False,
        "json": True,
        "llama_root": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class PatchGatesCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptor = SimpleNamespace(
            patch_id="P1", backend="agnostic", state="untested",
        )
        self.module = SimpleNamespace(
            patch_id="P1", content_hash="hash", state="untested",
        )
        self.composition = SimpleNamespace(modules=(self.module,))
        self.cfg = SimpleNamespace(pinned="b10901", sources={})
        self.registry = SimpleNamespace(get=mock.Mock(return_value=self.descriptor))
        self.repo = mock.Mock()
        self.repo.resolve_ref.side_effect = ["a" * 40, "b" * 40]

    def test_json_reports_all_applicable_gates_and_pass_exit(self) -> None:
        results = (
            GateResult(GateId.G0, GateStatus.PASS, "composition", "test"),
            GateResult(GateId.G1, GateStatus.NA, "documentation", "test"),
            GateResult(GateId.G2, GateStatus.PASS, "rebase", "test"),
            GateResult(GateId.G3, GateStatus.PASS, "package", "test"),
        )
        import bigcherry.cli.patch as cli_patch

        with (
            mock.patch.object(cli_patch.campaign_config, "load", return_value=self.cfg),
            mock.patch.object(cli_patch.patch_registry, "load_registry", return_value=self.registry),
            mock.patch.object(cli_patch.patchset, "catalog", return_value=[self.module]),
            mock.patch.object(
                cli_patch.patchset, "expand_composition",
                return_value=SimpleNamespace(expanded=("P1",)),
            ),
            mock.patch.object(cli_patch.patchset, "resolve_exact", return_value=self.composition),
            mock.patch.object(cli_patch, "UpstreamRepository", return_value=self.repo),
            mock.patch.object(cli_patch.patch_gates, "evaluate_patch_gates", return_value=results),
        ):
            output = StringIO()
            with redirect_stdout(output):
                exit_code = cmd_patch_gates(_args())

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["composition"], ["P1"])
        self.assertEqual([gate["id"] for gate in payload["gates"]], ["G0", "G1", "G2", "G3"])

    def test_fail_or_blocked_gate_returns_one(self) -> None:
        import bigcherry.cli.patch as cli_patch

        results = (GateResult(GateId.G0, GateStatus.BLOCKED, "composition", "test", ("missing",)),)
        with (
            mock.patch.object(cli_patch.campaign_config, "load", return_value=self.cfg),
            mock.patch.object(cli_patch.patch_registry, "load_registry", return_value=self.registry),
            mock.patch.object(cli_patch.patchset, "catalog", return_value=[self.module]),
            mock.patch.object(
                cli_patch.patchset, "expand_composition",
                return_value=SimpleNamespace(expanded=("P1",)),
            ),
            mock.patch.object(cli_patch.patchset, "resolve_exact", return_value=self.composition),
            mock.patch.object(cli_patch, "UpstreamRepository", return_value=self.repo),
            mock.patch.object(cli_patch.patch_gates, "evaluate_patch_gates", return_value=results),
        ):
            exit_code = cmd_patch_gates(_args())
        self.assertEqual(exit_code, 1)

    def test_build_requires_a_canonical_source(self) -> None:
        import bigcherry.cli.patch as cli_patch

        with (
            mock.patch.object(cli_patch.campaign_config, "load", return_value=self.cfg),
            mock.patch.object(cli_patch.patch_registry, "load_registry", return_value=self.registry),
            mock.patch.object(cli_patch.patchset, "catalog", return_value=[self.module]),
        ):
            exit_code = cmd_patch_gates(_args(intent="build"))
        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
