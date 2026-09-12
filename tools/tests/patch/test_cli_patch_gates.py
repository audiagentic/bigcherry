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

from bigcherry.cli.main import build_parser  # noqa: E402
from bigcherry.cli.patch import cmd_patch_gates  # noqa: E402
from bigcherry.campaign import resolution as campaign_resolution  # noqa: E402
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

    def test_source_build_forwards_ordered_composition_and_gate_context(self) -> None:
        import bigcherry.cli.patch as cli_patch

        first = SimpleNamespace(patch_id="A", content_hash="a", state="validated")
        focal = SimpleNamespace(patch_id="P1", content_hash="p", state="validated")
        composition = SimpleNamespace(modules=(first, focal))
        self.cfg.sources = {"bigcherry": SimpleNamespace(backend="hip")}
        self.repo.resolve_ref.side_effect = ["a" * 40, "b" * 40]
        rebase_report = {"report": "rebase"}
        all_report = {"report": "all"}
        results = (GateResult(GateId.G0, GateStatus.PASS, "composition", "test"),)
        with (
            mock.patch.object(cli_patch.campaign_config, "load", return_value=self.cfg),
            mock.patch.object(cli_patch.patch_registry, "load_registry", return_value=self.registry),
            mock.patch.object(cli_patch.patchset, "catalog", return_value=[first, focal]),
            mock.patch.object(
                campaign_resolution,
                "resolve_canonical_selection",
                return_value=SimpleNamespace(patch_ids=("A", "P1")),
            ) as resolve_source,
            mock.patch.object(
                cli_patch.patchset, "resolve_exact", return_value=composition,
            ) as resolve_exact,
            mock.patch.object(cli_patch.patch_rebase, "load_report",
                              side_effect=(rebase_report, all_report)) as load_report,
            mock.patch.object(cli_patch, "UpstreamRepository", return_value=self.repo),
            mock.patch.object(cli_patch.patch_gates, "evaluate_patch_gates", return_value=results) as evaluate,
        ):
            output = StringIO()
            with redirect_stdout(output):
                exit_code = cmd_patch_gates(_args(
                    intent="build", source="bigcherry",
                    rebase_report="rebase.json", all_report="all.json",
                    no_legacy_grandfather=True,
                ))

        self.assertEqual(exit_code, 0)
        resolve_source.assert_called_once_with(
            "bigcherry", self.cfg, [first, focal],
            catalog_directory=cli_patch.paths.PATCHES,
        )
        resolve_exact.assert_called_once_with(
            ("A", "P1"), directory=cli_patch.paths.PATCHES, allow_rejected=False,
        )
        self.assertEqual(load_report.call_args_list[0].args[0], Path("rebase.json"))
        self.assertEqual(load_report.call_args_list[1].args[0], Path("all.json"))
        context = evaluate.call_args.args[0]
        self.assertEqual(tuple(m.patch_id for m in context.composition.modules), ("A", "P1"))
        self.assertEqual(context.recipe_patch_ids, frozenset({"A", "P1"}))
        self.assertEqual(context.patch_context.backend, "hip")
        self.assertEqual(context.patch_context.source, "bigcherry")
        self.assertEqual(
            context.catalog_states,
            {"A": "validated", "P1": "validated"},
        )
        self.assertEqual(context.rebase_report, rebase_report)
        self.assertEqual(context.coverage_report, all_report)
        self.assertEqual(context.resolved_base_revision, "a" * 40)
        self.assertEqual(context.target_revision, "b" * 40)
        self.assertFalse(context.allow_legacy_grandfather)
        self.assertEqual(self.repo.resolve_ref.call_args_list[0].args, (self.cfg.pinned,))
        self.assertEqual(self.repo.resolve_ref.call_args_list[1].args, ("HEAD",))
        self.assertIn('"composition": [', output.getvalue())

    def test_source_mode_rejects_focal_outside_canonical_selection(self) -> None:
        import bigcherry.cli.patch as cli_patch

        self.cfg.sources = {"bigcherry": SimpleNamespace(backend="hip")}
        with (
            mock.patch.object(cli_patch.campaign_config, "load", return_value=self.cfg),
            mock.patch.object(cli_patch.patch_registry, "load_registry", return_value=self.registry),
            mock.patch.object(cli_patch.patchset, "catalog", return_value=[self.module]),
            mock.patch.object(
                campaign_resolution,
                "resolve_canonical_selection",
                return_value=SimpleNamespace(patch_ids=("OTHER",)),
            ) as resolve_source,
            mock.patch.object(cli_patch.patchset, "resolve_exact") as resolve,
        ):
            exit_code = cmd_patch_gates(_args(source="bigcherry"))

        self.assertEqual(exit_code, 2)
        resolve_source.assert_called_once()
        resolve.assert_not_called()

    def test_focal_promote_expands_dependency_closure_without_recipe_ids(self) -> None:
        import bigcherry.cli.patch as cli_patch

        dependency = SimpleNamespace(patch_id="D", content_hash="d", state="validated")
        focal = SimpleNamespace(patch_id="P1", content_hash="p", state="untested")
        composition = SimpleNamespace(modules=(dependency, focal))
        results = (GateResult(GateId.G5, GateStatus.PASS, "lifecycle", "test"),)
        with (
            mock.patch.object(cli_patch.campaign_config, "load", return_value=self.cfg),
            mock.patch.object(cli_patch.patch_registry, "load_registry", return_value=self.registry),
            mock.patch.object(cli_patch.patchset, "catalog", return_value=[dependency, focal]),
            mock.patch.object(
                cli_patch.patchset, "expand_composition",
                return_value=SimpleNamespace(expanded=("D", "P1")),
            ) as expand,
            mock.patch.object(cli_patch.patchset, "resolve_exact", return_value=composition) as resolve,
            mock.patch.object(cli_patch, "UpstreamRepository", return_value=self.repo),
            mock.patch.object(cli_patch.patch_gates, "evaluate_patch_gates", return_value=results) as evaluate,
        ):
            exit_code = cmd_patch_gates(_args(intent="promote"))

        self.assertEqual(exit_code, 0)
        expand.assert_called_once_with(("P1",), directory=cli_patch.paths.PATCHES)
        self.assertEqual(resolve.call_args.args[0], ("D", "P1"))
        self.assertIsNone(evaluate.call_args.args[0].recipe_patch_ids)

    def test_rebase_source_mode_is_successfully_wired(self) -> None:
        import bigcherry.cli.patch as cli_patch

        self.cfg.sources = {"bigcherry": SimpleNamespace(backend="hip")}
        composition = SimpleNamespace(modules=(self.module,))
        results = (GateResult(GateId.G0, GateStatus.PASS, "composition", "test"),)
        with (
            mock.patch.object(cli_patch.campaign_config, "load", return_value=self.cfg),
            mock.patch.object(cli_patch.patch_registry, "load_registry", return_value=self.registry),
            mock.patch.object(cli_patch.patchset, "catalog", return_value=[self.module]),
            mock.patch(
                "bigcherry.campaign.resolution.resolve_canonical_selection",
                return_value=SimpleNamespace(patch_ids=("P1",)),
            ),
            mock.patch.object(cli_patch.patchset, "resolve_exact", return_value=composition),
            mock.patch.object(cli_patch.patch_rebase, "load_report", return_value={"report": "r"}),
            mock.patch.object(cli_patch, "UpstreamRepository", return_value=self.repo),
            mock.patch.object(cli_patch.patch_gates, "evaluate_patch_gates", return_value=results) as evaluate,
        ):
            exit_code = cmd_patch_gates(_args(
                intent="rebase", source="bigcherry", rebase_report="r.json", all_report="a.json",
            ))
        self.assertEqual(exit_code, 0)
        self.assertEqual(evaluate.call_args.args[0].intent.value, "rebase")
        self.assertEqual(evaluate.call_args.args[0].recipe_patch_ids, frozenset({"P1"}))

    def test_malformed_or_non_object_report_returns_resolution_error(self) -> None:
        import bigcherry.cli.patch as cli_patch

        with (
            mock.patch.object(cli_patch.campaign_config, "load", return_value=self.cfg),
            mock.patch.object(cli_patch.patch_registry, "load_registry", return_value=self.registry),
            mock.patch.object(cli_patch.patchset, "catalog", return_value=[self.module]),
            mock.patch.object(cli_patch.patchset, "expand_composition",
                              return_value=SimpleNamespace(expanded=("P1",))),
            mock.patch.object(cli_patch.patchset, "resolve_exact", return_value=self.composition),
            mock.patch.object(cli_patch, "UpstreamRepository", return_value=self.repo),
            mock.patch.object(cli_patch.patch_rebase, "load_report", return_value=["not", "an", "object"]),
        ):
            exit_code = cmd_patch_gates(_args(rebase_report="bad.json"))
        self.assertEqual(exit_code, 2)

    def test_malformed_report_returns_resolution_error(self) -> None:
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
            mock.patch.object(
                cli_patch.patch_rebase,
                "load_report",
                side_effect=cli_patch.patch_rebase.RebaseCheckError("invalid JSON"),
            ),
        ):
            exit_code = cmd_patch_gates(_args(rebase_report="bad.json"))

        self.assertEqual(exit_code, 2)

    def test_human_output_is_derived_from_json_result_shape(self) -> None:
        import bigcherry.cli.patch as cli_patch

        results = (
            GateResult(GateId.G0, GateStatus.PASS, "composition", "test"),
            GateResult(GateId.G1, GateStatus.NA, "documentation", "test"),
        )
        self.repo.resolve_ref.side_effect = (
            lambda ref: "a" * 40 if ref == self.cfg.pinned else "b" * 40
        )
        with (
            mock.patch.object(cli_patch.campaign_config, "load", return_value=self.cfg),
            mock.patch.object(cli_patch.patch_registry, "load_registry", return_value=self.registry),
            mock.patch.object(cli_patch.patchset, "catalog", return_value=[self.module]),
            mock.patch.object(cli_patch.patchset, "expand_composition",
                              return_value=SimpleNamespace(expanded=("P1",))),
            mock.patch.object(cli_patch.patchset, "resolve_exact", return_value=self.composition),
            mock.patch.object(cli_patch, "UpstreamRepository", return_value=self.repo),
            mock.patch.object(cli_patch.patch_gates, "evaluate_patch_gates", return_value=results),
        ):
            json_output = StringIO()
            with redirect_stdout(json_output):
                self.assertEqual(cmd_patch_gates(_args(json=True)), 0)
            human = StringIO()
            with redirect_stdout(human):
                self.assertEqual(cmd_patch_gates(_args(json=False)), 0)
        payload = json.loads(json_output.getvalue())
        self.assertEqual(
            [(gate["id"], gate["status"]) for gate in payload["gates"]],
            [("G0", "PASS"), ("G1", "NA")],
        )
        self.assertIn("G0: PASS", human.getvalue())
        self.assertIn("G1: NA", human.getvalue())
        self.assertIn("RESULT: PASS", human.getvalue())

    def test_fail_result_is_nonzero_and_preserves_diagnostic(self) -> None:
        import bigcherry.cli.patch as cli_patch

        results = (GateResult(GateId.G7, GateStatus.FAIL, "admission", "patch_admission", ("rejected",)),)
        with (
            mock.patch.object(cli_patch.campaign_config, "load", return_value=self.cfg),
            mock.patch.object(cli_patch.patch_registry, "load_registry", return_value=self.registry),
            mock.patch.object(cli_patch.patchset, "catalog", return_value=[self.module]),
            mock.patch.object(cli_patch.patchset, "expand_composition",
                              return_value=SimpleNamespace(expanded=("P1",))),
            mock.patch.object(cli_patch.patchset, "resolve_exact", return_value=self.composition),
            mock.patch.object(cli_patch, "UpstreamRepository", return_value=self.repo),
            mock.patch.object(cli_patch.patch_gates, "evaluate_patch_gates", return_value=results),
        ):
            output = StringIO()
            with redirect_stdout(output):
                exit_code = cmd_patch_gates(_args(json=False))
        self.assertEqual(exit_code, 1)
        self.assertIn("rejected", output.getvalue())
        self.assertIn("RESULT: FAIL", output.getvalue())

    def test_parser_registers_patch_gates_and_required_intent(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["patch-gates", "P1", "--intent", "validate", "--json"])
        self.assertIs(args.func, cmd_patch_gates)
        self.assertEqual(args.patch_id, "P1")
        self.assertEqual(args.intent, "validate")
        self.assertIn("patch-gates", parser.format_help())

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
