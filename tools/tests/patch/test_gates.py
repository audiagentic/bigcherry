"""Contract tests for the shared PA21 gate types."""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import gates  # noqa: E402
from bigcherry.patch.gates import GateId, GateIntent, GateStatus, gate_applies  # noqa: E402


class GateContractTests(unittest.TestCase):
    def test_public_enums_are_string_valued(self) -> None:
        self.assertEqual(GateId.G0.value, "G0")
        self.assertEqual(GateStatus.BLOCKED.value, "BLOCKED")
        self.assertEqual(GateIntent.PROMOTE.value, "promote")

    def test_applicability_matrix_is_explicit(self) -> None:
        self.assertTrue(gate_applies(GateId.G0, GateIntent.AUTHOR))
        self.assertTrue(gate_applies(GateId.G4, GateIntent.PROMOTE))
        self.assertFalse(gate_applies(GateId.G2, GateIntent.AUTHOR))
        self.assertTrue(gate_applies(GateId.G7, GateIntent.BUILD))
        self.assertFalse(gate_applies(GateId.G7, GateIntent.REBASE))

    def test_lint_applicability_is_exactly_summary_and_package(self) -> None:
        self.assertEqual(
            tuple(gate_id for gate_id in GateId if gate_applies(gate_id, GateIntent.LINT)),
            (GateId.G1, GateId.G3),
        )

    def test_gate_result_dataclass_is_frozen(self) -> None:
        from bigcherry.patch.gates import GateResult

        result = GateResult(GateId.G0, GateStatus.PASS, "composition", "registry")
        self.assertTrue(dataclasses.is_dataclass(result))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.status = GateStatus.FAIL  # type: ignore[misc]

    def test_composition_gate_requires_exact_ordered_identity(self) -> None:
        modules = (
            SimpleNamespace(patch_id="A", content_hash="a"),
            SimpleNamespace(patch_id="B", content_hash="b"),
        )
        context = SimpleNamespace(
            composition=SimpleNamespace(modules=modules), patches_dir=Path("patches")
        )
        with mock.patch.object(
            gates.patchset, "resolve_exact",
            return_value=SimpleNamespace(modules=modules),
        ):
            result = gates.evaluate_composition_gate(context)
        self.assertEqual(result.status, GateStatus.PASS)

    def test_evidence_gate_passes_and_marks_promotion_prospectively(self) -> None:
        context = SimpleNamespace(
            descriptor=SimpleNamespace(patch_id="P1"),
            catalog_path=None,
            patches_dir=Path("patches"),
            pinned_ref="b10901",
            evidence_root=None,
            allow_legacy_grandfather=True,
            resolved_base_revision="abc",
            intent=GateIntent.PROMOTE,
        )
        evidence = SimpleNamespace(status="validated-evidence", ok=True, problems=())
        with mock.patch.object(
            gates.patch_catalog, "validation_evidence_statuses", return_value={"P1": evidence}
        ) as verifier:
            result = gates.evaluate_evidence_gate(context)
        self.assertEqual(result.status, GateStatus.PASS)
        self.assertEqual(verifier.call_args.kwargs["assume_validated"], frozenset({"P1"}))

    def test_evidence_gate_does_not_treat_required_validation_as_not_required(self) -> None:
        context = SimpleNamespace(
            descriptor=SimpleNamespace(patch_id="P1"),
            catalog_path=None,
            patches_dir=Path("patches"),
            pinned_ref="b10901",
            evidence_root=None,
            allow_legacy_grandfather=True,
            resolved_base_revision=None,
            intent=GateIntent.VALIDATE,
        )
        evidence = SimpleNamespace(status="not-required", ok=True, problems=())
        with mock.patch.object(
            gates.patch_catalog, "validation_evidence_statuses", return_value={"P1": evidence}
        ):
            result = gates.evaluate_evidence_gate(context)
        self.assertEqual(result.status, GateStatus.FAIL)

    def test_evidence_gate_blocks_unknown_or_inconsistent_status(self) -> None:
        context = SimpleNamespace(
            descriptor=SimpleNamespace(patch_id="P1"),
            catalog_path=None, patches_dir=Path("patches"), pinned_ref="b10901",
            evidence_root=None, allow_legacy_grandfather=True,
            resolved_base_revision=None, intent=GateIntent.VALIDATE,
        )
        evidence = SimpleNamespace(status="future-status", ok=True, problems=())
        with mock.patch.object(
            gates.patch_catalog, "validation_evidence_statuses", return_value={"P1": evidence}
        ):
            result = gates.evaluate_evidence_gate(context)
        self.assertEqual(result.status, GateStatus.BLOCKED)

    def test_framework_promotion_blocks_before_evidence_resolution(self) -> None:
        descriptor = SimpleNamespace(
            patch_id="P1", representation="packaged", kind="framework",
            origin="local", external_source=None, experiment_contracts=(),
            plan_ids=(), plan_item=None,
        )
        context = SimpleNamespace(
            descriptor=descriptor, intent=GateIntent.PROMOTE,
        )
        with mock.patch.object(gates.patch_catalog, "validation_evidence_statuses") as verifier:
            result = gates.evaluate_evidence_gate(context)
        self.assertEqual(result.status, GateStatus.BLOCKED)
        verifier.assert_not_called()

    def test_repository_lint_adapter_preserves_static_policy_and_avoids_operational_gates(self) -> None:
        descriptors = tuple(
            SimpleNamespace(patch_id=patch_id)
            for patch_id in ("P1", "P2", "P3", "P4", "LEGACY")
        )
        registry = SimpleNamespace(descriptors=descriptors, root=Path("patches"))
        package_report = gates.validation_policy.PackagePolicyReport(
            statuses=(
                gates.validation_policy.PackagePolicyStatus("P1", "current"),
                gates.validation_policy.PackagePolicyStatus("P2", "invalid", ("P2: invalid",)),
                gates.validation_policy.PackagePolicyStatus("P3", "not-required"),
                gates.validation_policy.PackagePolicyStatus("P4", "grandfathered", ("P4: shape",)),
            ),
            problems=("P2: invalid",),
            grandfathered=("P4",),
        )
        performance = {
            "P1": ("P1: missing native baseline",),
            "P2": (),
            "P3": ("P3: missing native baseline",),
            "P4": (),
            "LEGACY": (),
        }

        with (
            mock.patch.object(gates.patch_registry, "load_registry", return_value=registry) as load_registry,
            mock.patch.object(gates, "gate_applies", wraps=gates.gate_applies) as gate_applies,
            mock.patch.object(
                gates.patch_docs,
                "check_summary_for_patch",
                side_effect=lambda descriptor, root: (
                    (f"{descriptor.patch_id}: summary mismatch",)
                    if descriptor.patch_id == "P1" else ()
                ),
            ),
            mock.patch.object(
                gates.validation_policy,
                "check_validation_packages",
                return_value=package_report,
            ) as check_packages,
            mock.patch.object(
                gates.validation_policy,
                "check_performance_evidence_for_patch",
                side_effect=lambda descriptor, **kwargs: performance[descriptor.patch_id],
            ) as check_performance,
            mock.patch.object(gates.validation_policy, "require_execution_package") as require_package,
            mock.patch.object(gates.patch_catalog, "validation_evidence_statuses") as evidence,
            mock.patch.object(gates.rebase, "require_fresh_report") as rebase,
            mock.patch.object(gates.patch_disposition, "compute_coverage") as coverage,
            mock.patch.object(gates.patch_disposition, "list_dispositions") as dispositions,
            mock.patch("bigcherry.patch_admission.admit") as admission,
        ):
            report = gates.evaluate_repository_lint_gates(
                patches_dir=Path("patches"),
                external_sources_path=Path("external.toml"),
                validation_baseline_path=Path("baseline.json"),
            )

        load_registry.assert_called_once_with(Path("patches"))
        gate_applies.assert_any_call(GateId.G1, GateIntent.LINT)
        gate_applies.assert_any_call(GateId.G3, GateIntent.LINT)
        check_packages.assert_called_once_with(
            root=Path("patches"), registry_path=Path("patches"),
            external_sources_path=Path("external.toml"), baseline_path=Path("baseline.json"),
        )
        self.assertEqual(check_performance.call_count, len(descriptors))
        self.assertTrue(all(call.kwargs["assume_validated"] is False
                            for call in check_performance.call_args_list))
        self.assertEqual(
            report.problems,
            (
                "P1: summary mismatch", "P2: invalid",
                "P1: missing native baseline", "P3: missing native baseline",
            ),
        )
        self.assertEqual(report.grandfathered, ("P4",))
        outcomes = {
            (item.patch_id, item.result.id): item.result.status
            for item in report.results
        }
        self.assertEqual(outcomes[("P1", GateId.G1)], GateStatus.FAIL)
        self.assertEqual(outcomes[("P1", GateId.G3)], GateStatus.FAIL)
        self.assertEqual(outcomes[("P2", GateId.G3)], GateStatus.FAIL)
        self.assertEqual(outcomes[("P3", GateId.G3)], GateStatus.FAIL)
        self.assertEqual(outcomes[("P4", GateId.G3)], GateStatus.PASS)
        self.assertEqual(outcomes[("LEGACY", GateId.G3)], GateStatus.NA)
        require_package.assert_not_called()
        evidence.assert_not_called()
        rebase.assert_not_called()
        coverage.assert_not_called()
        dispositions.assert_not_called()
        admission.assert_not_called()

    def test_repository_lint_unknown_package_status_is_fail_closed(self) -> None:
        descriptor = SimpleNamespace(patch_id="P1")
        registry = SimpleNamespace(descriptors=(descriptor,), root=Path("patches"))
        package_report = gates.validation_policy.PackagePolicyReport(
            statuses=(gates.validation_policy.PackagePolicyStatus("P1", "future"),),
        )
        with (
            mock.patch.object(gates.patch_registry, "load_registry", return_value=registry),
            mock.patch.object(gates.patch_docs, "check_summary_for_patch", return_value=()),
            mock.patch.object(
                gates.validation_policy,
                "check_validation_packages",
                return_value=package_report,
            ),
            mock.patch.object(
                gates.validation_policy,
                "check_performance_evidence_for_patch",
                return_value=(),
            ),
        ):
            report = gates.evaluate_repository_lint_gates()

        outcome = next(
            item.result for item in report.results
            if item.patch_id == "P1" and item.result.id is GateId.G3
        )
        self.assertEqual(outcome.status, GateStatus.BLOCKED)
        self.assertEqual(report.problems, outcome.detail)

    def test_admission_gate_passes_full_composition(self) -> None:
        modules = (SimpleNamespace(patch_id="A"), SimpleNamespace(patch_id="B"))
        context = SimpleNamespace(
            composition=SimpleNamespace(modules=modules), catalog_path=None,
            patches_dir=Path("patches"), pinned_ref="b10901",
            resolved_base_revision="abc", evidence_root=None,
            allow_legacy_grandfather=True,
        )
        result = SimpleNamespace(
            admissible=True, gate_active=True, status="admitted",
            failures=(), warnings=(),
        )
        with mock.patch("bigcherry.patch_admission.admit", return_value=result) as admit:
            outcome = gates.evaluate_admission_gate(context)
        self.assertEqual(outcome.status, GateStatus.PASS)
        self.assertEqual(admit.call_args.args[0], ("A", "B"))

    def test_admission_gate_blocks_inactive_not_ready_result(self) -> None:
        context = SimpleNamespace(
            composition=SimpleNamespace(modules=()), catalog_path=None,
            patches_dir=Path("patches"), pinned_ref="b10901",
            resolved_base_revision=None, evidence_root=None,
            allow_legacy_grandfather=True,
        )
        result = SimpleNamespace(
            admissible=True, gate_active=False, status="not-ready",
            failures=(), warnings=("bootstrap",)
        )
        with mock.patch("bigcherry.patch_admission.admit", return_value=result):
            outcome = gates.evaluate_admission_gate(context)
        self.assertEqual(outcome.status, GateStatus.BLOCKED)

    def test_admission_gate_blocks_inactive_rejected_result(self) -> None:
        context = SimpleNamespace(
            composition=SimpleNamespace(modules=()), catalog_path=None,
            patches_dir=Path("patches"), pinned_ref="b10901",
            resolved_base_revision=None, evidence_root=None,
            allow_legacy_grandfather=True,
        )
        result = SimpleNamespace(
            admissible=False, gate_active=False, status="rejected",
            failures=("bootstrap",), warnings=(),
        )
        with mock.patch("bigcherry.patch_admission.admit", return_value=result):
            outcome = gates.evaluate_admission_gate(context)
        self.assertEqual(outcome.status, GateStatus.BLOCKED)

    def test_admission_gate_maps_active_rejection_to_fail(self) -> None:
        context = SimpleNamespace(
            composition=SimpleNamespace(modules=(SimpleNamespace(patch_id="P1"),)),
            catalog_path=None, patches_dir=Path("patches"), pinned_ref="b10901",
            resolved_base_revision="abc", evidence_root=None,
            allow_legacy_grandfather=True,
        )
        result = SimpleNamespace(
            admissible=False, gate_active=True, status="rejected",
            failures=("policy rejected composition",), warnings=(),
        )
        with mock.patch("bigcherry.patch_admission.admit", return_value=result) as admit:
            outcome = gates.evaluate_admission_gate(context)
        self.assertEqual(outcome.status, GateStatus.FAIL)
        self.assertEqual(outcome.detail, ("policy rejected composition",))
        self.assertEqual(admit.call_args.args[0], ("P1",))

    def test_admission_gate_cross_checks_real_rejected_composition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patches = root / "patches"
            patches.mkdir()
            (patches / "0001_test.py").write_text(
                'GROUP = "core"\nSTATE = "validated"\nPATCHES = []\n',
                encoding="utf-8",
            )
            catalog = root / "catalog.toml"
            catalog.write_text(
                'version = 1\n\n[[patch]]\nid = "0001_test"\n'
                'kind = "framework"\norigin = "local"\nbackend = "hip"\n'
                'state = "validated"\n',
                encoding="utf-8",
            )
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / "0002_eligible.json").write_text(
                json.dumps({"records": [{"eligible_for_validated_state": True}]}),
                encoding="utf-8",
            )
            context = SimpleNamespace(
                composition=SimpleNamespace(
                    modules=(SimpleNamespace(patch_id="0001_test"),),
                ),
                catalog_path=catalog,
                patches_dir=patches,
                pinned_ref="pin",
                resolved_base_revision=None,
                evidence_root=evidence,
                allow_legacy_grandfather=True,
            )

            outcome = gates.evaluate_admission_gate(context)

        self.assertEqual(outcome.id, GateId.G7)
        self.assertEqual(outcome.status, GateStatus.FAIL)
        self.assertIn("0001_test", outcome.detail[0])

    def test_admission_gate_blocks_malformed_failure_fields(self) -> None:
        context = SimpleNamespace(
            composition=SimpleNamespace(modules=()), catalog_path=None,
            patches_dir=Path("patches"), pinned_ref="b10901",
            resolved_base_revision=None, evidence_root=None,
            allow_legacy_grandfather=True,
        )
        result = SimpleNamespace(
            admissible=True, gate_active=True, status="admitted",
            failures="not-a-sequence", warnings=(),
        )
        with mock.patch("bigcherry.patch_admission.admit", return_value=result):
            outcome = gates.evaluate_admission_gate(context)
        self.assertEqual(outcome.status, GateStatus.BLOCKED)

    def test_admission_gate_blocks_production_escape_hatch(self) -> None:
        context = SimpleNamespace(
            composition=SimpleNamespace(modules=()), catalog_path=None,
            patches_dir=Path("patches"), pinned_ref="b10901",
            resolved_base_revision=None, evidence_root=None,
            allow_legacy_grandfather=True,
        )
        result = SimpleNamespace(
            admissible=False, gate_active=True, status="escape-hatch",
            failures=("override",), warnings=(),
        )
        with mock.patch("bigcherry.patch_admission.admit", return_value=result):
            outcome = gates.evaluate_admission_gate(context)
        self.assertEqual(outcome.status, GateStatus.BLOCKED)

    def test_patch_gate_evaluator_preserves_order_and_applicability(self) -> None:
        context = SimpleNamespace(intent=GateIntent.AUTHOR)
        g0 = gates.GateResult(GateId.G0, GateStatus.PASS, "composition", "test")
        g1 = gates.GateResult(GateId.G1, GateStatus.PASS, "documentation", "test")
        with mock.patch.object(gates, "evaluate_composition_gate", return_value=g0), \
             mock.patch.object(gates, "evaluate_summary_gate", return_value=g1):
            results = gates.evaluate_patch_gates(context)
        self.assertEqual(tuple(result.id for result in results), (GateId.G0, GateId.G1))

    def test_patch_gate_evaluator_preserves_promote_order(self) -> None:
        context = SimpleNamespace(intent=GateIntent.PROMOTE)
        values = {
            GateId.G0: gates.GateResult(GateId.G0, GateStatus.PASS, "", "test"),
            GateId.G1: gates.GateResult(GateId.G1, GateStatus.PASS, "", "test"),
            GateId.G2: gates.GateResult(GateId.G2, GateStatus.PASS, "", "test"),
            GateId.G3: gates.GateResult(GateId.G3, GateStatus.PASS, "", "test"),
            GateId.G4: gates.GateResult(GateId.G4, GateStatus.PASS, "", "test"),
            GateId.G5: gates.GateResult(GateId.G5, GateStatus.PASS, "", "test"),
        }
        with mock.patch.object(gates, "evaluate_composition_gate", return_value=values[GateId.G0]), \
             mock.patch.object(gates, "evaluate_summary_gate", return_value=values[GateId.G1]), \
             mock.patch.object(gates, "evaluate_rebase_gate", return_value=values[GateId.G2]), \
             mock.patch.object(gates, "evaluate_package_gate", return_value=values[GateId.G3]), \
             mock.patch.object(gates, "evaluate_evidence_gate", return_value=values[GateId.G4]), \
             mock.patch.object(gates, "evaluate_lifecycle_gate", return_value=values[GateId.G5]):
            results = gates.evaluate_patch_gates(context)
        self.assertEqual(tuple(result.id for result in results),
                         (GateId.G0, GateId.G1, GateId.G2, GateId.G3, GateId.G4, GateId.G5))

    def test_patch_gate_evaluator_preserves_build_order(self) -> None:
        context = SimpleNamespace(intent=GateIntent.BUILD)
        values = {
            gate_id: gates.GateResult(gate_id, GateStatus.PASS, "", "test")
            for gate_id in (GateId.G0, GateId.G1, GateId.G2, GateId.G4, GateId.G6, GateId.G7)
        }
        with mock.patch.object(gates, "evaluate_composition_gate", return_value=values[GateId.G0]), \
             mock.patch.object(gates, "evaluate_summary_gate", return_value=values[GateId.G1]), \
             mock.patch.object(gates, "evaluate_rebase_gate", return_value=values[GateId.G2]), \
             mock.patch.object(gates, "evaluate_evidence_gate", return_value=values[GateId.G4]), \
             mock.patch.object(gates, "evaluate_disposition_gate", return_value=values[GateId.G6]), \
             mock.patch.object(gates, "evaluate_admission_gate", return_value=values[GateId.G7]):
            results = gates.evaluate_patch_gates(context)
        self.assertEqual(tuple(result.id for result in results),
                         (GateId.G0, GateId.G1, GateId.G2, GateId.G4, GateId.G6, GateId.G7))

    def test_lifecycle_gate_blocks_without_all_prerequisites(self) -> None:
        context = SimpleNamespace(
            intent=GateIntent.PROMOTE,
            descriptor=SimpleNamespace(patch_id="P1"),
            composition=SimpleNamespace(modules=(SimpleNamespace(patch_id="P1", state="untested"),)),
        )
        result = gates.evaluate_lifecycle_gate(context, {})
        self.assertEqual(result.status, GateStatus.BLOCKED)
        self.assertIn("G0", result.detail[0])

    def test_lifecycle_gate_passes_untested_patch_after_g0_to_g4(self) -> None:
        context = SimpleNamespace(
            intent=GateIntent.PROMOTE,
            descriptor=SimpleNamespace(patch_id="P1"),
            composition=SimpleNamespace(modules=(SimpleNamespace(patch_id="P1", state="untested"),)),
        )
        prior = {
            gate_id: gates.GateResult(gate_id, GateStatus.PASS, "test", "test")
            for gate_id in (GateId.G0, GateId.G1, GateId.G2, GateId.G3, GateId.G4)
        }
        result = gates.evaluate_lifecycle_gate(context, prior)
        self.assertEqual(result.status, GateStatus.PASS)

    def test_lifecycle_gate_propagates_blocked_prerequisite(self) -> None:
        context = SimpleNamespace(
            intent=GateIntent.PROMOTE,
            descriptor=SimpleNamespace(patch_id="P1"),
            composition=SimpleNamespace(modules=(SimpleNamespace(patch_id="P1", state="untested"),)),
        )
        prior = {
            GateId.G0: gates.GateResult(GateId.G0, GateStatus.BLOCKED, "", "test"),
            GateId.G1: gates.GateResult(GateId.G1, GateStatus.PASS, "", "test"),
            GateId.G2: gates.GateResult(GateId.G2, GateStatus.PASS, "", "test"),
            GateId.G3: gates.GateResult(GateId.G3, GateStatus.PASS, "", "test"),
            GateId.G4: gates.GateResult(GateId.G4, GateStatus.PASS, "", "test"),
        }
        result = gates.evaluate_lifecycle_gate(context, prior)
        self.assertEqual(result.status, GateStatus.BLOCKED)

    def test_disposition_gate_blocks_without_complete_inputs(self) -> None:
        context = SimpleNamespace(
            catalog_states=None, coverage_report=None, target_revision=None,
        )
        result = gates.evaluate_disposition_gate(context)
        self.assertEqual(result.status, GateStatus.BLOCKED)

    def test_disposition_gate_maps_authority_coverage(self) -> None:
        base = {
            "catalog_states": {"P1": "untested"},
            "coverage_report": {
                "selection": {"all_patches": True}, "upstream_revision": "abc",
                "patches": [{"patch_id": "P1", "status": "clean"}],
            },
            "recipe_patch_ids": frozenset(), "target_revision": "abc",
            "dispositions_dir": Path("dispositions"),
            "source_root": Path("llama.cpp"),
        }
        complete = SimpleNamespace(complete=True, uncovered_patch_ids=())
        with mock.patch.object(gates.rebase, "require_fresh_report", return_value=("P1",)), \
             mock.patch.object(
                 gates.patch_disposition, "compute_coverage", return_value=complete
             ):
            result = gates.evaluate_disposition_gate(SimpleNamespace(**base))
        self.assertEqual(result.status, GateStatus.PASS)

        incomplete = SimpleNamespace(complete=False, uncovered_patch_ids=("P1",))
        with mock.patch.object(gates.rebase, "require_fresh_report", return_value=("P1",)), \
             mock.patch.object(
                 gates.patch_disposition, "compute_coverage", return_value=incomplete
             ):
            result = gates.evaluate_disposition_gate(SimpleNamespace(**base))
        self.assertEqual(result.status, GateStatus.FAIL)
        self.assertEqual(result.detail, ("P1",))

    def test_disposition_gate_blocks_malformed_authority_result(self) -> None:
        base = {
            "catalog_states": {"P1": "untested"},
            "coverage_report": {
                "selection": {"all_patches": True}, "upstream_revision": "abc",
            },
            "recipe_patch_ids": frozenset(), "target_revision": "abc",
            "dispositions_dir": Path("dispositions"),
            "source_root": Path("llama.cpp"),
        }
        malformed = SimpleNamespace(complete=True, uncovered_patch_ids="P1")
        with mock.patch.object(gates.rebase, "require_fresh_report", return_value=("P1",)), \
             mock.patch.object(gates.patch_disposition, "compute_coverage", return_value=malformed):
            result = gates.evaluate_disposition_gate(SimpleNamespace(**base))
        self.assertEqual(result.status, GateStatus.BLOCKED)

    def test_disposition_gate_blocks_when_recipe_ids_are_omitted(self) -> None:
        context = SimpleNamespace(
            catalog_states={"P1": "untested"},
            coverage_report={
                "selection": {"all_patches": True}, "upstream_revision": "abc",
            },
            recipe_patch_ids=None, target_revision="abc",
            dispositions_dir=Path("dispositions"), source_root=Path("llama.cpp"),
        )
        with mock.patch.object(gates.patch_disposition, "compute_coverage") as compute:
            result = gates.evaluate_disposition_gate(context)
        self.assertEqual(result.status, GateStatus.BLOCKED)
        compute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
