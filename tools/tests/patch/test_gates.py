"""Contract tests for the shared PA21 gate types."""

from __future__ import annotations

import dataclasses
import sys
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

    def test_admission_gate_passes_full_composition(self) -> None:
        modules = (SimpleNamespace(patch_id="A"), SimpleNamespace(patch_id="B"))
        context = SimpleNamespace(
            composition=SimpleNamespace(modules=modules), catalog_path=None,
            patches_dir=Path("patches"), pinned_ref="b10901",
            resolved_base_revision="abc", evidence_root=None,
            allow_legacy_grandfather=True,
        )
        result = SimpleNamespace(admissible=True, gate_active=True, status="admitted")
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
            admissible=True, gate_active=False, status="not-ready", warnings=("bootstrap",)
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


if __name__ == "__main__":
    unittest.main()
