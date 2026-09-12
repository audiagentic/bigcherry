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


if __name__ == "__main__":
    unittest.main()
