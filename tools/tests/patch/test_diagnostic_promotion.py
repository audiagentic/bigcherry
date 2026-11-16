"""PRQ05 diagnostic promotion: offline evidence and G5 lifecycle admission."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bigcherry.patch import gates, profile_evidence, registry

_PATCH_ID = "rad0004_moe_remote_diagnostics"


def _record(*, activation: str = "own counter incremented"):
    return profile_evidence.ProfileEvidence(
        _PATCH_ID, "pin", "digest", False, (),
        {"mechanics": "patch tests and lint pass", "activation": activation},
        profile_evidence.DIAGNOSTIC_TIER,
    )


class DiagnosticPromotionTests(unittest.TestCase):
    def test_diagnostic_is_a_state_not_a_kind_and_has_no_production_scope(self):
        self.assertIn("diagnostic", registry.STATES)
        self.assertIn("diagnostic", registry.PATCH_KINDS)
        self.assertEqual(registry.where_it_builds("diagnostic"), "diagnostic, experiments")
        self.assertEqual(registry.where_it_builds("validated"), "production, diagnostic, experiments")
        self.assertEqual(registry.where_it_builds("superseded"), "nowhere (retired)")

    def test_diagnostic_record_has_exactly_two_checks_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            profile_evidence.write(package, _record())
            loaded = profile_evidence.load(package)
            self.assertEqual(loaded, _record())
            self.assertEqual(loaded.document()["tier"], "diagnostic")
            self.assertEqual(tuple(loaded.document()["checks"]), ("activation", "mechanics"))
            self.assertEqual(profile_evidence.verify(
                package, patch_id=_PATCH_ID, pinned_ref="pin",
                subject_digest="digest", tier="diagnostic",
            ).problems, ())
            self.assertTrue(profile_evidence.verify(
                package, patch_id=_PATCH_ID, pinned_ref="pin",
                subject_digest="digest",
            ).problems)
            self.assertTrue(profile_evidence.problems(_record(activation="")))
            self.assertTrue(profile_evidence.problems(
                profile_evidence.ProfileEvidence(_PATCH_ID, "pin", "digest", False, (),
                    {"mechanics": "pass", "activation": "seen", "ab": "forbidden"}, "diagnostic")
            ))

    def test_g4_accepts_marker_and_refuses_missing_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / _PATCH_ID
            package.mkdir()
            descriptor = SimpleNamespace(
                patch_id=_PATCH_ID, package_root=Path(_PATCH_ID),
                implementation_path=Path(_PATCH_ID) / "patch.py",
            )
            context = SimpleNamespace(
                intent=gates.GateIntent.PROMOTE,
                promotion_state="diagnostic", descriptor=descriptor,
                patches_dir=root, pinned_ref="pin",
            )
            with mock.patch.object(gates.patch_evidence, "patch_validation_subject_digest", return_value="digest"):
                self.assertEqual(gates.evaluate_evidence_gate(context).status, gates.GateStatus.FAIL)
                profile_evidence.write(package, _record())
                good = gates.evaluate_evidence_gate(context)
                self.assertEqual(good.status, gates.GateStatus.PASS)
                self.assertIn(gates.DIAGNOSTIC_EVIDENCE_DETAIL, good.detail)
                with self.assertRaisesRegex(ValueError, "activation"):
                    profile_evidence.write(package, _record(activation=""))
                record_path = profile_evidence.record_path(package)
                malformed = json.loads(record_path.read_text())
                malformed["checks"]["activation"] = ""
                record_path.write_text(json.dumps(malformed))
                self.assertEqual(gates.evaluate_evidence_gate(context).status, gates.GateStatus.FAIL)

    def test_g5_requires_diagnostic_tier_and_upstream_gates(self):
        context = SimpleNamespace(
            intent=gates.GateIntent.PROMOTE, promotion_state="diagnostic",
            descriptor=SimpleNamespace(patch_id=_PATCH_ID),
            composition=SimpleNamespace(modules=(SimpleNamespace(patch_id=_PATCH_ID, state="untested"),)),
        )
        def result(key, status, detail=()):
            return gates.GateResult(key, status, "test", "test", detail)
        prior = {gate_id: result(gate_id, gates.GateStatus.PASS) for gate_id in (
            gates.GateId.G0, gates.GateId.G1, gates.GateId.G2, gates.GateId.G4,
        )}
        prior[gates.GateId.G3] = result(gates.GateId.G3, gates.GateStatus.NA)
        prior[gates.GateId.G4] = result(
            gates.GateId.G4, gates.GateStatus.PASS, (gates.DIAGNOSTIC_EVIDENCE_DETAIL,),
        )
        self.assertEqual(gates.evaluate_lifecycle_gate(context, prior).status, gates.GateStatus.PASS)
        prior[gates.GateId.G4] = result(gates.GateId.G4, gates.GateStatus.PASS)
        self.assertEqual(gates.evaluate_lifecycle_gate(context, prior).status, gates.GateStatus.FAIL)
        prior[gates.GateId.G4] = result(
            gates.GateId.G4, gates.GateStatus.PASS, (gates.DIAGNOSTIC_EVIDENCE_DETAIL,),
        )
        prior[gates.GateId.G2] = result(gates.GateId.G2, gates.GateStatus.BLOCKED)
        self.assertEqual(gates.evaluate_lifecycle_gate(context, prior).status, gates.GateStatus.BLOCKED)


if __name__ == "__main__":
    unittest.main()
