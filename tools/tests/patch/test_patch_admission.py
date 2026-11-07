"""HI102 admission policy tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import patch_admission  # noqa: E402


class TestPatchAdmission(unittest.TestCase):
    def _catalog(self, root: Path, state: str = "validated") -> tuple[Path, Path]:
        patches = root / "patches"
        patches.mkdir()
        (patches / "0001_test.py").write_text(
            f'GROUP = "core"\nSTATE = "{state}"\nPATCHES = []\n', encoding="utf-8"
        )
        catalog = root / "catalog.toml"
        catalog.write_text(
            'version = 1\n\n[[patch]]\nid = "0001_test"\n'
            'kind = "framework"\norigin = "local"\nbackend = "hip"\n'
            f'state = "{state}"\n', encoding="utf-8"
        )
        return catalog, patches

    def test_bootstrap_gate_is_non_blocking_without_real_eligible_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, patches = self._catalog(root)
            evidence = root / "evidence"
            result = patch_admission.admit(
                "llamacpp", ["0001_test"], catalog_path=catalog, patches_dir=patches,
                pinned_ref="pin", evidence_root=evidence,
            )
            self.assertTrue(result.admissible)
            self.assertFalse(result.gate_active)
            self.assertEqual(result.status, "not-ready")

    def test_build_mode_admits_stale_evidence_with_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, patches = self._catalog(root)
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / "0002_eligible.json").write_text(
                json.dumps({"records": [{"eligible_for_validated_state": True}]}),
                encoding="utf-8",
            )
            result = patch_admission.admit(
                "llamacpp", ["0001_test"], mode="build", catalog_path=catalog,
                patches_dir=patches, pinned_ref="pin", evidence_root=evidence,
            )
            self.assertTrue(result.admissible)
            self.assertEqual(result.failures, ())
            self.assertEqual(result.status, "admitted-with-warnings")
            self.assertTrue(any("not fully validated" in w for w in result.warnings))

    def test_production_gate_rejects_stale_validated_state_after_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, patches = self._catalog(root)
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / "0002_eligible.json").write_text(
                json.dumps({"records": [{"eligible_for_validated_state": True}]}),
                encoding="utf-8",
            )
            result = patch_admission.admit(
                "llamacpp", ["0001_test"], catalog_path=catalog, patches_dir=patches,
                pinned_ref="pin", evidence_root=evidence,
            )
            self.assertFalse(result.admissible)
            self.assertTrue(result.gate_active)
            self.assertIn("0001_test", result.failures[0])

    def test_apply_escape_hatch_is_explicit_and_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, patches = self._catalog(root)
            evidence = root / "evidence"
            with mock.patch.object(patch_admission, "_has_non_grandfathered_eligible", return_value=True):
                result = patch_admission.admit(
                    "llamacpp", ["0001_test"], mode="apply", catalog_path=catalog,
                    patches_dir=patches, pinned_ref="pin", evidence_root=evidence,
                    allow_stale_validation_evidence=True,
                )
            self.assertTrue(result.admissible)
            self.assertEqual(result.status, "escape-hatch")
            self.assertTrue(result.warnings)

    def test_evidence_with_provenance_notes_is_admitted_without_warnings(self):
        from bigcherry.patch import evidence as patch_evidence
        noted = patch_evidence.EvidenceCheck(
            "validated-evidence", notes=("evidence recorded at pin old-pin (now pin)",),
        )
        with mock.patch.object(patch_admission, "_has_non_grandfathered_eligible", return_value=True), \
             mock.patch.object(patch_admission.patch_catalog, "validation_evidence_statuses",
                               return_value={"0001_test": noted}) as statuses:
            result = patch_admission.admit("llamacpp", ["0001_test"], pinned_ref="pin")
        self.assertNotIn("carry_forward", statuses.call_args.kwargs)
        self.assertTrue(result.admissible)
        self.assertEqual(result.status, "admitted")
        self.assertEqual(result.warnings, ())

    def test_ineligible_evidence_is_still_rejected(self):
        from bigcherry.patch import evidence as patch_evidence
        ineligible = patch_evidence.EvidenceCheck("missing-or-stale", ("correctness did not pass",))
        with mock.patch.object(patch_admission, "_has_non_grandfathered_eligible", return_value=True), \
             mock.patch.object(patch_admission.patch_catalog, "validation_evidence_statuses",
                               return_value={"0001_test": ineligible}):
            result = patch_admission.admit("llamacpp", ["0001_test"], pinned_ref="pin")
        self.assertFalse(result.admissible)


class EvidenceGateNoteTests(unittest.TestCase):
    def _g4(self, intent, check):
        from bigcherry.patch import gates
        context = mock.Mock(intent=intent)
        context.descriptor.patch_id = "p"
        with mock.patch.object(gates.validation_policy, "is_framework_configuration_patch", return_value=False), \
             mock.patch.object(gates.patch_catalog, "validation_evidence_statuses",
                               return_value={"p": check}) as statuses:
            result = gates.evaluate_evidence_gate(context)
        self.assertNotIn("carry_forward", statuses.call_args.kwargs)
        return result

    def test_build_and_promote_pass_with_notes_as_detail(self):
        from bigcherry.patch import evidence as patch_evidence, gates
        check = patch_evidence.EvidenceCheck(
            "validated-evidence", notes=("evidence recorded at pin v1.16.0 (now v1.17.0)",),
        )
        for intent in (gates.GateIntent.BUILD, gates.GateIntent.PROMOTE):
            with self.subTest(intent=intent):
                result = self._g4(intent, check)
                self.assertEqual(result.status, gates.GateStatus.PASS)
                self.assertEqual(
                    result.detail,
                    (gates.EVIDENCE_NOTE_PREFIX + "evidence recorded at pin v1.16.0 (now v1.17.0)",),
                )

    def test_carried_forward_is_no_longer_a_recognized_status(self):
        from bigcherry.patch import evidence as patch_evidence, gates
        self.assertFalse(patch_evidence.EvidenceCheck("carried-forward").ok)
        result = self._g4(gates.GateIntent.BUILD, patch_evidence.EvidenceCheck("carried-forward"))
        self.assertEqual(result.status, gates.GateStatus.BLOCKED)

    def test_live_revision_uses_shared_identity_primitive(self):
        with mock.patch.object(patch_admission.source_identity, "git_revision", return_value="abc"):
            self.assertEqual(patch_admission.live_revision(Path("/source")), "abc")
            with self.assertRaises(ValueError):
                patch_admission.verify_live_revision(Path("/source"), "def")


if __name__ == "__main__":
    unittest.main()
