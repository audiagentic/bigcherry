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
                ["0001_test"], catalog_path=catalog, patches_dir=patches,
                pinned_ref="pin", evidence_root=evidence,
            )
            self.assertTrue(result.admissible)
            self.assertFalse(result.gate_active)
            self.assertEqual(result.status, "not-ready")

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
                ["0001_test"], catalog_path=catalog, patches_dir=patches,
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
                    ["0001_test"], mode="apply", catalog_path=catalog,
                    patches_dir=patches, pinned_ref="pin", evidence_root=evidence,
                    allow_stale_validation_evidence=True,
                )
            self.assertTrue(result.admissible)
            self.assertEqual(result.status, "escape-hatch")
            self.assertTrue(result.warnings)

    def test_carried_forward_status_is_admitted_with_warning(self):
        from bigcherry.patch import evidence as patch_evidence
        carried = patch_evidence.EvidenceCheck(
            "carried-forward", ("qualified at old-pin; not revalidated at pin -- revalidate on request",),
        )
        with mock.patch.object(patch_admission, "_has_non_grandfathered_eligible", return_value=True),              mock.patch.object(patch_admission.patch_catalog, "validation_evidence_statuses",
                               return_value={"0001_test": carried}) as statuses:
            result = patch_admission.admit(["0001_test"], pinned_ref="pin")
        self.assertTrue(statuses.call_args.kwargs["carry_forward"])
        self.assertTrue(result.admissible)
        self.assertEqual(result.status, "admitted")
        self.assertIn("carried-forward", result.warnings[0])

    def test_non_pin_evidence_failure_is_still_rejected(self):
        from bigcherry.patch import evidence as patch_evidence
        stale = patch_evidence.EvidenceCheck("missing-or-stale", ("implementation digest mismatch",))
        with mock.patch.object(patch_admission, "_has_non_grandfathered_eligible", return_value=True),              mock.patch.object(patch_admission.patch_catalog, "validation_evidence_statuses",
                               return_value={"0001_test": stale}):
            result = patch_admission.admit(["0001_test"], pinned_ref="pin")
        self.assertFalse(result.admissible)


class CarryForwardAuthorityTests(unittest.TestCase):
    def _records(self, *pins):
        return tuple({"base_ref": ref, "base_revision": rev} for ref, rev in pins)

    def test_qualifying_at_an_earlier_pin_carries_forward(self):
        from bigcherry.patch import catalog, evidence as patch_evidence
        failing = patch_evidence.EvidenceCheck("missing-or-stale", ("stale base_ref",))
        seen = []

        def evaluate(pid, ref, rev):
            seen.append(ref)
            return patch_evidence.EvidenceCheck("validated-evidence" if ref == "old" else "missing-or-stale")

        with mock.patch.object(patch_evidence, "load_records", return_value=self._records(("old", "a" * 40))):
            check = catalog._carried_forward("p", failing, "new", None, evaluate)
        self.assertEqual(check.status, "carried-forward")
        self.assertIn("old", check.problems[0])
        self.assertEqual(seen, ["old"])

    def test_failure_at_every_earlier_pin_is_not_carried_forward(self):
        from bigcherry.patch import catalog, evidence as patch_evidence
        failing = patch_evidence.EvidenceCheck("missing-or-stale", ("implementation digest mismatch",))
        with mock.patch.object(patch_evidence, "load_records", return_value=self._records(("old", "a" * 40))):
            check = catalog._carried_forward(
                "p", failing, "new", None,
                lambda pid, ref, rev: patch_evidence.EvidenceCheck("missing-or-stale"),
            )
        self.assertIs(check, failing)

    def test_live_revision_uses_shared_identity_primitive(self):
        with mock.patch.object(patch_admission.source_identity, "git_revision", return_value="abc"):
            self.assertEqual(patch_admission.live_revision(Path("/source")), "abc")
            with self.assertRaises(ValueError):
                patch_admission.verify_live_revision(Path("/source"), "def")


if __name__ == "__main__":
    unittest.main()
