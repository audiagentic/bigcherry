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

    def _stale_eligible_record(self, evidence: Path, patches: Path, *, same_digest: bool) -> None:
        from bigcherry.patch import patchset
        digest = {m.patch_id: m.content_hash for m in patchset.catalog(patches)}["0001_test"]
        evidence.mkdir()
        (evidence / "0001_test.json").write_text(json.dumps({
            "schema_version": 5, "patch_id": "0001_test",
            "records": [{
                "eligible_for_validated_state": True, "base_ref": "old-pin",
                "patch_implementation_digest": digest if same_digest else "0" * 64,
            }],
        }), encoding="utf-8")

    def test_stale_evidence_for_same_implementation_is_carried_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, patches = self._catalog(root)
            evidence = root / "evidence"
            self._stale_eligible_record(evidence, patches, same_digest=True)
            result = patch_admission.admit(
                ["0001_test"], catalog_path=catalog, patches_dir=patches,
                pinned_ref="pin", evidence_root=evidence,
            )
            self.assertTrue(result.admissible)
            self.assertEqual(result.status, "admitted")
            self.assertIn("carried-forward", result.warnings[0])
            self.assertIn("old-pin", result.warnings[0])

    def test_changed_implementation_is_not_carried_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, patches = self._catalog(root)
            evidence = root / "evidence"
            self._stale_eligible_record(evidence, patches, same_digest=False)
            result = patch_admission.admit(
                ["0001_test"], catalog_path=catalog, patches_dir=patches,
                pinned_ref="pin", evidence_root=evidence,
            )
            self.assertFalse(result.admissible)
            self.assertIn("0001_test", result.failures[0])

    def test_live_revision_uses_shared_identity_primitive(self):
        with mock.patch.object(patch_admission.source_identity, "git_revision", return_value="abc"):
            self.assertEqual(patch_admission.live_revision(Path("/source")), "abc")
            with self.assertRaises(ValueError):
                patch_admission.verify_live_revision(Path("/source"), "def")


if __name__ == "__main__":
    unittest.main()
