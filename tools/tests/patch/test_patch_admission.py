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

    def test_live_revision_uses_shared_identity_primitive(self):
        with mock.patch.object(patch_admission.source_identity, "git_revision", return_value="abc"):
            self.assertEqual(patch_admission.live_revision(Path("/source")), "abc")
            with self.assertRaises(ValueError):
                patch_admission.verify_live_revision(Path("/source"), "def")


if __name__ == "__main__":
    unittest.main()
