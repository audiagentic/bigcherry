"""Canonical patch catalog and exact-resolution tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import patchset  # noqa: E402


class PatchResolutionTests(unittest.TestCase):
    def test_catalog_has_canonical_hashes_and_numeric_order(self):
        catalog = patchset.catalog()
        self.assertTrue(catalog)
        self.assertEqual(len({item.patch_id for item in catalog}), len(catalog))
        self.assertTrue(all(len(item.content_hash) == 64 for item in catalog))
        self.assertEqual(catalog, sorted(catalog, key=lambda item: (item.order, item.patch_id)))

    def test_exact_resolution_is_deterministic_and_state_checked(self):
        selected = patchset.resolve_exact(
            ["1000_rdna4_mmq_q2k_q6k_fix", "0100_cmake_options"],
            required_state="validated",
        )
        self.assertEqual(
            [item.patch_id for item in selected.modules],
            ["0100_cmake_options", "1000_rdna4_mmq_q2k_q6k_fix"],
        )

    def test_unknown_duplicate_and_untested_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown patch"):
            patchset.resolve_exact(["does-not-exist"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            patchset.resolve_exact(["0100_cmake_options", "0100_cmake_options"])
        with self.assertRaisesRegex(ValueError, "required state"):
            patchset.resolve_exact(["1002_hip_unsafe_math_opt_in"], required_state="validated")

    def test_catalog_ignores_private_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "_private.py").write_text("STATE='rejected'\n", encoding="utf-8")
            (root / "0100_first.py").write_text("PATCH=None\n", encoding="utf-8")
            items = patchset.catalog(root)
            self.assertEqual([item.patch_id for item in items], ["0100_first"])


if __name__ == "__main__":
    unittest.main()
