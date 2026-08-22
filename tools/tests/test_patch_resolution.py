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


class ExpandCompositionTests(unittest.TestCase):
    """RE42: computing a REQUIRES closure above resolve_exact()'s fail-closed
    exact layer -- new, opt-in, does not change resolve_lane's behavior."""

    def test_1217s_real_closure_pulls_in_1215_and_1216(self):
        result = patchset.expand_composition(["1217_rd44_graph_opt_default_rdna35"])
        self.assertEqual(result.requested, ("1217_rd44_graph_opt_default_rdna35",))
        self.assertEqual(
            result.expanded,
            (
                "1215_rd394041_amd_stream_moe_overlap",
                "1216_rd43_concurrent_join_fusion_guard",
                "1217_rd44_graph_opt_default_rdna35",
            ),
        )
        self.assertEqual(
            result.pulled_in,
            ("1215_rd394041_amd_stream_moe_overlap", "1216_rd43_concurrent_join_fusion_guard"),
        )

    def test_a_patch_with_no_requires_expands_to_itself(self):
        result = patchset.expand_composition(["0100_cmake_options"])
        self.assertEqual(result.expanded, ("0100_cmake_options",))
        self.assertEqual(result.pulled_in, ())

    def test_expanded_output_of_1217s_closure_passes_resolve_exact(self):
        # The whole point: the caller can now hand the expanded set straight
        # to resolve_exact() and have it pass, without hand-listing it.
        result = patchset.expand_composition(["1217_rd44_graph_opt_default_rdna35"])
        resolved = patchset.resolve_exact(list(result.expanded))
        self.assertEqual(len(resolved.modules), 3)

    def test_unknown_patch_id_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown patch"):
            patchset.expand_composition(["does-not-exist"])

    def test_cycle_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0100_a.py").write_text(
                "PATCH=None\nREQUIRES=('0200_b',)\n", encoding="utf-8")
            (root / "0200_b.py").write_text(
                "PATCH=None\nREQUIRES=('0100_a',)\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "REQUIRES cycle"):
                patchset.expand_composition(["0100_a"], directory=root)

    def test_diamond_dependency_expands_without_duplication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0100_base.py").write_text("PATCH=None\n", encoding="utf-8")
            (root / "0200_left.py").write_text(
                "PATCH=None\nREQUIRES=('0100_base',)\n", encoding="utf-8")
            (root / "0300_right.py").write_text(
                "PATCH=None\nREQUIRES=('0100_base',)\n", encoding="utf-8")
            (root / "0400_top.py").write_text(
                "PATCH=None\nREQUIRES=('0200_left', '0300_right')\n", encoding="utf-8")
            result = patchset.expand_composition(["0400_top"], directory=root)
            self.assertEqual(
                result.expanded,
                ("0100_base", "0200_left", "0300_right", "0400_top"),
            )


if __name__ == "__main__":
    unittest.main()
