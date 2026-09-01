"""CliPatchSelection / resolve_cli_selection.

``--source NAME`` is the sole selection mechanism -- exercised against the
REAL project catalog/config, same convention as
test_campaign_resolution.py's CanonicalSelectionTests.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest

from bigcherry.patch import patchset, selection  # noqa: E402


def _args(**kwargs) -> Namespace:
    base = {"source": None}
    base.update(kwargs)
    return Namespace(**base)


class ResolveCliSelectionTests(unittest.TestCase):
    def test_source_selection_has_real_patch_set_identity(self):
        sel = selection.resolve_cli_selection(_args(source="bigcherry"))
        self.assertEqual(sel.source_name, "bigcherry")
        self.assertEqual(len(sel.patch_ids), 15)  # real framework patch-set
        self.assertIsNotNone(sel.patch_set_id)
        self.assertEqual(sel.overlay, True)
        self.assertIsNotNone(sel.overlay_digest)  # overlay=True -> real digest

    def test_source_with_no_overlay_leaves_overlay_digest_none(self):
        sel = selection.resolve_cli_selection(_args(source="llama-native"))
        self.assertEqual(sel.overlay, False)
        self.assertIsNone(sel.overlay_digest)
        self.assertEqual(sel.patch_ids, ())  # llama-native has no patch-sets

    def test_unknown_source_raises_selection_error(self):
        with self.assertRaises(selection.SelectionError):
            selection.resolve_cli_selection(_args(source="not-a-real-source"))

    def test_no_selector_is_select_all(self):
        sel = selection.resolve_cli_selection(_args())
        self.assertTrue(sel.select_all)
        self.assertIsNone(sel.source_name)


class MatchesTests(unittest.TestCase):
    def test_matches_only_listed_ids(self):
        sel = selection.CliPatchSelection(
            label="", patch_ids=("0100_cmake_options",),
        )
        catalog = patchset.catalog()
        by_id = {m.patch_id: m for m in catalog}
        self.assertTrue(sel.matches(by_id["0100_cmake_options"]))
        self.assertFalse(sel.matches(by_id["1000_rdna4_mmq_q2k_q6k_fix"]))


class TreeStateKeyTests(unittest.TestCase):
    def test_key_changes_with_patch_set_id_not_just_ref(self):
        a = selection.CliPatchSelection(
            label="", patch_set_id="psid-a",
            overlay=True, overlay_digest="digest-x",
        )
        b = selection.CliPatchSelection(
            label="", patch_set_id="psid-b",
            overlay=True, overlay_digest="digest-x",
        )
        self.assertNotEqual(a.tree_state_key("deadbeef"), b.tree_state_key("deadbeef"))

    def test_key_excludes_source_name_by_design(self):
        # Two source aliases resolving to byte-identical composition should
        # share one checkout-state key -- this is the tree-state IDENTITY,
        # not the logical selection identity (that's patch_set_id's job).
        a = selection.CliPatchSelection(
            label="", source_name="alias-a", patch_set_id="psid",
            overlay=True, overlay_digest="digest-x",
        )
        b = selection.CliPatchSelection(
            label="", source_name="alias-b", patch_set_id="psid",
            overlay=True, overlay_digest="digest-x",
        )
        self.assertEqual(a.tree_state_key("deadbeef"), b.tree_state_key("deadbeef"))

    def test_without_overlay_resolved_raises(self):
        sel = selection.CliPatchSelection(label="", patch_set_id="psid")
        with self.assertRaises(selection.SelectionError):
            sel.tree_state_key("deadbeef")

    def test_overlay_false_ignores_overlay_digest(self):
        a = selection.CliPatchSelection(
            label="", patch_set_id="psid",
            overlay=False, overlay_digest="should-be-ignored",
        )
        b = selection.CliPatchSelection(
            label="", patch_set_id="psid",
            overlay=False, overlay_digest="different-but-ignored",
        )
        self.assertEqual(a.tree_state_key("deadbeef"), b.tree_state_key("deadbeef"))


if __name__ == "__main__":
    unittest.main()
