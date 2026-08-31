"""CliPatchSelection / resolve_cli_selection (compat.recipe removal plan).

gpt-dev-agent reviewed design, dev-gpt-agent gateway session
ses_5307d9c58ec645cb: exercised against the REAL project catalog/config,
same convention as test_campaign_resolution.py's CanonicalSelectionTests.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest

from bigcherry.patch import patchset, selection  # noqa: E402


def _args(**kwargs) -> Namespace:
    base = {"source": None, "recipe": None, "groups": None, "states": None}
    base.update(kwargs)
    return Namespace(**base)


class ResolveCliSelectionTests(unittest.TestCase):
    def test_source_selection_is_exact_mode_with_real_patch_set_identity(self):
        sel = selection.resolve_cli_selection(_args(source="bigcherry"))
        self.assertEqual(sel.mode, "exact")
        self.assertEqual(sel.source_name, "bigcherry")
        self.assertEqual(len(sel.patch_ids), 15)  # real framework patch-set
        self.assertIsNotNone(sel.patch_set_id)
        self.assertEqual(sel.overlay, True)
        self.assertIsNotNone(sel.overlay_digest)  # overlay=True -> real digest

    def test_source_with_no_overlay_leaves_overlay_digest_none(self):
        sel = selection.resolve_cli_selection(_args(source="llama-native"))
        self.assertEqual(sel.mode, "exact")
        self.assertEqual(sel.overlay, False)
        self.assertIsNone(sel.overlay_digest)
        self.assertEqual(sel.patch_ids, ())  # llama-native has no patch-sets

    def test_unknown_source_raises_selection_error(self):
        with self.assertRaises(selection.SelectionError):
            selection.resolve_cli_selection(_args(source="not-a-real-source"))

    def test_source_with_groups_is_rejected(self):
        with self.assertRaises(selection.SelectionError):
            selection.resolve_cli_selection(_args(source="bigcherry", groups="core"))

    def test_source_with_states_is_rejected(self):
        with self.assertRaises(selection.SelectionError):
            selection.resolve_cli_selection(_args(source="bigcherry", states="validated"))

    def test_recipe_selection_is_predicate_mode(self):
        sel = selection.resolve_cli_selection(_args(recipe="bigcherry"))
        self.assertEqual(sel.mode, "predicate")
        self.assertEqual(sel.states, frozenset({"validated"}))

    def test_recipe_with_groups_override_is_legal(self):
        # --recipe + --groups/--states stays legal during the compatibility
        # period -- only --source rejects the override axes.
        sel = selection.resolve_cli_selection(
            _args(recipe="bigcherry", groups="core")
        )
        self.assertEqual(sel.mode, "predicate")
        self.assertEqual(sel.groups, frozenset({"core"}))

    def test_no_selector_is_predicate_mode_matching_everything(self):
        sel = selection.resolve_cli_selection(_args())
        self.assertEqual(sel.mode, "predicate")
        self.assertIsNone(sel.groups)
        self.assertIsNone(sel.states)

    def test_unknown_recipe_raises_selection_error(self):
        with self.assertRaises(selection.SelectionError):
            selection.resolve_cli_selection(_args(recipe="not-a-real-recipe"))


class MatchesTests(unittest.TestCase):
    def test_exact_mode_matches_only_listed_ids(self):
        sel = selection.CliPatchSelection(
            mode="exact", label="", patch_ids=("0100_cmake_options",),
        )
        catalog = patchset.catalog()
        by_id = {m.patch_id: m for m in catalog}
        self.assertTrue(sel.matches(by_id["0100_cmake_options"]))
        self.assertFalse(sel.matches(by_id["1000_rdna4_mmq_q2k_q6k_fix"]))

    def test_predicate_mode_matches_by_group_and_state(self):
        sel = selection.CliPatchSelection(
            mode="predicate", label="",
            groups=frozenset({"core"}), states=frozenset({"validated"}),
        )
        catalog = patchset.catalog()
        by_id = {m.patch_id: m for m in catalog}
        core_validated = by_id["0100_cmake_options"]
        self.assertEqual(core_validated.group, "core")
        self.assertEqual(core_validated.state, "validated")
        self.assertTrue(sel.matches(core_validated))


class TreeStateKeyTests(unittest.TestCase):
    def test_predicate_mode_matches_legacy_tree_state_key_exactly(self):
        from bigcherry import recipes as recipes_module

        sel = selection.CliPatchSelection(
            mode="predicate", label="",
            groups=frozenset({"core"}), states=frozenset({"validated"}),
        )
        expected = recipes_module.tree_state_key(
            "deadbeef", frozenset({"core"}), frozenset({"validated"})
        )
        self.assertEqual(sel.tree_state_key("deadbeef"), expected)

    def test_exact_mode_key_changes_with_patch_set_id_not_just_ref(self):
        a = selection.CliPatchSelection(
            mode="exact", label="", patch_set_id="psid-a",
            overlay=True, overlay_digest="digest-x",
        )
        b = selection.CliPatchSelection(
            mode="exact", label="", patch_set_id="psid-b",
            overlay=True, overlay_digest="digest-x",
        )
        self.assertNotEqual(a.tree_state_key("deadbeef"), b.tree_state_key("deadbeef"))

    def test_exact_mode_key_excludes_source_name_by_design(self):
        # Two source aliases resolving to byte-identical composition should
        # share one checkout-state key -- this is the tree-state IDENTITY,
        # not the logical recipe identity (that's patch_set_id's job).
        a = selection.CliPatchSelection(
            mode="exact", label="", source_name="alias-a", patch_set_id="psid",
            overlay=True, overlay_digest="digest-x",
        )
        b = selection.CliPatchSelection(
            mode="exact", label="", source_name="alias-b", patch_set_id="psid",
            overlay=True, overlay_digest="digest-x",
        )
        self.assertEqual(a.tree_state_key("deadbeef"), b.tree_state_key("deadbeef"))

    def test_exact_mode_without_overlay_resolved_raises(self):
        sel = selection.CliPatchSelection(mode="exact", label="", patch_set_id="psid")
        with self.assertRaises(selection.SelectionError):
            sel.tree_state_key("deadbeef")

    def test_exact_mode_overlay_false_ignores_overlay_digest(self):
        a = selection.CliPatchSelection(
            mode="exact", label="", patch_set_id="psid",
            overlay=False, overlay_digest="should-be-ignored",
        )
        b = selection.CliPatchSelection(
            mode="exact", label="", patch_set_id="psid",
            overlay=False, overlay_digest="different-but-ignored",
        )
        self.assertEqual(a.tree_state_key("deadbeef"), b.tree_state_key("deadbeef"))


if __name__ == "__main__":
    unittest.main()
