"""Bridge from v2 campaign config sources to workspace.SourcePlan (RE14)."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import config as campaign_config # noqa: E402
from bigcherry.patch import patchset # noqa: E402
from bigcherry.core import paths # noqa: E402
from bigcherry.source import workspace # noqa: E402
from bigcherry.campaign.source import (CampaignSourceError, source_plan_for,  # noqa: E402
                                       source_plan_id)

_RECIPES_PATH = paths.RECIPES


class CampaignSourceTests(unittest.TestCase):
    """Against the real recipes.toml/patch catalog, like test_campaign_resolution.py --

    resolve_lane() rejects any injected catalog whose patch-id set does not
    exactly match the physical catalog on disk (a deliberate safety check),
    so these tests cannot use a synthetic recipes.toml with made-up patch
    names the way earlier revisions of this file did.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = campaign_config.load(_RECIPES_PATH)
        cls.catalog = patchset.catalog()

    def test_source_with_no_patch_sets_has_no_patches(self):
        plan = source_plan_for(self.cfg, "llama-native", catalog=self.catalog)
        self.assertEqual(plan.upstream_revision, self.cfg.pinned)
        self.assertEqual(plan.overlay_enabled, False)
        self.assertEqual(plan.patch_ids, ())
        self.assertIsNone(plan.required_state)

    def test_bigcherry_source_composes_patch_sets_via_resolve_lane(self):
        plan = source_plan_for(self.cfg, "bigcherry", catalog=self.catalog)
        self.assertEqual(plan.upstream_revision, self.cfg.pinned)
        self.assertEqual(plan.overlay_enabled, True)
        self.assertGreater(len(plan.patch_ids), 0)
        self.assertEqual(plan.required_state, "validated")
        # Cross-check against the same content-hash-validated resolution
        # campaign_resolution's own tests assert.
        from bigcherry.campaign import resolution as campaign_resolution
        lane = campaign_resolution.resolve_lane("bigcherry", self.cfg, self.catalog)
        self.assertEqual(plan.patch_ids, lane.patch_set.module_ids)

    def test_patch_set_id_and_classification_carry_through_from_resolve_lane(self):
        # RE03 (RV48 audit): resolve_lane already computes patch_set_id/
        # classification; source_plan_for() used to discard both -- nothing
        # downstream of resolution carried this reviewed logical-composition
        # identity. Now it survives into the SourcePlan itself.
        from bigcherry.campaign import resolution as campaign_resolution
        plan = source_plan_for(self.cfg, "bigcherry", catalog=self.catalog)
        lane = campaign_resolution.resolve_lane("bigcherry", self.cfg, self.catalog)
        self.assertEqual(plan.patch_set_id, lane.patch_set.patch_set_id)
        self.assertTrue(plan.patch_set_id)
        self.assertEqual(plan.classification, lane.patch_set.classification)

    def test_source_with_no_patch_sets_still_carries_a_patch_set_id(self):
        # Empty composition is still a real, reviewed identity (_digest of
        # an empty module list), not an absence of one.
        plan = source_plan_for(self.cfg, "llama-native", catalog=self.catalog)
        self.assertTrue(plan.patch_set_id)

    def test_unknown_source_rejected(self):
        with self.assertRaises(CampaignSourceError):
            source_plan_for(self.cfg, "does-not-exist", catalog=self.catalog)

    def test_defaults_to_the_real_physical_catalog_when_none_supplied(self):
        # No catalog= passed: source_plan_for must fetch patchset.catalog()
        # itself, not silently require every caller to supply one.
        plan = source_plan_for(self.cfg, "bigcherry")
        self.assertGreater(len(plan.patch_ids), 0)

    def test_experiment_lane_relaxes_materialize_state_gate(self):
        # The --experiment path exists to bench 'untested' patches. resolve_lane
        # enforces the base set's 'validated' state at planning and admits the
        # experiment module with required_state=None; SourcePlan must not re-
        # apply 'validated' to the merged module list at materialize time, or
        # the lane is rejected moments after planning approved it. Base lanes
        # keep the strict gate (covered by
        # test_bigcherry_source_composes_patch_sets_via_resolve_lane).
        plan = source_plan_for(
            self.cfg, "bigcherry-native", catalog=self.catalog,
            experiment="rd19-only")
        self.assertEqual(plan.classification, "experimental")
        self.assertIsNone(plan.required_state)
        self.assertIn("1200_rd19_single_gpu_meta_bypass", plan.patch_ids)


class ExplicitRefTests(unittest.TestCase):
    """A source with no patch-sets never touches the physical catalog inside
    resolve_lane (it returns an empty ResolvedPatchSet immediately), so this
    narrow bit of ref-resolution logic can use a minimal synthetic config
    without violating resolve_lane's catalog-must-match-disk invariant.
    """

    _DOC = """
version = 2
pinned = "abc123"

[source.explicit-ref]
ref = "deadbeef"
overlay = true
patch-sets = []
"""

    def test_explicit_ref_bypasses_pinned(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.toml"
            path.write_text(self._DOC, encoding="utf-8")
            cfg = campaign_config.load(path)
        plan = source_plan_for(cfg, "explicit-ref", catalog=[])
        self.assertEqual(plan.upstream_revision, "deadbeef")
        self.assertEqual(plan.patch_ids, ())


class SourcePlanIdTests(unittest.TestCase):
    def test_same_plan_produces_same_id(self):
        plan = workspace.SourcePlan("rev1", True, ("a", "b"), "validated")
        self.assertEqual(source_plan_id(plan), source_plan_id(plan))

    def test_patch_id_order_does_not_affect_identity(self):
        forward = workspace.SourcePlan("rev1", True, ("a", "b"), "validated")
        backward = workspace.SourcePlan("rev1", True, ("b", "a"), "validated")
        self.assertEqual(source_plan_id(forward), source_plan_id(backward))

    def test_differing_plans_produce_different_ids(self):
        base = workspace.SourcePlan("rev1", True, ("a",), "validated")
        different_revision = workspace.SourcePlan("rev2", True, ("a",), "validated")
        different_overlay = workspace.SourcePlan("rev1", False, ("a",), "validated")
        different_patches = workspace.SourcePlan("rev1", True, ("a", "b"), "validated")
        different_state = workspace.SourcePlan("rev1", True, ("a",), "untested")
        ids = {source_plan_id(p) for p in
               (base, different_revision, different_overlay, different_patches, different_state)}
        self.assertEqual(len(ids), 5)


if __name__ == "__main__":
    unittest.main()
