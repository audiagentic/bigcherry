"""Exact patch-set/lane policy tests."""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.campaign import resolution as campaign_resolution
from bigcherry.core import config, paths
from bigcherry.patch import patchset # noqa: E402


def _write_patch(patches_root: Path, patch_id: str, *, marker_text: str) -> None:
    patches_root.mkdir(parents=True, exist_ok=True)
    (patches_root / f"{patch_id}.py").write_text(
        "from bigcherry.patcher import Edit, FilePatch\n"
        "GROUP = 'core'\n"
        "STATE = 'validated'\n"
        "PATCH = FilePatch(\n"
        "    path='source.txt',\n"
        "    edits=(Edit(\n"
        f"        id='{patch_id}',\n"
        "        anchor=r'one',\n"
        f"        text='{marker_text}',\n"
        "        mode='insert_after',\n"
        "    ),),\n"
        ")\n",
        encoding="utf-8",
    )


class CampaignResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config.load(paths.RECIPES)
        cls.catalog = patchset.catalog()

    def test_upstream_has_no_bigcherry_modules(self):
        lane = campaign_resolution.resolve_lane("llama-native", self.cfg, self.catalog)
        self.assertEqual(lane.patch_set.module_ids, ())
        self.assertEqual(lane.patch_set.classification, "upstream")

    def test_base_is_exactly_the_fifteen_validated_core_modules(self):
        lane = campaign_resolution.resolve_lane("bigcherry-native", self.cfg, self.catalog)
        # Scoped to the framework patch-set's own declared list, not "every
        # validated module in the catalog" -- since RD19's promotion
        # (2026-08-24), a validated module can also live in
        # validated-enhancements, which bigcherry-native's "framework"
        # patch-set must NOT pull in.
        framework_patch_ids = frozenset(self.cfg.patch_sets["framework"].patches)
        expected = tuple(
            module.patch_id for module in self.catalog
            if module.state == "validated" and module.patch_id in framework_patch_ids
        )
        # HI70: patches/1100_hi70_direct_op_evidence/patch.py added a 15th
        # validated core module (deterministic direct-op correctness corpus
        # for MMQ fb1 / MMF nwarps candidates).
        self.assertEqual(len(expected), 15)
        self.assertEqual(lane.patch_set.module_ids, expected)
        self.assertEqual(
            sum(module.group == "core" for module in self.catalog
                if module.state == "validated" and module.patch_id in framework_patch_ids),
            14,
        )
        # bigcherry-native is FRAMEWORK ONLY -- it must never report or build
        # a promoted enhancement. That separation is what makes it usable as
        # the control arm of a validation campaign: if enhancements leaked in
        # here, every A/B would be measured against a moving baseline.
        #
        # This assertion caught a real defect the moment validated-enhancements
        # stopped being empty (2026-09-05, RD73): promoted_enhancements
        # returned the whole global set for EVERY source, so the
        # framework-only control source reported an enhancement it does not
        # build. Selection itself was always correct; the report was not.
        #
        # (RD19 was briefly in validated-enhancements on 2026-08-24; that
        # promotion post-dated the HI83 evidence contract with no qualifying
        # evidence, and was deliberately reverted -- see
        # docs/planning/active/patch-system/PA05.md.)
        self.assertEqual(lane.promoted_enhancements, ())
        # ...and belt-and-braces on the thing that actually gets built.
        non_empty = self.cfg.patch_sets["validated-enhancements"].patches
        for patch_id in non_empty:
            self.assertNotIn(
                patch_id, lane.patch_set.module_ids,
                f"{patch_id} leaked into the framework-only control source",
            )

    def test_release_source_is_framework_plus_validated_enhancements(self):
        """[source.bigcherry] is the release build: framework + whatever has
        actually qualified. This pins the STRUCTURE rather than a count, so
        promoting a patch does not break the test -- only breaking the
        composition does."""
        native = campaign_resolution.resolve_lane(
            "bigcherry-native", self.cfg, self.catalog)
        release = campaign_resolution.resolve_lane(
            "bigcherry", self.cfg, self.catalog)

        native_ids = set(native.patch_set.module_ids)
        release_ids = set(release.patch_set.module_ids)

        # The release build is a strict superset of the native baseline...
        self.assertTrue(native_ids <= release_ids)
        # ...and everything extra is exactly the promoted enhancements.
        self.assertEqual(release_ids - native_ids, set(release.promoted_enhancements))
        # Every promoted enhancement must really be STATE="validated" --
        # required-state on the patch-set is what enforces this, and a
        # release build must never ship an unvalidated patch.
        by_id = {module.patch_id: module for module in self.catalog}
        for patch_id in release.promoted_enhancements:
            self.assertEqual(
                by_id[patch_id].state, "validated",
                f"{patch_id} is in a release patch-set but is not validated",
            )

    def test_one_explicit_experiment_does_not_leak_all_noncore_patches(self):
        experiment = config.Experiment(
            name="one-fix", patches=("1002_hip_unsafe_math_opt_in",),
            cmake_options=(), runtime_env=(), requires=(), conflicts=(),
        )
        cfg = dataclasses.replace(self.cfg, experiments={"one-fix": experiment})
        lane = campaign_resolution.resolve_lane(
            "bigcherry-native", cfg, self.catalog, experiment="one-fix"
        )
        self.assertEqual(len(lane.patch_set.module_ids), 16)
        self.assertIn("1002_hip_unsafe_math_opt_in", lane.patch_set.module_ids)
        self.assertNotIn("1003_quantized_cpy_thread_block_fix", lane.patch_set.module_ids)
        self.assertEqual(lane.patch_set.classification, "experimental")


class CanonicalSelectionTests(unittest.TestCase):
    """resolve_canonical_selection() is the migration helper replacing
    recipes.py's legacy [compat.recipe.*] bridge -- see the compat.recipe
    removal plan. These tests pin its exact contract before any --recipe
    consumer is migrated onto it."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = config.load(paths.RECIPES)
        cls.catalog = patchset.catalog()

    def test_unknown_source_raises(self):
        with self.assertRaises(campaign_resolution.ResolutionError):
            campaign_resolution.resolve_canonical_selection(
                "not-a-real-source", self.cfg, self.catalog
            )

    def test_ref_resolves_the_pinned_sentinel_exactly_like_campaign_source(self):
        # Mirrors campaign/source.py's own
        # `cfg.pinned if source.ref == "pinned" else source.ref` -- a
        # migrated legacy caller must see the identical ref it saw under
        # the old Recipe.follows_pin semantics.
        selection = campaign_resolution.resolve_canonical_selection(
            "bigcherry", self.cfg, self.catalog
        )
        source = self.cfg.sources["bigcherry"]
        expected_ref = self.cfg.pinned if source.ref == "pinned" else source.ref
        self.assertEqual(selection.source_ref, expected_ref)
        self.assertEqual(selection.source_name, "bigcherry")

    def test_patch_ids_and_patch_set_id_match_resolve_lane_exactly(self):
        # The whole point of this helper is to expose resolve_lane()'s real
        # patch-set identity, not a second, independently-derived one --
        # confirm byte-for-byte agreement, not just "looks similar".
        lane = campaign_resolution.resolve_lane("bigcherry", self.cfg, self.catalog)
        selection = campaign_resolution.resolve_canonical_selection(
            "bigcherry", self.cfg, self.catalog
        )
        self.assertEqual(selection.patch_ids, lane.patch_set.module_ids)
        self.assertEqual(selection.patch_set_id, lane.patch_set.patch_set_id)

    def test_experiment_forwards_through_to_the_resolved_patch_set(self):
        experiment = config.Experiment(
            name="one-fix", patches=("1002_hip_unsafe_math_opt_in",),
            cmake_options=(), runtime_env=(), requires=(), conflicts=(),
        )
        cfg = dataclasses.replace(self.cfg, experiments={"one-fix": experiment})
        selection = campaign_resolution.resolve_canonical_selection(
            "bigcherry-native", cfg, self.catalog, experiment="one-fix"
        )
        self.assertIn("1002_hip_unsafe_math_opt_in", selection.patch_ids)


class PatchSetIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config.load(paths.RECIPES)
        cls.catalog = patchset.catalog()

    def test_all_is_rejected_and_identity_changes_with_content(self):
        with self.assertRaisesRegex(campaign_resolution.ResolutionError, "not a valid"):
            campaign_resolution.resolve_patch_set("all", self.cfg, self.catalog)
        first = campaign_resolution.resolve_lane("bigcherry-native", self.cfg, self.catalog)
        changed = dataclasses.replace(
            self.cfg.patch_sets["framework"],
            patches=self.cfg.patch_sets["framework"].patches[:-1],
        )
        cfg = dataclasses.replace(
            self.cfg, patch_sets={**self.cfg.patch_sets, "framework": changed}
        )
        second = campaign_resolution.resolve_lane("bigcherry-native", cfg, self.catalog)
        self.assertNotEqual(first.patch_set.patch_set_id, second.patch_set.patch_set_id)


class MultiPatchSetCompositionIdentityTests(unittest.TestCase):
    """GPT-auto-agent review (RE03 comprehensive follow-up, 2026-08-17):
    RE03's own contract says two byte-identical sources can have distinct
    REVIEWED logical compositions and must get distinct patch_set_ids.
    resolve_lane()'s multi-set case used to flatten every combination into
    one synthetic "__merged__" name before hashing identity, so two
    genuinely different named-set compositions resolving to the same
    modules/state/classification collided on one patch_set_id."""

    def test_two_different_multiset_compositions_resolving_to_the_same_modules_do_not_collide(self):
        with tempfile.TemporaryDirectory() as directory:
            patches_root = Path(directory) / "patches"
            _write_patch(patches_root, "0001_a", marker_text="a")
            _write_patch(patches_root, "0002_b", marker_text="b")
            catalog = patchset.catalog(directory=patches_root)

            cfg = config.Config(
                pinned="unused",
                patch_sets={
                    "set-a": config.PatchSet(name="set-a", patches=("0001_a",), required_state="validated"),
                    "set-b": config.PatchSet(name="set-b", patches=("0002_b",), required_state="validated"),
                    "set-ab": config.PatchSet(
                        name="set-ab", patches=("0001_a", "0002_b"), required_state="validated"),
                },
                sources={
                    # Two-set composition [set-a, set-b] and single-set
                    # [set-ab] resolve to the IDENTICAL final module set
                    # ("0001_a", "0002_b") -- same bytes, different
                    # reviewed logical composition.
                    "via-two-sets": config.Source(
                        name="via-two-sets", ref="pinned", overlay=False,
                        patch_sets=("set-a", "set-b")),
                    "via-one-set": config.Source(
                        name="via-one-set", ref="pinned", overlay=False,
                        patch_sets=("set-ab",)),
                },
                builds={}, platforms={}, experiments={}, campaigns={},
                path=Path(directory) / "recipes.toml",
            )

            lane_two_sets = campaign_resolution.resolve_lane(
                "via-two-sets", cfg, catalog, catalog_directory=patches_root)
            lane_one_set = campaign_resolution.resolve_lane(
                "via-one-set", cfg, catalog, catalog_directory=patches_root)

            self.assertEqual(lane_two_sets.patch_set.module_ids, lane_one_set.patch_set.module_ids)
            self.assertNotEqual(
                lane_two_sets.patch_set.patch_set_id, lane_one_set.patch_set.patch_set_id)

    def test_two_different_two_set_compositions_resolving_to_the_same_modules_do_not_collide(self):
        # The narrower case: TWO different multi-set compositions (both
        # going through the "__merged__" synthetic-name path), not one
        # multi-set vs one single-set.
        with tempfile.TemporaryDirectory() as directory:
            patches_root = Path(directory) / "patches"
            _write_patch(patches_root, "0001_a", marker_text="a")
            _write_patch(patches_root, "0002_b", marker_text="b")
            catalog = patchset.catalog(directory=patches_root)

            cfg = config.Config(
                pinned="unused",
                patch_sets={
                    "set-a": config.PatchSet(name="set-a", patches=("0001_a",), required_state="validated"),
                    "set-b": config.PatchSet(name="set-b", patches=("0002_b",), required_state="validated"),
                    "set-empty-1": config.PatchSet(name="set-empty-1", patches=(), required_state="validated"),
                    "set-empty-2": config.PatchSet(name="set-empty-2", patches=(), required_state="validated"),
                },
                sources={
                    "via-a-then-empty1": config.Source(
                        name="via-a-then-empty1", ref="pinned", overlay=False,
                        patch_sets=("set-a", "set-b", "set-empty-1")),
                    "via-a-then-empty2": config.Source(
                        name="via-a-then-empty2", ref="pinned", overlay=False,
                        patch_sets=("set-a", "set-b", "set-empty-2")),
                },
                builds={}, platforms={}, experiments={}, campaigns={},
                path=Path(directory) / "recipes.toml",
            )

            lane_1 = campaign_resolution.resolve_lane(
                "via-a-then-empty1", cfg, catalog, catalog_directory=patches_root)
            lane_2 = campaign_resolution.resolve_lane(
                "via-a-then-empty2", cfg, catalog, catalog_directory=patches_root)

            self.assertEqual(lane_1.patch_set.module_ids, lane_2.patch_set.module_ids)
            self.assertNotEqual(lane_1.patch_set.patch_set_id, lane_2.patch_set.patch_set_id)


class EmptyBaseExperimentResolutionTests(unittest.TestCase):
    """External patch-management review, 2026-08-20: resolve_lane() used to
    return on an empty-base source (no patch_sets, e.g. a clean Vulkan
    stock lane) before `experiment` was ever consulted, silently dropping
    any requested experiment instead of resolving it."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.patches_root = Path(self.directory.name) / "patches"
        _write_patch(self.patches_root, "0001_a", marker_text="a")
        self.catalog = patchset.catalog(directory=self.patches_root)
        experiment = config.Experiment(
            name="exp-a", patches=("0001_a",),
            cmake_options=(), runtime_env=(), requires=(), conflicts=(),
        )
        self.cfg = config.Config(
            pinned="unused",
            patch_sets={},
            sources={
                "clean": config.Source(
                    name="clean", ref="pinned", overlay=False, patch_sets=()),
            },
            builds={}, platforms={}, experiments={"exp-a": experiment}, campaigns={},
            path=Path(self.directory.name) / "recipes.toml",
        )

    def test_empty_base_no_experiment_resolves_empty(self):
        lane = campaign_resolution.resolve_lane(
            "clean", self.cfg, self.catalog, catalog_directory=self.patches_root)
        self.assertEqual(lane.patch_set.module_ids, ())
        self.assertEqual(lane.patch_set.classification, "upstream")

    def test_empty_base_with_experiment_resolves_exactly_the_experiment(self):
        lane = campaign_resolution.resolve_lane(
            "clean", self.cfg, self.catalog, experiment="exp-a",
            catalog_directory=self.patches_root)
        self.assertEqual(lane.patch_set.module_ids, ("0001_a",))
        self.assertEqual(lane.patch_set.classification, "experimental")

    def test_empty_base_with_unknown_experiment_fails_closed(self):
        with self.assertRaisesRegex(campaign_resolution.ResolutionError, "unknown experiment"):
            campaign_resolution.resolve_lane(
                "clean", self.cfg, self.catalog, experiment="not-real",
                catalog_directory=self.patches_root)


class MultiSetIndependentRequiredStateTests(unittest.TestCase):
    """External patch-management review, 2026-08-20: a multi-patch-set
    source used to flatten every set into one synthetic list and validate
    ALL of it under only the first set's required_state -- silently wrong
    once two named sets in the same source genuinely diverge."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.patches_root = Path(self.directory.name) / "patches"
        _write_patch(self.patches_root, "0001_a", marker_text="a")
        _write_patch(self.patches_root, "0002_b", marker_text="b")

    def _catalog_with_states(self, state_a: str, state_b: str) -> list[patchset.PatchModule]:
        (self.patches_root / "0001_a.py").write_text(
            (self.patches_root / "0001_a.py").read_text(encoding="utf-8").replace(
                "STATE = 'validated'", f"STATE = {state_a!r}"),
            encoding="utf-8",
        )
        (self.patches_root / "0002_b.py").write_text(
            (self.patches_root / "0002_b.py").read_text(encoding="utf-8").replace(
                "STATE = 'validated'", f"STATE = {state_b!r}"),
            encoding="utf-8",
        )
        return patchset.catalog(directory=self.patches_root)

    def test_diverging_policies_validate_each_set_independently(self):
        # set-a requires 'validated' and 0001_a really is validated; set-b
        # requires 'untested' and 0002_b really is untested. Under the OLD
        # buggy behaviour (first set's policy applied to everything), set-b
        # would incorrectly be validated against 'validated' too -- which
        # 0002_b would actually still pass here, so use the inverse: make
        # 0001_a untested (fails set-a's own 'validated' requirement) while
        # 0002_b stays validated (passes set-b's 'untested' requirement
        # trivially) is not a clean signal either. Instead assert the
        # positive case directly: each set's own real state is honoured.
        catalog = self._catalog_with_states("validated", "untested")
        cfg = config.Config(
            pinned="unused",
            patch_sets={
                "set-a": config.PatchSet(name="set-a", patches=("0001_a",), required_state="validated"),
                "set-b": config.PatchSet(name="set-b", patches=("0002_b",), required_state="untested"),
            },
            sources={
                "multi": config.Source(
                    name="multi", ref="pinned", overlay=False, patch_sets=("set-a", "set-b")),
            },
            builds={}, platforms={}, experiments={}, campaigns={},
            path=Path(self.directory.name) / "recipes.toml",
        )
        lane = campaign_resolution.resolve_lane(
            "multi", cfg, catalog, catalog_directory=self.patches_root)
        self.assertEqual(set(lane.patch_set.module_ids), {"0001_a", "0002_b"})
        # Two genuinely different policies were used -- the composite
        # required_state is honestly ambiguous, not silently the first
        # set's value.
        self.assertIsNone(lane.patch_set.required_state)

    def test_diverging_policy_rejects_a_module_that_fails_its_own_sets_policy(self):
        # 0001_a is only 'untested' but set-a demands 'validated' -- under
        # the OLD bug this couldn't even be expressed (everything used
        # source.patch_sets[0]'s policy, so as long as the FIRST set's
        # patches matched the first policy, nothing failed); now each set
        # is validated against its own declared policy independently and
        # this must fail.
        catalog = self._catalog_with_states("untested", "validated")
        cfg = config.Config(
            pinned="unused",
            patch_sets={
                "set-a": config.PatchSet(name="set-a", patches=("0001_a",), required_state="validated"),
                "set-b": config.PatchSet(name="set-b", patches=("0002_b",), required_state="validated"),
            },
            sources={
                "multi": config.Source(
                    name="multi", ref="pinned", overlay=False, patch_sets=("set-a", "set-b")),
            },
            builds={}, platforms={}, experiments={}, campaigns={},
            path=Path(self.directory.name) / "recipes.toml",
        )
        with self.assertRaisesRegex(ValueError, "does not satisfy required state"):
            campaign_resolution.resolve_lane(
                "multi", cfg, catalog, catalog_directory=self.patches_root)

    def test_shared_policy_across_sets_is_unchanged_backward_compatible(self):
        # Today's only real production shape (bigcherry: framework +
        # validated-enhancements, both 'validated') -- required_state must
        # resolve to the shared string, not None, so patch_set_id does not
        # silently change for every currently-configured multi-set source.
        catalog = self._catalog_with_states("validated", "validated")
        cfg = config.Config(
            pinned="unused",
            patch_sets={
                "set-a": config.PatchSet(name="set-a", patches=("0001_a",), required_state="validated"),
                "set-b": config.PatchSet(name="set-b", patches=("0002_b",), required_state="validated"),
            },
            sources={
                "multi": config.Source(
                    name="multi", ref="pinned", overlay=False, patch_sets=("set-a", "set-b")),
            },
            builds={}, platforms={}, experiments={}, campaigns={},
            path=Path(self.directory.name) / "recipes.toml",
        )
        lane = campaign_resolution.resolve_lane(
            "multi", cfg, catalog, catalog_directory=self.patches_root)
        self.assertEqual(lane.patch_set.required_state, "validated")
        self.assertEqual(set(lane.patch_set.module_ids), {"0001_a", "0002_b"})


if __name__ == "__main__":
    unittest.main()


class PerLaneExperimentTests(unittest.TestCase):
    """VA26: a patch-qualification profile holds arms that carry the patch and
    arms that deliberately do not. A request-level --experiment applies to
    every lane and so cannot express it."""

    def setUp(self):
        from bigcherry.core import config, paths
        self.cfg = config.load(paths.RECIPES)

    def test_profile_declares_patched_and_unpatched_arms(self):
        lanes = self.cfg.campaigns["patch-qualification"].lanes
        experiments = [lane.experiment for lane in lanes]
        self.assertIn(None, experiments, "baseline arms must carry no experiment")
        self.assertTrue(
            any(e is not None for e in experiments),
            "at least one arm must carry the patch under test",
        )

    def test_same_source_appears_patched_and_unpatched_without_colliding(self):
        # The pair that gives the isolated comparison its meaning: identical
        # source/build/platform, differing only by the experiment. Before
        # lane_id folded in the experiment these collided and the duplicate
        # check dropped one.
        from bigcherry.campaign.planner import CampaignRequest, lane_id, plan

        lanes = plan(
            CampaignRequest(
                selectors=tuple(self.cfg.campaigns["patch-qualification"].lanes),
                architectures=("gfx1100",),
            ),
            self.cfg,
        )
        ids = [lane_id(lane) for lane in lanes]
        # Pin the PROPERTY (no collisions, and the patched/unpatched pair both
        # survive), not a lane count -- the profile legitimately grows as
        # build variants like tune/replay are added.
        self.assertEqual(len(ids), len(set(ids)), f"lane ids collided: {ids}")
        native = [i for i in ids if i.startswith("bigcherry-native:control")]
        self.assertEqual(len(native), 2, "expected a patched and unpatched pair")

    def test_lane_experiment_overrides_request_level(self):
        from bigcherry.campaign.planner import CampaignRequest, plan

        lanes = plan(
            CampaignRequest(
                selectors=tuple(self.cfg.campaigns["patch-qualification"].lanes),
                architectures=("gfx1100",), experiment="rd73-only",
            ),
            self.cfg,
        )
        # Request-level fills the baselines that declare none; a lane that
        # declares its own keeps it.
        by_source = {(l.source_name, l.experiment) for l in lanes}
        self.assertIn(("bigcherry", "rd73-only"), by_source)

    def test_unknown_lane_experiment_is_rejected(self):
        # A profile naming an experiment that does not exist must fail at load
        # time, not silently plan an arm that is identical to its baseline --
        # which would make the comparison quietly meaningless.
        text = paths.RECIPES.read_text(encoding="utf-8").replace(
            'experiment = "rd73-only"', 'experiment = "no-such-experiment"', 1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            recipes = Path(tmp) / "recipes.toml"
            recipes.write_text(text, encoding="utf-8")
            with self.assertRaises(config.ConfigError) as caught:
                config.load(recipes)
        self.assertIn("no-such-experiment", str(caught.exception))
