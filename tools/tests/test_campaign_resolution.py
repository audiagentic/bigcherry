"""Exact patch-set/lane policy tests."""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import campaign_resolution, config, patchset, paths  # noqa: E402


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
        expected = tuple(
            module.patch_id for module in self.catalog
            if module.state == "validated"
        )
        # HI70: patches/1100_hi70_direct_op_evidence.py added a 15th
        # validated core module (deterministic direct-op correctness corpus
        # for MMQ fb1 / MMF nwarps candidates).
        self.assertEqual(len(expected), 15)
        self.assertEqual(lane.patch_set.module_ids, expected)
        self.assertEqual(
            sum(module.group == "core" for module in self.catalog if module.state == "validated"),
            14,
        )
        self.assertEqual(lane.promoted_enhancements, ())

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
