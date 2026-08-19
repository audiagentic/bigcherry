"""Exact patch-set/lane policy tests."""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import campaign_resolution, config, patchset  # noqa: E402


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
        cls.cfg = config.load(Path(__file__).resolve().parents[2] / "recipes.toml")
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


if __name__ == "__main__":
    unittest.main()
