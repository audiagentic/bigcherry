"""Exact patch-set/lane policy tests."""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import campaign_resolution, config, patchset  # noqa: E402


class CampaignResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config.load(Path(__file__).resolve().parents[2] / "recipes.toml")
        cls.catalog = patchset.catalog()

    def test_upstream_has_no_bigcherry_modules(self):
        lane = campaign_resolution.resolve_lane("llama-native", self.cfg, self.catalog)
        self.assertEqual(lane.patch_set.module_ids, ())
        self.assertEqual(lane.patch_set.classification, "upstream")

    def test_base_is_exactly_the_fourteen_validated_core_modules(self):
        lane = campaign_resolution.resolve_lane("bigcherry-native", self.cfg, self.catalog)
        expected = tuple(
            module.patch_id for module in self.catalog
            if module.state == "validated"
        )
        self.assertEqual(len(expected), 14)
        self.assertEqual(lane.patch_set.module_ids, expected)
        self.assertEqual(
            sum(module.group == "core" for module in self.catalog if module.state == "validated"),
            13,
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
        self.assertEqual(len(lane.patch_set.module_ids), 15)
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


if __name__ == "__main__":
    unittest.main()
