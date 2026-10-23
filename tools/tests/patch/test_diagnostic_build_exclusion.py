"""MEN12: a diagnostic-build patch (rad0004) must never enter a production source, patch set or experiment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bigcherry.core import config, engines, paths
from bigcherry.patch import gates, registry

RAD0002 = "rad0002_moe_remote_handoff"
RAD0004 = "rad0004_moe_remote_diagnostics"
RAD0006 = "rad0006_heat_freeze_diagnostic"


def _recipes(body: str) -> config.Config:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "recipes.toml"
        path.write_text('version = 2\nengine = "radiance"\npinned = "abc1234"\n' + body, encoding="utf-8")
        return config.load(path)


def _registry(*diagnostic: str, plain: tuple[str, ...] = ()):
    descriptors = [SimpleNamespace(patch_id=pid, diagnostic_build="BC_MOE_REMOTE_DIAG") for pid in diagnostic]
    descriptors += [SimpleNamespace(patch_id=pid, diagnostic_build=None) for pid in plain]
    return SimpleNamespace(descriptors=descriptors)


class DiagnosticBuildExclusionTests(unittest.TestCase):
    def test_rad0004_declares_its_macro_and_is_not_a_production_package(self):
        root = engines.load(paths.REPO_ROOT, "radiance").patches_root(paths.REPO_ROOT)
        reg = registry.load_registry("radiance", root)
        self.assertEqual(reg.get(RAD0004).diagnostic_build, "BC_MOE_REMOTE_DIAG")
        self.assertEqual(reg.get(RAD0004).kind, "diagnostic")
        self.assertEqual(reg.get(RAD0004).requires, (RAD0002,))
        self.assertIsNone(reg.get(RAD0002).diagnostic_build)

    def test_the_real_radiance_recipes_keep_rad0004_out_of_production(self):
        layout = engines.load(paths.REPO_ROOT, "radiance")
        cfg = config.load(layout.recipes_path(paths.REPO_ROOT))
        reg = registry.load_registry("radiance", layout.patches_root(paths.REPO_ROOT))
        self.assertEqual(gates.diagnostic_build_problems(reg, cfg), [])
        for name, patch_set in cfg.patch_sets.items():
            if RAD0004 in patch_set.patches:
                self.assertTrue(name.endswith("-diag"), name)
        for name, experiment in cfg.experiments.items():
            self.assertNotIn(RAD0004, experiment.patches, name)
        for name, source in cfg.sources.items():
            carries = any(RAD0004 in cfg.patch_sets[s].patches for s in source.patch_sets)
            self.assertEqual(carries, name == "radiance-diag", name)

    def test_the_diagnostic_source_is_production_plus_rad0004_with_the_macro_build(self):
        cfg = config.load(engines.load(paths.REPO_ROOT, "radiance").recipes_path(paths.REPO_ROOT))
        diag = cfg.sources["radiance-diag"]
        production = cfg.sources["radiance"]
        self.assertTrue(set(production.patch_sets) <= set(diag.patch_sets))
        extra = set(diag.patch_sets) - set(production.patch_sets)
        self.assertEqual(extra, {"moe-remote-diag"})
        self.assertEqual(cfg.patch_sets["moe-remote-diag"].patches, (RAD0002, RAD0004, RAD0006))
        self.assertIn(("CMAKE_CXX_FLAGS", "-DBC_MOE_REMOTE_DIAG"), cfg.builds["moe-diag"].options)

    def test_a_production_patch_set_with_rad0004_fails(self):
        cfg = _recipes(f'[patch-set.serving-core]\npatches = ["{RAD0004}"]\n[source.radiance]\noverlay = false\n'
                       'patch-sets = ["serving-core"]\n')
        problems = gates.diagnostic_build_problems(_registry(RAD0004), cfg)
        self.assertEqual(len(problems), 1)
        self.assertIn("patch set 'serving-core'", problems[0])

    def test_a_diag_patch_set_used_by_a_production_source_fails(self):
        cfg = _recipes(f'[patch-set.moe-remote-diag]\npatches = ["{RAD0004}"]\n[source.radiance]\noverlay = false\n'
                       'patch-sets = ["moe-remote-diag"]\n')
        problems = gates.diagnostic_build_problems(_registry(RAD0004), cfg)
        self.assertEqual(len(problems), 1)
        self.assertIn("source 'radiance'", problems[0])

    def test_a_production_experiment_with_rad0004_fails(self):
        cfg = _recipes(f'[experiment.moe-remote-handoff]\npatches = ["{RAD0002}", "{RAD0004}"]\n')
        problems = gates.diagnostic_build_problems(_registry(RAD0004, plain=(RAD0002,)), cfg)
        self.assertEqual(len(problems), 1)
        self.assertIn("experiment 'moe-remote-handoff'", problems[0])

    def test_diag_named_recipes_may_carry_it(self):
        cfg = _recipes(f'[patch-set.moe-remote-diag]\npatches = ["{RAD0002}", "{RAD0004}"]\n'
                       '[source.radiance-diag]\noverlay = false\npatch-sets = ["moe-remote-diag"]\n'
                       f'[experiment.moe-remote-diag]\npatches = ["{RAD0004}"]\n')
        self.assertEqual(gates.diagnostic_build_problems(_registry(RAD0004, plain=(RAD0002,)), cfg), [])


if __name__ == "__main__":
    unittest.main()
