"""RAD01 PR 3: every engine declares its recipes document, and a document names the engine it belongs to."""

import tempfile
import unittest
from pathlib import Path

from bigcherry.core import config, engines, paths


class EngineRecipesTests(unittest.TestCase):
    def test_every_declared_engine_loads_its_own_recipes(self):
        for name in engines.names(paths.REPO_ROOT):
            layout = engines.load(paths.REPO_ROOT, name)
            cfg = config.load(layout.recipes_path(paths.REPO_ROOT))
            self.assertEqual(cfg.engine, name)

    def test_radiance_recipes_carry_its_pin_sources_and_profiles(self):
        cfg = config.load(engines.load(paths.REPO_ROOT, engines.RADIANCE).recipes_path(paths.REPO_ROOT))
        self.assertEqual(cfg.pinned, "89cee7ce414486c717aa2085c99e1f522e4c0ea6")
        self.assertEqual(set(cfg.sources), {"radiance-native", "radiance", "radiance-lab"})
        self.assertEqual(set(cfg.runtime_profiles), {"flashnext-2xtx", "qwen38-27b-2xtx"})
        flash = cfg.runtime_profiles["flashnext-2xtx"].server_args
        self.assertIn("exact", flash)
        self.assertFalse(any(arg.startswith("R3_") for arg in flash))

    def test_radiance_patch_sets_name_only_radiance_packages(self):
        cfg = config.load(engines.load(paths.REPO_ROOT, engines.RADIANCE).recipes_path(paths.REPO_ROOT))
        root = paths.patches_root(engines.RADIANCE)
        for patch_set in cfg.patch_sets.values():
            for patch_id in patch_set.patches:
                self.assertTrue((root / patch_id / "patch.toml").is_file(), patch_id)
        for experiment in cfg.experiments.values():
            for patch_id in experiment.patches:
                self.assertTrue((root / patch_id / "patch.toml").is_file(), patch_id)

    def _write(self, text: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8")
        handle.write(text)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return Path(handle.name)

    def test_a_document_without_an_engine_is_refused(self):
        with self.assertRaisesRegex(config.ConfigError, "engine must name"):
            config.load(self._write('version = 2\npinned = "b1"\n'))

    def test_an_undeclared_engine_is_refused(self):
        with self.assertRaisesRegex(config.ConfigError, "not declared"):
            config.load(self._write('version = 2\nengine = "vllm"\npinned = "b1"\n'))


if __name__ == "__main__":
    unittest.main()
