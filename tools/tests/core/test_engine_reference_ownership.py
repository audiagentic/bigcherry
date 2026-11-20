"""Engine-owned static registries must not escape into generic config or serving scans."""

from __future__ import annotations

import unittest

from bigcherry.core import engines, paths
from bigcherry.source import sources


class EngineReferenceOwnershipTests(unittest.TestCase):
    def test_llamacpp_external_registry_comes_from_engine_declaration(self) -> None:
        layout = engines.load(paths.REPO_ROOT, engines.LLAMACPP)
        actual = layout.external_sources_path(paths.REPO_ROOT)
        self.assertIsNotNone(actual)
        self.assertEqual(paths.EXTERNAL_SOURCES, actual)
        self.assertEqual(actual, paths.REPO_ROOT / "engines/llamacpp/external-sources.toml")
        self.assertFalse((paths.REPO_ROOT / "config/external-sources.toml").exists())
        registry = sources.load_engine_registry("llamacpp")
        self.assertEqual(registry["version"], 1)
        self.assertGreater(len(registry["sources"]), 0)

    def test_radiance_reference_inventory_is_not_a_runtime_config(self) -> None:
        inventory = paths.REPO_ROOT / "engines/radiance/reference/serving-settings-inventory.toml"
        runtime_dir = paths.REPO_ROOT / "engines/radiance/configs"
        self.assertTrue(inventory.is_file())
        self.assertIn("[[setting]]", inventory.read_text(encoding="utf-8"))
        self.assertNotIn(inventory, runtime_dir.glob("*.toml"))
        self.assertNotEqual(runtime_dir.resolve(), inventory.parent.resolve())
        self.assertFalse((paths.REPO_ROOT / "config/serving-settings.toml").exists())


if __name__ == "__main__":
    unittest.main()
