"""Every radiance source that carries patches is production plus its own additions, never a different base.

There is no separate lab source: the lab runs the production build (plus experiments, which are not sources), so
source.radiance's patch sets must be a subset of every other patched radiance source's. Native (unpatched) sources are
exempt."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

RECIPES = Path(__file__).resolve().parents[3] / "engines" / "radiance" / "recipes.toml"


class RadianceSourcesShareProduction(unittest.TestCase):
    def test_patched_sources_include_every_production_patch_set(self) -> None:
        sources = tomllib.loads(RECIPES.read_text(encoding="utf-8"))["source"]
        production = set(sources["radiance"]["patch-sets"])
        self.assertTrue(production, "source.radiance carries patch sets")
        for name, source in sources.items():
            sets = set(source["patch-sets"])
            if name == "radiance-native" or name == "radiance" or not sets:
                continue
            self.assertLessEqual(production, sets, f"source.{name} lacks production patch sets {sorted(production - sets)}")

    def test_there_is_no_separate_lab_source(self) -> None:
        sources = tomllib.loads(RECIPES.read_text(encoding="utf-8"))["source"]
        self.assertNotIn("radiance-lab", sources)


if __name__ == "__main__":
    unittest.main()
