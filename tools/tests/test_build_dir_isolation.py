"""bigcherry build's output directory must not collide across --llama-root (HI50)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import __main__ as main_module  # noqa: E402
from bigcherry import paths, recipes  # noqa: E402


class BuildDirIsolationTests(unittest.TestCase):
    def test_default_root_keeps_original_unsuffixed_name(self):
        recipe = recipes.Recipe(
            name="r", ref="pinned", groups=None, states=None,
            follows_pin=True, builds=("b",), platform="p",
        )
        build = recipes.Build(name="b", description="", options={})
        default_dir = main_module._build_dir(recipe, build, paths.llama_root())
        self.assertEqual(default_dir, paths.REPO_ROOT / "build" / "r-b")

    def test_non_default_root_gets_a_distinct_directory(self):
        recipe = recipes.Recipe(
            name="r", ref="pinned", groups=None, states=None,
            follows_pin=True, builds=("b",), platform="p",
        )
        build = recipes.Build(name="b", description="", options={})
        probe_root = Path("/tmp/some-probe-checkout").resolve()
        other_probe_root = Path("/tmp/another-probe-checkout").resolve()
        default_dir = main_module._build_dir(recipe, build, paths.llama_root())
        probe_dir = main_module._build_dir(recipe, build, probe_root)
        other_probe_dir = main_module._build_dir(recipe, build, other_probe_root)
        self.assertNotEqual(default_dir, probe_dir)
        self.assertNotEqual(probe_dir, other_probe_dir)


if __name__ == "__main__":
    unittest.main()
