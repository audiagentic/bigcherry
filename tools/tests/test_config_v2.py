"""Strict v2 configuration tests for the campaign migration."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import config  # noqa: E402


def _write(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8")
    with handle:
        handle.write(text)
    return Path(handle.name)


class ConfigV2Tests(unittest.TestCase):
    def test_v1_is_rejected_instead_of_silently_reinterpreted(self):
        path = _write('pinned = "b10362"\n')
        with self.assertRaisesRegex(config.ConfigError, "version = 2"):
            config.load(path)

    def test_exact_sources_and_build_needs_parse(self):
        path = _write(
            """
version = 2
pinned = "b10362"
[patch-set.framework]
patches = ["0100_cmake_options"]
required-state = "validated"
[source.llama-native]
ref = "pinned"
overlay = false
patch-sets = []
[source.bigcherry-native]
ref = "pinned"
overlay = true
patch-sets = ["framework"]
[build.tune]
options = { GGML_HIP_AUTOTUNE = "ON" }
variant-set = "workload-max"
needs = ["inventory", "promoted-winners"]
[platform.brutus]
targets = ["gfx1100", "gfx1201"]
options = { GGML_HIP = "ON" }
"""
        )
        loaded = config.load(path)
        self.assertFalse(loaded.sources["llama-native"].overlay)
        self.assertEqual(loaded.sources["bigcherry-native"].patch_sets, ("framework",))
        self.assertEqual(
            loaded.builds["tune"].needs,
            frozenset({"inventory", "promoted-winners"}),
        )

    def test_unknown_patch_set_fails(self):
        path = _write(
            """
version = 2
pinned = "b10362"
[source.bad]
ref = "pinned"
overlay = false
patch-sets = ["missing"]
"""
        )
        with self.assertRaisesRegex(config.ConfigError, "unknown patch set"):
            config.load(path)

    def test_wildcard_source_selectors_are_not_v2(self):
        path = _write(
            """
version = 2
pinned = "b10362"
[source.bad]
ref = "pinned"
overlay = true
patch-sets = []
states = ["validated"]
"""
        )
        with self.assertRaisesRegex(config.ConfigError, "exact patch-sets"):
            config.load(path)

    def test_unknown_top_level_field_fails(self):
        path = _write('version = 2\npinned = "b10362"\nfuture = true\n')
        with self.assertRaisesRegex(config.ConfigError, "unknown top-level"):
            config.load(path)

    def test_shipped_v2_preserves_legacy_execution_properties(self):
        loaded = config.load(Path(__file__).resolve().parents[2] / "recipes.toml")
        self.assertEqual(loaded.pinned, "b10362")
        self.assertEqual(loaded.sources["llama-native"].patch_sets, ())
        self.assertEqual(loaded.sources["bigcherry-native"].patch_sets, ("framework",))
        self.assertEqual(loaded.sources["bigcherry"].patch_sets, ("framework", "validated-enhancements"))
        self.assertEqual(loaded.builds["control"].options, (("GGML_HIP_AUTOTUNE", "ON"),))
        self.assertEqual(loaded.builds["tune"].needs, frozenset({"inventory"}))
        self.assertEqual(loaded.builds["replay"].needs, frozenset({"inventory", "promoted-winners"}))


if __name__ == "__main__":
    unittest.main()
