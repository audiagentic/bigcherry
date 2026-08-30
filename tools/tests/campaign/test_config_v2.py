"""Strict v2 configuration tests for the campaign migration."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import config, paths # noqa: E402


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

    def test_source_backend_defaults_to_hip(self):
        # RE30 phase 3: every source predating the backend field must be
        # byte-for-byte unchanged.
        path = _write(
            """
version = 2
pinned = "b10362"
[source.llama-native]
ref = "pinned"
overlay = false
patch-sets = []
"""
        )
        loaded = config.load(path)
        self.assertEqual(loaded.sources["llama-native"].backend, "hip")

    def test_source_backend_vulkan_parses(self):
        path = _write(
            """
version = 2
pinned = "b10362"
[source.vulkan-stock]
ref = "pinned"
overlay = false
patch-sets = []
backend = "vulkan"
"""
        )
        loaded = config.load(path)
        self.assertEqual(loaded.sources["vulkan-stock"].backend, "vulkan")

    def test_backend_stack_profiles_parse_and_stay_separate_from_platform(self):
        path = _write(
            """
version = 2
pinned = "b10362"
[stack.rocm-10]
backend = "hip"
sdk-root = "/opt/rocm-10"
c-compiler = "/opt/clang/bin/clang"
runtime-library-dirs = ["/opt/rocm-10/lib"]
environment = { HIP_VISIBLE_DEVICES = "0,1" }
required-providers = ["hip", "rocblas"]
[platform.linux]
targets = ["gfx1100"]
options = { GGML_HIP = "ON" }
"""
        )
        loaded = config.load(path)
        stack = loaded.stacks["rocm-10"]
        self.assertIsInstance(stack, config.BackendStack)
        self.assertEqual(stack.backend, "hip")
        self.assertEqual(stack.sdk_root, "/opt/rocm-10")
        self.assertEqual(stack.environment, (("HIP_VISIBLE_DEVICES", "0,1"),))
        self.assertEqual(stack.required_providers, ("hip", "rocblas"))
        self.assertEqual(loaded.platforms["linux"].targets, ("gfx1100",))

    def test_backend_stack_rejects_unknown_backend(self):
        path = _write(
            """
version = 2
pinned = "b10362"
[stack.bad]
backend = "cuda"
"""
        )
        with self.assertRaisesRegex(config.ConfigError, "stack.bad.backend"):
            config.load(path)

    def test_backend_stack_rejects_invalid_environment_values(self):
        path = _write(
            """
version = 2
pinned = "b10362"
[stack.bad]
backend = "hip"
environment = { HIP_VISIBLE_DEVICES = true }
"""
        )
        with self.assertRaisesRegex(config.ConfigError, "must use"):
            config.load(path)

    def test_unknown_source_backend_rejected(self):
        path = _write(
            """
version = 2
pinned = "b10362"
[source.bad]
ref = "pinned"
overlay = false
patch-sets = []
backend = "cuda"
"""
        )
        with self.assertRaisesRegex(config.ConfigError, "backend"):
            config.load(path)

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
        # Asserts the shape survives loading, not a specific pin value --
        # the real recipes.toml's `pinned` moves every time the project
        # rebases to a newer llama.cpp release (see
        # docs/reference/archive/build/PIN_REBASE_REVIEW_B10502.md), which this test
        # must not need editing for.
        loaded = config.load(paths.RECIPES)
        self.assertTrue(loaded.pinned)
        self.assertEqual(loaded.sources["llama-native"].patch_sets, ())
        self.assertEqual(loaded.sources["bigcherry-native"].patch_sets, ("framework",))
        self.assertEqual(loaded.sources["bigcherry"].patch_sets, ("framework", "validated-enhancements"))
        self.assertEqual(loaded.builds["control"].options, (("GGML_HIP_AUTOTUNE", "ON"),))
        self.assertEqual(loaded.builds["tune"].needs, frozenset({"inventory"}))
        self.assertEqual(loaded.builds["replay"].needs, frozenset({"inventory", "promoted-winners"}))


if __name__ == "__main__":
    unittest.main()
