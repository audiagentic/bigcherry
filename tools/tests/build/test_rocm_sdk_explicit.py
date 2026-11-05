"""BRVP06: the ROCm SDK is an explicit, recorded input of every build (card-free)."""
from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from bigcherry.core import config as campaign_config  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("ci_build_layers", ROOT / ".github/scripts/ci_build_layers.py")
layers = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(layers)

_RECIPES = (ROOT / "config/recipes.toml", *sorted(ROOT.glob("engines/*/recipes.toml")))


class EveryPlatformNamesAStack(unittest.TestCase):
    def test_every_recipe_platform_names_a_declared_stack(self):
        for path in _RECIPES:
            document = tomllib.loads(path.read_text())
            stacks = document.get("stack", {})
            for name, body in document.get("platform", {}).items():
                with self.subTest(platform=name):
                    self.assertIn(body.get("stack"), stacks, f"{path.name} platform.{name}.stack")

    def test_a_platform_without_a_stack_is_a_configuration_error(self):
        text = ('version = 2\nengine = "llamacpp"\npinned = "a"\n[stack.s]\nbackend = "hip"\n'
                '[platform.p]\ntargets = ["gfx1100"]\noptions = {}\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.toml"
            path.write_text(text)
            with self.assertRaisesRegex(campaign_config.ConfigError, r"platform\.p\.stack is required"):
                campaign_config.load(path)

    def test_a_platform_naming_an_unknown_stack_is_a_configuration_error(self):
        text = ('version = 2\nengine = "llamacpp"\npinned = "a"\n[stack.s]\nbackend = "hip"\n'
                '[platform.p]\nstack = "nope"\ntargets = ["gfx1100"]\noptions = {}\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.toml"
            path.write_text(text)
            with self.assertRaisesRegex(campaign_config.ConfigError, "is not a"):
                campaign_config.load(path)

    def test_the_radiance_lane_resolves_its_sdk_from_the_stack(self):
        platform = campaign_config.load(ROOT / "engines/radiance/recipes.toml").platforms["linux-radiance"]
        self.assertEqual((platform.stack, platform.sdk), ("rocm-7-2", "7.2"))
        self.assertIsNotNone(platform.sdk_root)


class CiImageIsBoundToTheStack(unittest.TestCase):
    def _tree(self, directory: str, *, arg: str | None = None, example: str | None = None) -> Path:
        root = Path(directory)
        for relative in ("engines/radiance/recipes.toml", "config/recipes.toml", "config/environment.example.toml",
                         ".github/docker/radiance-build/Dockerfile"):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            text = (ROOT / relative).read_text()
            if relative.endswith("Dockerfile") and arg:
                text = re.sub(r"^ARG ROCM_VERSION=.*$", f"ARG ROCM_VERSION={arg}", text, flags=re.M)
            if relative.endswith("environment.example.toml") and example:
                text = re.sub(r'^"7\.2" = ".*"$', f'"7.2" = "{example}"', text, flags=re.M)
            target.write_text(text)
        return root

    def test_the_dockerfile_arg_agrees_with_the_stack_table(self):
        identity = layers.ci_sdk_identity(ROOT)
        self.assertRegex(identity, r"^stack=rocm-7-2;sdk=7\.2;version=7\.2\.\d+;root=/opt/rocm-7\.2\.\d+$")

    def test_a_dockerfile_on_another_sdk_than_the_stack_names_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(layers.LayerError, "stack rocm-7-2 names SDK 7.2"):
                layers.ci_sdk_identity(self._tree(directory, arg="7.14.0"))

    def test_an_example_host_path_that_is_not_the_installed_version_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(layers.LayerError, "maps SDK 7.2"):
                layers.ci_sdk_identity(self._tree(directory, example="/opt/rocm"))

    def test_the_sdk_identity_is_part_of_the_layer_key(self):
        with mock.patch.object(layers, "ci_sdk_identity", return_value="a"):
            a = layers.input_keys(ROOT)
        with mock.patch.object(layers, "ci_sdk_identity", return_value="b"):
            b = layers.input_keys(ROOT)
        self.assertEqual(set(a), set(b))
        for name in a:
            self.assertNotEqual(a[name], b[name], name)


class NoImplicitSystemRocmInBuildPaths(unittest.TestCase):
    """A hardcoded unversioned /opt/rocm (the symlink to whatever the host installed last) must not appear in a build
    path. Allowed: prose that explains the rule, and fallbacks that the build never reaches because it passes each
    engine.toml build.rocm-paths value explicitly."""

    PATTERN = re.compile(r"/opt/rocm(?![-\w.])")
    ALLOW = {
        # LLVM binutils search for a size-analysis tool; not a build input
        "tools/bigcherry/analysis/binary_size.py",
        # a comment naming the ROCm SMI header the file deliberately does not include
        "engines/llamacpp/overlay/ggml/src/ggml-cuda/hip-autotune-smi.cpp",
        # find_library fallbacks, skipped when the build passes RAD_HIP_RUNTIME / RAD_HSA_RUNTIME (engine.toml rocm-paths)
        "engines/radiance/components/libr3/CMakeLists.txt",
        "engines/radiance/components/moe-remote/CMakeLists.txt",
        # documentation of the rule
        "tools/bigcherry/campaign/build.py",
        "tools/bigcherry/build/compile_check.py",
        "engines/radiance/engine.toml",
        "config/environment.example.toml",
        ".github/docker/radiance-build/Dockerfile",
        "tools/env/bigcherry-env.sh",
        "tools/env/make-rocm-campaign-prefix.sh",
        "tools/env/substitute_paths.py",
    }

    def test_no_build_path_names_the_unversioned_sdk(self):
        roots = ("tools/bigcherry", "config", "engines/radiance", "engines/llamacpp", ".github", "tools/env",
                 "tools/lab/radiance")
        hits = []
        for top in roots:
            for path in sorted((ROOT / top).rglob("*")):
                relative = path.relative_to(ROOT).as_posix()
                if (not path.is_file() or path.suffix in {".patch", ".md", ".hip", ".pyc"} or relative in self.ALLOW
                        or "/patches/" in relative or "/tests/" in relative or "__pycache__" in relative):
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                hits += [f"{relative}:{n}" for n, line in enumerate(text.splitlines(), 1) if self.PATTERN.search(line)]
        self.assertEqual(hits, [], "unversioned /opt/rocm in a build path: name a stack instead")


if __name__ == "__main__":
    unittest.main()
