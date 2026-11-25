"""A content-addressed build dir is shared by every checkout with identical sources, but a component's CMake cache pins the
source path it was first configured from and CMake refuses another. The runtime bundle must configure such a component in
its own tree beside the existing one, and never touch (let alone wipe) the tree another run may be building in."""

import hashlib
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bigcherry.campaign import runtime_bundle as rb
from bigcherry.core import components, config as campaign_config, paths


def _cache(tree: Path, home: Path) -> Path:
    tree.mkdir(parents=True, exist_ok=True)
    f = tree / "CMakeCache.txt"
    f.write_text(f"CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_HOME_DIRECTORY:INTERNAL={home}\n", encoding="utf-8")
    return f


class ComponentTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.build = Path(self._tmp.name) / "build"
        self.src_a = Path(self._tmp.name) / "checkout-a" / "components" / "x"
        self.src_b = Path(self._tmp.name) / "checkout-b" / "components" / "x"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_fresh_tree_and_a_matching_cache_use_the_plain_directory(self) -> None:
        plain = self.build / "components" / "x"
        self.assertEqual(plain, rb.component_build_dir(self.build, "x", self.src_a))
        _cache(plain, self.src_a)
        self.assertEqual(plain, rb.component_build_dir(self.build, "x", self.src_a))

    def test_a_cache_from_another_source_path_gets_a_separate_stable_directory(self) -> None:
        plain = self.build / "components" / "x"
        cache = _cache(plain, self.src_a)
        before = cache.read_bytes()
        other = rb.component_build_dir(self.build, "x", self.src_b)
        self.assertNotEqual(plain, other)
        self.assertEqual(self.build / "components", other.parent)
        self.assertEqual(other, rb.component_build_dir(self.build, "x", self.src_b))  # stable
        self.assertEqual(before, cache.read_bytes())  # the other tree is untouched
        # once configured there, its own cache matches and it is found again
        _cache(other, self.src_b)
        self.assertEqual(other, rb.component_build_dir(self.build, "x", self.src_b))
        # a third checkout gets a third tree
        third = rb.component_build_dir(self.build, "x", self.src_b.parent.parent / "checkout-c" / "components" / "x")
        self.assertNotIn(third, (plain, other))

    def test_the_directory_without_a_source_keeps_its_old_meaning(self) -> None:
        self.assertEqual(self.build / "components" / "x", rb.component_build_dir(self.build, "x"))


class BuildRuntimeTests(unittest.TestCase):
    def test_a_component_cached_for_another_checkout_is_built_in_its_own_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build, root = Path(tmp) / "build", Path(tmp) / "src"
            _cache(build, root).write_text("BC_SERVING_CONTRACT_VERSION:STRING=3\n", encoding="utf-8")
            declared = components.load_all(paths.REPO_ROOT, "radiance")
            name, comp = sorted(declared.items())[0]
            old = build / "components" / name
            cache = _cache(old, Path(tmp) / "other-checkout" / name)
            before = cache.read_bytes()
            marker = old / "keep.o"
            marker.write_text("object", encoding="utf-8")
            configured: list[tuple[str, str]] = []

            def runner(cmd: list[str], cwd: Path) -> None:
                if cmd[:2] == ["cmake", "-S"]:
                    src, tree = cmd[2], Path(cmd[cmd.index("-B") + 1])
                    configured.append((src, str(tree)))
                    native_consumers = {str(declared[name].root) for name in ("serving", "moe-remote")}
                    self.assertEqual(src in native_consumers, "-DBC_SERVING_CONTRACT_VERSION=3" in cmd)
                    if src == str(declared["libr3"].root):
                        self.assertIn("-DR3_ONLY=", cmd)
                    _cache(tree, Path(src))
                    for output in comp.outputs if src == str(comp.root) else ():
                        (tree / Path(output).name).write_text("x", encoding="utf-8")
                    for other_name, other in declared.items():
                        if other.root == Path(src):
                            for output in other.outputs:
                                (tree / Path(output).name).write_text("x", encoding="utf-8")

            platform = campaign_config.Platform(name="p", targets=(), options=(), stack="s", sdk=None, sdk_root=None)
            with mock.patch.object(rb, "rocm_configure_options", return_value={}):
                rb.build_runtime("radiance", source_root=root, build_dir=build, platform=platform, runner=runner)
            mine = [tree for src, tree in configured if src == str(comp.root)]
            self.assertEqual(1, len(mine))
            self.assertNotEqual(str(old), mine[0])  # configured beside the pinned one
            self.assertEqual(before, cache.read_bytes())  # the shared tree was not reconfigured
            self.assertTrue(marker.exists())

    def test_runtime_outputs_are_unambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, build = Path(tmp) / "src", Path(tmp) / "build"
            declared = components.load_all(paths.REPO_ROOT, "radiance")
            name, comp = sorted(declared.items())[0]
            platform = campaign_config.Platform(name="p", targets=(), options=(), stack="s", sdk=None, sdk_root=None)

            def runner(cmd: list[str], cwd: Path) -> None:
                if cmd[:2] == ["cmake", "-S"]:
                    tree = Path(cmd[cmd.index("-B") + 1])
                    _cache(tree, Path(cmd[2]))
                    for output in comp.outputs:
                        (tree / Path(output).name).write_text("current")
                    stale = tree / "stale"
                    stale.mkdir()
                    (stale / Path(comp.outputs[0]).name).write_text("old")

            with mock.patch.object(rb.components, "load_all", return_value={name: comp}), \
                    mock.patch.object(rb, "rocm_configure_options", return_value={}):
                with self.assertRaisesRegex(rb.RuntimeBundleError, "ambiguous outputs"):
                    rb.build_runtime("radiance", source_root=root, build_dir=build, platform=platform, runner=runner)

            duplicate = replace(comp, name="duplicate")
            with mock.patch.object(rb.components, "load_all", return_value={name: comp, "duplicate": duplicate}):
                with self.assertRaisesRegex(rb.RuntimeBundleError, "both declare"):
                    rb.build_runtime("radiance", source_root=root, build_dir=build, platform=platform, runner=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
