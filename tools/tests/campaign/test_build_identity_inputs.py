"""A build's identity (build_plan_id) must cover every file its components read.

Components are hashed by content (component.content_digest), but a component's CMake can also read a file from OUTSIDE its own
directory: radiance's moe-remote reads archs.toml to choose its GPU targets, and serving embeds archs.toml in bc-serve (layouts are
run-time data, never compiled in). If such a file changed while the plan id did not, two different bc-serve binaries would land in one
builds/<slice>/<plan id> and the second would be silently "already built" (and a release bundle named by that id would collide
with the earlier one). engine.toml [build] repo-inputs lists them, and the lane puts their bytes in the plan.
"""
from __future__ import annotations

import fnmatch
import posixpath
import re
import tempfile
import unittest
from pathlib import Path

from bigcherry.core import engines, paths

ROOT = paths.REPO_ROOT
RADIANCE = engines.load(ROOT, engines.RADIANCE)

#: out-of-tree paths a component may name without declaring them: the manifest contract directory is passed to the component
#: TESTS only (BC_CONTRACT_DIR / BC_CONTRACT_DOCS_DIR), and nothing built into a runtime reads it
# Read by tests only: the v2 contract docs, and the example layouts (run-time data the tests resolve; no binary embeds them).
TEST_ONLY = ("docs/design/serving-manifest-v2", "engines/radiance/layouts")
_PATH = re.compile(r"\$\{CMAKE_(?:CURRENT_SOURCE_DIR|CURRENT_LIST_DIR)\}/((?:\.\./)+[^\s\"')]*)")


def _out_of_tree_reads() -> list[tuple[str, str]]:
    """(component cmake file, repository-relative path) for every ../ path a component's CMake names."""
    found = []
    for cmake in sorted((ROOT / "engines" / "radiance" / "components").glob("*/*")):
        if cmake.name != "CMakeLists.txt" and cmake.suffix != ".cmake":
            continue
        for line in cmake.read_text(encoding="utf-8").splitlines():
            code = line.split("#", 1)[0]
            for relative in _PATH.findall(code):
                target = posixpath.normpath(posixpath.join(cmake.parent.relative_to(ROOT).as_posix(), relative))
                found.append((cmake.relative_to(ROOT).as_posix(), target))
    return found


class BuildIdentityInputsTests(unittest.TestCase):
    def test_radiance_declares_the_files_its_components_read_outside_their_directories(self) -> None:
        self.assertEqual(set(RADIANCE.repo_inputs), {"engines/radiance/archs.toml"})

    def test_every_out_of_tree_read_is_a_component_a_declared_input_or_test_only(self) -> None:
        components = {f"engines/radiance/components/{name}" for name in (p.name for p in (ROOT / "engines/radiance/components").iterdir() if p.is_dir())}
        undeclared = []
        for cmake, target in _out_of_tree_reads():
            if any(target == c or target.startswith(c + "/") for c in components):
                continue                                    # another component: its content is in the plan on its own
            if any(target == t or target.startswith(t + "/") for t in TEST_ONLY):
                continue
            pattern = re.sub(r"\$\{[^}]*\}", "*", target)   # e.g. engines/radiance/${bc_data}.toml in a foreach
            if any(fnmatch.fnmatch(declared, pattern) for declared in RADIANCE.repo_inputs):
                continue
            undeclared.append(f"{cmake} reads {target}")
        self.assertEqual(undeclared, [],
                         "a component reads a file outside its directory that engine.toml build.repo-inputs does not declare: "
                         "the build plan id would not change when it does")

    def test_the_scan_finds_the_reads_it_is_meant_to_guard(self) -> None:
        targets = {t for _, t in _out_of_tree_reads()}
        self.assertIn("engines/radiance/archs.toml", targets)
        # the layouts are read by tests only (TEST_ONLY), never by a build: they are not a build input
        self.assertTrue(all(t.startswith("engines/radiance/layouts") for t in targets if "layouts" in t), targets)

    def test_a_declared_inputs_bytes_change_the_identity_and_nothing_else_does(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "engines/radiance").mkdir(parents=True)
            (root / "engines/radiance/archs.toml").write_text("a = 1\n")
            before = RADIANCE.repo_input_hashes(root)
            self.assertEqual(before, RADIANCE.repo_input_hashes(root))
            (root / "engines/radiance/layouts").mkdir()
            (root / "engines/radiance/layouts/x.toml").write_text("a = 2\n")
            self.assertEqual(before, RADIANCE.repo_input_hashes(root), "a layout file is not a build input")
            (root / "engines/radiance/archs.toml").write_text("a = 2\n")
            self.assertNotEqual(before, RADIANCE.repo_input_hashes(root))
            (root / "engines/radiance/archs.toml").unlink()
            with self.assertRaises(engines.EngineError):
                RADIANCE.repo_input_hashes(root)

    def test_the_lane_puts_the_declared_inputs_in_the_build_plan(self) -> None:
        # planning a lane needs the upstream source materialised, so the wiring is pinned at its text: the BuildPlan's input_hashes
        # are built from component digests AND the engine's repo inputs
        text = (ROOT / "tools/bigcherry/campaign/lane.py").read_text(encoding="utf-8")
        plan = text[text.index("input_hashes=tuple("):][:1800]
        self.assertIn("component.content_digest()", plan)
        self.assertIn("layout.repo_input_hashes(paths.REPO_ROOT)", plan)

    def test_an_engine_without_repo_inputs_adds_nothing(self) -> None:
        llama = engines.load(ROOT, engines.LLAMACPP)
        self.assertEqual(llama.repo_inputs, ())
        self.assertEqual(llama.repo_input_hashes(ROOT), ())


if __name__ == "__main__":
    unittest.main()
