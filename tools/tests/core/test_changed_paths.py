"""The CI path matcher (.github/scripts/changed_paths.py): include globs, `!` excludes, and what a docs-only change does."""
import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "changed_paths", Path(__file__).resolve().parents[3] / ".github" / "scripts" / "changed_paths.py")
changed_paths = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(changed_paths)

RADIANCE = ["engines/radiance/**", "!engines/radiance/**/*.md", "tools/bigcherry/build/**"]


class ChangedPathsTests(unittest.TestCase):
    def test_include_only_behaves_as_before(self):
        self.assertEqual(changed_paths.first_match(["engines/radiance/a.cpp"], ["engines/radiance/**"]), "engines/radiance/a.cpp")
        self.assertIsNone(changed_paths.first_match(["docs/x.md"], ["engines/radiance/**"]))

    def test_markdown_under_the_engine_does_not_trigger_the_engine_build(self):
        for path in ("engines/radiance/README.md", "engines/radiance/components/libr3/README.md",
                     "engines/radiance/patches/rad0002_moe_remote_handoff/SUMMARY.md"):
            self.assertIsNone(changed_paths.first_match([path], RADIANCE), path)

    def test_source_build_files_and_data_still_trigger_it(self):
        for path in ("engines/radiance/components/moe-remote/CMakeLists.txt", "engines/radiance/layouts/flashnext.toml",
                     "engines/radiance/components/libr3/native/r4d_moe.hip.rw",
                     "engines/radiance/patches/rad0002_moe_remote_handoff/patch.py", "tools/bigcherry/build/x.py"):
            self.assertEqual(changed_paths.first_match([path], RADIANCE), path)

    def test_one_code_file_among_docs_triggers_it(self):
        changed = ["engines/radiance/README.md", "docs/planning/x.md", "engines/radiance/archs.toml"]
        self.assertEqual(changed_paths.first_match(changed, RADIANCE), "engines/radiance/archs.toml")

    def test_an_exclude_wins_wherever_it_is_listed(self):
        self.assertIsNone(changed_paths.first_match(["engines/radiance/a.md"], ["!engines/radiance/**/*.md", "engines/radiance/**"]))

    def test_only_excludes_match_nothing(self):
        self.assertIsNone(changed_paths.first_match(["engines/radiance/a.cpp"], ["!engines/radiance/**/*.md"]))

    def test_a_markdown_file_directly_in_the_engine_folder_is_excluded_too(self):
        self.assertIsNone(changed_paths.first_match(["engines/radiance/README.md"], RADIANCE))


if __name__ == "__main__":
    unittest.main()
