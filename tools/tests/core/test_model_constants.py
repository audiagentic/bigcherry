"""RCS12 model-constant execution guard: the real tree is clean, and the checker fails on each kind of mistake."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bigcherry import model_constants as guard

ROOT = Path(__file__).resolve().parents[3]

CONFIG = """
[scope]
roots = ["engines/radiance/components"]
extensions = [".h", ".cpp"]
literals = [2560, 640]
symbols = ["kHidden"]
test_patterns = ["_test\\\\.cpp$"]
"""


def entry(path, anchor, category, reason="because", owner=None):
    lines = ["", "[[allow]]", f'path = "{path}"', f'anchor = "{anchor}"', f'category = "{category}"', f'reason = "{reason}"']
    if owner:
        lines.append(f'owner = "{owner}"')
    return "\n".join(lines)


class RealTree(unittest.TestCase):
    def test_every_model_shaped_literal_is_classified_and_owned(self):
        self.assertEqual(guard.check(ROOT), [])

    def test_no_execution_path_entry_is_unowned(self):
        config = guard.load(ROOT)
        owned = [e for e in config["allow"] if e["category"] in guard.OWNED]
        self.assertTrue(owned, "the audit's execution-path findings are listed, not hidden")
        self.assertTrue(all(e.get("owner") for e in owned))


class Synthetic(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "docs/planning/active/build-x").mkdir(parents=True)
        (self.root / "docs/planning/active/build-x/RCS99.md").write_text("open item")
        (self.root / "engines/radiance/components/c").mkdir(parents=True)

    def source(self, name, text):
        (self.root / "engines/radiance/components/c" / name).write_text(text)

    def config(self, *entries):
        (self.root / "engines/radiance").mkdir(parents=True, exist_ok=True)
        (self.root / guard.CONFIG).write_text(CONFIG + "\n".join(entries) + "\n")

    def problems(self):
        return guard.check(self.root)

    def test_clean_when_everything_is_listed(self):
        self.source("a.h", "constexpr int kW = 2560;\n")
        self.config(entry("engines/radiance/components/c/a.h", "kW = 2560", "C", owner="RCS99"))
        self.assertEqual(self.problems(), [])

    def test_an_unlisted_literal_fails(self):
        self.source("a.h", "constexpr int kW = 640;\n")
        self.config()
        found = self.problems()
        self.assertEqual(len(found), 1)
        self.assertIn("not in the allowlist", found[0])

    def test_an_unlisted_symbol_fails(self):
        self.source("a.cpp", "int x = kHidden * 2;\n")
        self.config()
        self.assertEqual(len(self.problems()), 1)

    def test_comments_and_test_files_are_not_scanned(self):
        self.source("a.h", "// 2560 in a comment\n/* 640 */ int x = 1;\n* 2560 doc line\n")
        self.source("a_test.cpp", "int x = 2560;\n")
        self.config()
        self.assertEqual(self.problems(), [])

    def test_a_stale_entry_fails(self):
        self.source("a.h", "int x = 1;\n")
        self.config(entry("engines/radiance/components/c/a.h", "kW = 2560", "B"))
        self.assertIn("stale", self.problems()[0])

    def test_an_execution_path_entry_needs_an_open_owner(self):
        self.source("a.h", "constexpr int kW = 2560;\n")
        for owner in (None, "RCS00"):
            self.config(entry("engines/radiance/components/c/a.h", "kW = 2560", "C", owner=owner))
            self.assertIn("open plan item", self.problems()[0])
        (self.root / "docs/planning/active/build-x/RCS99.md").unlink()
        self.config(entry("engines/radiance/components/c/a.h", "kW = 2560", "D", owner="RCS99"))
        self.assertIn("open plan item", self.problems()[0], "completing the owner invalidates the entry")

    def test_bounded_and_diagnostic_entries_need_no_owner(self):
        self.source("a.h", "constexpr int kW = 2560;\nint y = 640;\n")
        self.config(entry("engines/radiance/components/c/a.h", "kW = 2560", "A"),
                    entry("engines/radiance/components/c/a.h", "y = 640", "B"))
        self.assertEqual(self.problems(), [])

    def test_unknown_category_and_missing_reason_fail(self):
        self.source("a.h", "int x = 2560;\n")
        self.config(entry("engines/radiance/components/c/a.h", "2560", "Z"))
        self.assertTrue(any("category must be" in p for p in self.problems()))


if __name__ == "__main__":
    unittest.main()
