"""QFP23: runtime profile files - grammar, flattening and validation (tools/bigcherry/patch/runtime_profiles.py)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patch import runtime_profiles as rp  # noqa: E402

_GOOD = """
# comment
[base]
description = generic
X_ONE = 1

[model]
description = a model profile
arch = qwen4exp
@base
Y_TWO = a,b
"""


class RuntimeProfiles(unittest.TestCase):
    def _dir(self, td, **files):
        d = Path(td)
        for name, text in files.items():
            (d / f"{name}.ini").write_text(text, encoding="utf-8")
        return d

    def test_parse_and_flatten(self):
        with tempfile.TemporaryDirectory() as td:
            d = self._dir(td, a=_GOOD)
            profiles = rp.load(d)
            self.assertEqual(profiles["model"].arch, ["qwen4exp"])
            self.assertEqual(rp.flatten(profiles, "model"), {"X_ONE": "1", "Y_TWO": "a,b"})
            self.assertEqual(rp.check(d, documented_flags={"X_ONE", "Y_TWO"}), [])

    def test_undocumented_flag_reported(self):
        with tempfile.TemporaryDirectory() as td:
            d = self._dir(td, a=_GOOD)
            problems = rp.check(d, documented_flags={"X_ONE"})
            self.assertTrue(any("Y_TWO" in p and "ENV_DOCS" in p for p in problems), problems)

    def test_rejects(self):
        bad = {
            "duplicate profile across files": ({"a": "[p]\ndescription = d\n", "b": "[p]\ndescription = d\n"}),
            "unknown include": ({"a": "[p]\ndescription = d\n@q\n"}),
            "cycle": ({"a": "[p]\ndescription = d\n@q\n[q]\ndescription = d\n@p\n"}),
            "conflict": ({"a": "[p]\ndescription = d\nX = 1\n[q]\ndescription = d\nX = 2\n"
                               "[r]\ndescription = d\n@p\n@q\n"}),
            "duplicate key": ({"a": "[p]\ndescription = d\nX = 1\nX = 1\n"}),
            "bad name": ({"a": "[Bad]\ndescription = d\n"}),
            "bad value": ({"a": "[p]\ndescription = d\nX = a b\n"}),
            "entry outside section": ({"a": "X = 1\n"}),
            "unknown key": ({"a": "[p]\ndescription = d\nlower = 1\n"}),
        }
        for label, files in bad.items():
            with tempfile.TemporaryDirectory() as td:
                d = self._dir(td, **files)
                self.assertTrue(rp.check(d, documented_flags={"X"}), label)

    def test_missing_description_reported(self):
        with tempfile.TemporaryDirectory() as td:
            d = self._dir(td, a="[p]\nX = 1\n")
            self.assertTrue(any("no description" in p for p in rp.check(d, documented_flags={"X"})))

    def test_repository_profiles_clean(self):
        self.assertEqual(rp.check(documented_flags=rp.documented_flags()), [])


if __name__ == "__main__":
    unittest.main()
