"""v2 campaign config parsing (RE14): platform toolchain fields."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import config as campaign_config  # noqa: E402


_DOC = """
version = 2
pinned = "abc123"

[platform.with-toolchain]
targets = ["gfx1100"]
options = {}
c-compiler = "/opt/rocm/llvm/bin/clang"
cxx-compiler = "/opt/rocm/llvm/bin/clang++"

[platform.without-toolchain]
targets = ["gfx1100"]
options = {}
"""


class PlatformToolchainTests(unittest.TestCase):
    """A prior version of config.Platform silently dropped c-compiler/
    cxx-compiler from [platform.*] bodies: the dataclass had no fields for
    them and no unknown-key check caught the loss, so a campaign path
    reading only config.Platform could never reproduce the legacy build's
    toolchain -- it would silently fall back to $PATH.
    """

    def setUp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.toml"
            path.write_text(_DOC, encoding="utf-8")
            self.cfg = campaign_config.load(path)

    def test_compiler_fields_are_preserved(self):
        platform = self.cfg.platforms["with-toolchain"]
        self.assertEqual(platform.c_compiler, "/opt/rocm/llvm/bin/clang")
        self.assertEqual(platform.cxx_compiler, "/opt/rocm/llvm/bin/clang++")

    def test_compiler_fields_default_to_none(self):
        platform = self.cfg.platforms["without-toolchain"]
        self.assertIsNone(platform.c_compiler)
        self.assertIsNone(platform.cxx_compiler)

    def test_non_string_compiler_field_rejected(self):
        doc = _DOC + "\n[platform.bad]\ntargets = [\"gfx1100\"]\nc-compiler = 5\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.toml"
            path.write_text(doc, encoding="utf-8")
            with self.assertRaises(campaign_config.ConfigError):
                campaign_config.load(path)


if __name__ == "__main__":
    unittest.main()
