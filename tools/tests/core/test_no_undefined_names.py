"""A name used but never defined or imported fails only when its line runs (#1395 removed an import that producer.py
still used, so the patch-validation producer broke at its last step). Check the whole tool tree statically."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2]


@unittest.skipUnless(importlib.util.find_spec("ruff") or importlib.util.find_spec("pyflakes"), "needs ruff or pyflakes")
class UndefinedNameTests(unittest.TestCase):
    def test_tool_tree_has_no_undefined_names(self) -> None:
        if importlib.util.find_spec("ruff"):
            cmd = [sys.executable, "-m", "ruff", "check", str(TOOLS / "bigcherry"), "--select", "F821", "--no-cache"]
        else:
            cmd = [sys.executable, "-m", "pyflakes", str(TOOLS / "bigcherry")]
        done = subprocess.run(cmd, capture_output=True, text=True)
        out = done.stdout + done.stderr
        if "pyflakes" in cmd[2]:
            out = "\n".join(l for l in out.splitlines() if "undefined name" in l)
            self.assertEqual(out, "")
        else:
            self.assertEqual(done.returncode, 0, out)


if __name__ == "__main__":
    unittest.main()
