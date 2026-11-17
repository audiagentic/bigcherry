"""Run a real pytest subprocess and force its executable-temp fallback.

This confirms the pytest lifecycle hook runs after both successful and failed
sessions, not just when called directly with a fake Config.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
TOOLS = TESTS.parent


class RealPytestLifecycleTests(unittest.TestCase):
    def test_success_and_failure_leave_no_generated_scratch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "work"
            workspace.mkdir()
            plugin = root / "forced_exec_temp.py"
            plugin.write_text(
                "from pathlib import Path\n"
                "import tempfile\n"
                "from bigcherry.release import pin_bump\n"
                "def forced(parent):\n"
                "    parent.mkdir(parents=True, exist_ok=True)\n"
                "    return Path(tempfile.mkdtemp(prefix='pin-bump-tests-', dir=parent))\n"
                "pin_bump.exec_capable_tempdir = forced\n"
            )
            suite = root / "suite"
            suite.mkdir()
            test_file = suite / "test_result.py"
            env = os.environ.copy()
            env["BC_WORK_ROOT"] = str(workspace)
            env["PYTHONPATH"] = os.pathsep.join((str(root), str(TESTS), str(TOOLS)))
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            for passes in (True, False):
                with self.subTest(passes=passes):
                    test_file.write_text(f"def test_case():\n    assert {passes!r}\n")
                    result = subprocess.run(
                        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                         "-p", "forced_exec_temp", "-p", "conftest", str(test_file)],
                        env=env, cwd=root, capture_output=True, text=True, timeout=30, check=False,
                    )
                    self.assertEqual(result.returncode, 0 if passes else 1,
                                     result.stdout + result.stderr)
                    self.assertFalse(list((workspace / "test-tmp").iterdir()),
                                     "a real pytest session must not leave fallback scratch")


if __name__ == "__main__":
    unittest.main()
