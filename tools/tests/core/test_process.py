"""tools.bigcherry.core.process -- subprocess returncode decoding."""

from __future__ import annotations

import sys
import io
import subprocess
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core.process import describe_returncode, run_checked  # noqa: E402


class DescribeReturncodeTests(unittest.TestCase):
    def test_plain_unix_exit_code_is_unchanged(self):
        self.assertEqual(describe_returncode(1), "1")
        self.assertEqual(describe_returncode(0), "0")
        self.assertEqual(describe_returncode(139), "139")  # SIGSEGV-shaped, not NTSTATUS

    def test_dll_not_found_from_unsigned_form(self):
        # The real b10705 incident: subprocess reported this exact value.
        rendered = describe_returncode(3221225781)
        self.assertIn("0xC0000135", rendered)
        self.assertIn("STATUS_DLL_NOT_FOUND", rendered)
        self.assertIn("PATH", rendered)

    def test_dll_not_found_from_signed_form(self):
        rendered = describe_returncode(-1073741515)
        self.assertIn("0xC0000135", rendered)
        self.assertIn("STATUS_DLL_NOT_FOUND", rendered)

    def test_access_violation_has_no_fabricated_hint(self):
        rendered = describe_returncode(0xC0000005)
        self.assertIn("STATUS_ACCESS_VIOLATION", rendered)
        self.assertNotIn("--", rendered)  # no hint suffix for a generic crash

    def test_unrecognized_ntstatus_shaped_value_gets_hex_but_no_fabricated_name(self):
        rendered = describe_returncode(0xC0001234)
        self.assertIn("0xC0001234", rendered)
        self.assertNotIn("STATUS_", rendered)


class CheckedRunnerTests(unittest.TestCase):
    def test_preserves_checked_subprocess_contract(self):
        # every command but a cmake build: unchanged arguments, cwd and check=True
        cwd = Path("/build")
        argv = ["cmake", "--install", "out"]
        with mock.patch("bigcherry.core.process.subprocess.run") as run:
            self.assertIsNone(run_checked(argv, cwd))
        run.assert_called_once_with(argv, cwd=cwd, check=True)

    def test_a_cmake_build_keeps_its_arguments_and_cwd_and_checks_the_exit(self):
        # a build is streamed (so a compiler crash can be recognised); its arguments and cwd are unchanged
        cwd = Path("/build")
        argv = ["cmake", "--build", "out", "-j"]
        with mock.patch("bigcherry.core.process.subprocess.Popen") as popen:
            popen.return_value.stdout = io.BytesIO(b"[1/1] done\n")
            popen.return_value.wait.return_value = 0
            self.assertIsNone(run_checked(argv, cwd))
        self.assertEqual(argv, popen.call_args.args[0])
        self.assertEqual(cwd, popen.call_args.kwargs["cwd"])
        popen.return_value.wait.return_value = 2
        popen.return_value.stdout = io.BytesIO(b"error: expected ;\n")
        with mock.patch("bigcherry.core.process.subprocess.Popen", popen), self.assertRaises(subprocess.CalledProcessError):
            run_checked(argv, cwd)


if __name__ == "__main__":
    unittest.main()
