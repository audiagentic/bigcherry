"""tools.bigcherry.core.process -- subprocess returncode decoding."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core.process import describe_returncode  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
