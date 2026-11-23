"""RSP01/S16: legacy source-worktree release installer is absent from the platform CLI.

The native bc-release install/verify/rollback contract is exercised by
bc_release_install_test (component CTest), including tag binding and rollback.
This offline test prevents unintentional resurrection of the Python path.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest

from bigcherry.cli.main import build_parser


class RetiredReleaseInstallTests(unittest.TestCase):
    def test_python_release_installer_is_not_importable(self):
        self.assertIsNone(importlib.util.find_spec("bigcherry.release.install"))

    def test_platform_cli_has_no_legacy_release_install_command(self):
        parser = build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                parser.parse_args(["--engine", "radiance", "release-install", "latest"])
        self.assertEqual(result.exception.code, 2)

    def test_release_packaging_remains_platform_side(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                build_parser().parse_args(["--engine", "radiance", "release-bundle", "--help"])
        self.assertEqual(result.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
