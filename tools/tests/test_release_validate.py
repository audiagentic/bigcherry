from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bigcherry import release_validate  # noqa: E402


class SafeNameTests(unittest.TestCase):
    def test_safe_name_cannot_escape_staging_root(self):
        self.assertEqual(release_validate.safe_name("../../etc/passwd"), "etc-passwd")
        self.assertEqual(release_validate.safe_name("b10362"), "b10362")
        self.assertEqual(release_validate.safe_name("..."), "upstream")


class ProbeTests(unittest.TestCase):
    def test_run_already_exists_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            (staging / "dup").mkdir()
            with self.assertRaises(FileExistsError):
                release_validate.probe("dup", staging, "master", "bigcherry")

    def test_pull_failure_short_circuits_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            with mock.patch.object(release_validate, "_run_logged", side_effect=[False]) as run:
                code, path = release_validate.probe("r1", staging, "master", "bigcherry")
            self.assertEqual(code, 1)
            self.assertEqual(run.call_count, 1)
            record = path.read_text(encoding="utf-8")
            self.assertIn('"outcome": "pull-failed"', record)

    def test_build_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            with mock.patch.object(release_validate, "_run_logged", side_effect=[True, False]):
                code, path = release_validate.probe("r2", staging, "master", "bigcherry")
            self.assertEqual(code, 1)
            record = path.read_text(encoding="utf-8")
            self.assertIn('"outcome": "patch-drift-or-build-failed"', record)

    def test_clean_probe_reports_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            with mock.patch.object(release_validate, "_run_logged", side_effect=[True, True]) as run:
                code, path = release_validate.probe("r3", staging, "master", "bigcherry")
            self.assertEqual(code, 0)
            pull_command = run.call_args_list[0].args[0]
            build_command = run.call_args_list[1].args[0]
            self.assertLess(pull_command.index("--llama-root"), pull_command.index("pull"))
            self.assertLess(build_command.index("--llama-root"), build_command.index("build"))
            record = path.read_text(encoding="utf-8")
            self.assertIn('"outcome": "compatible"', record)

    def test_inventory_is_forwarded_to_isolated_build(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            inventory = staging / "inventory.json"
            inventory.write_text("{}", encoding="utf-8")
            with mock.patch.object(release_validate, "_run_logged", side_effect=[True, True]) as run:
                code, _ = release_validate.probe("r4", staging, "master", "workstation", inventory)
            self.assertEqual(code, 0)
            build_command = run.call_args_list[1].args[0]
            self.assertEqual(build_command[-2:], ["--inventory", str(inventory)])


if __name__ == "__main__":
    unittest.main()
