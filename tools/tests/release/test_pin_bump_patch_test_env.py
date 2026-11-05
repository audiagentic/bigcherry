"""pin-bump runs the patch mechanics tests in a temp directory that allows execution."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry.release import pin_bump


class ExecCapableTempdirTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_a_system_tmp_that_allows_execution_is_used_as_it_is(self):
        with patch.object(pin_bump.tempfile, "gettempdir", return_value=str(self.root)):
            self.assertIsNone(pin_bump.exec_capable_tempdir(self.root / "work-tmp"))
        self.assertFalse((self.root / "work-tmp").exists(), "nothing is created when nothing is needed")

    def test_a_noexec_tmp_gets_a_scratch_directory_under_the_work_root(self):
        denied = subprocess.CompletedProcess([], 126, b"", b"permission denied")
        with patch.object(pin_bump.tempfile, "gettempdir", return_value=str(self.root)), \
                patch.object(pin_bump.subprocess, "run", return_value=denied):
            scratch = pin_bump.exec_capable_tempdir(self.root / "work-tmp")
        self.assertIsNotNone(scratch)
        self.assertEqual(scratch.parent, self.root / "work-tmp")
        self.assertTrue(scratch.is_dir())
        self.assertTrue(scratch.name.startswith("pin-bump-tests-"))

    def test_a_tmp_that_cannot_be_probed_falls_back_too(self):
        with patch.object(pin_bump.tempfile, "gettempdir", return_value=str(self.root / "missing")):
            scratch = pin_bump.exec_capable_tempdir(self.root / "work-tmp")
        self.assertIsNotNone(scratch)
        self.assertEqual(scratch.parent, self.root / "work-tmp")


class PatchTestRunTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        (self.root / "tools" / "tests" / "patch").mkdir(parents=True)

    def run_with(self, scratch):
        seen = {}

        def fake_run(argv, **kwargs):
            seen["env"] = kwargs["env"]
            seen["scratch_existed"] = scratch is None or scratch.is_dir()
            return subprocess.CompletedProcess(argv, 0, "", "OK")

        context = type("Context", (), {"work_root": self.root / "work"})()
        with patch.object(pin_bump, "select_patch_test_modules", return_value=["test_x"]), \
                patch.object(pin_bump.ProjectContext, "resolve", return_value=context), \
                patch.object(pin_bump, "exec_capable_tempdir", return_value=scratch), \
                patch.object(pin_bump.subprocess, "run", side_effect=fake_run):
            failing = pin_bump.run_patch_tests(self.root, "radiance")
        return failing, seen

    def test_the_tests_get_the_scratch_directory_and_it_is_removed_afterwards(self):
        scratch = self.root / "scratch"
        scratch.mkdir()
        failing, seen = self.run_with(scratch)
        self.assertEqual(failing, [])
        self.assertEqual(seen["env"]["TMPDIR"], str(scratch))
        self.assertTrue(seen["scratch_existed"])
        self.assertFalse(scratch.exists(), "the scratch directory does not outlive the run")

    def test_with_a_usable_system_tmp_the_environment_is_left_alone(self):
        failing, seen = self.run_with(None)
        self.assertEqual(failing, [])
        self.assertEqual(seen["env"].get("TMPDIR"), os.environ.get("TMPDIR"))


if __name__ == "__main__":
    unittest.main()
