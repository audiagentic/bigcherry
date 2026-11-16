"""Regression tests for pytest's executable temporary-directory lifecycle.

The tests import the repository conftest under a private name and never create
anything in the real BigCherry work root.
"""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conftest.py"
SPEC = importlib.util.spec_from_file_location("_bigcherry_pytest_temp_plugin", CONFIG_PATH)
assert SPEC is not None and SPEC.loader is not None
plugin = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plugin)


class PytestTempLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.previous_tempdir = tempfile.tempdir
        self.addCleanup(setattr, tempfile, "tempdir", self.previous_tempdir)

    def fallback(self):
        parent = self.root / "test-tmp"
        parent.mkdir(exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="pin-bump-tests-", dir=parent))

    def test_no_work_root_does_not_change_environment(self):
        config = SimpleNamespace()
        with (
            patch.dict(os.environ, {"TMPDIR": "pre-existing"}, clear=True),
            patch.object(plugin, "exec_capable_tempdir") as fallback,
        ):
            plugin.pytest_configure(config)
            plugin.pytest_unconfigure(config)
            fallback.assert_not_called()
            self.assertEqual(os.environ["TMPDIR"], "pre-existing")
            self.assertEqual(tempfile.tempdir, self.previous_tempdir)

    def test_exec_capable_system_temp_does_not_create_or_replace_anything(self):
        config = SimpleNamespace()
        with (
            patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root), "TMPDIR": "pre-existing"}, clear=True),
            patch.object(plugin, "exec_capable_tempdir", return_value=None),
        ):
            plugin.pytest_configure(config)
            plugin.pytest_unconfigure(config)
            self.assertEqual(os.environ["TMPDIR"], "pre-existing")
            self.assertEqual(tempfile.tempdir, self.previous_tempdir)
            self.assertFalse((self.root / "test-tmp").exists())

    def test_cleanup_removes_owned_nested_files_and_restores_existing_settings(self):
        scratch = self.fallback()
        nested = scratch / "some-test" / "nested"
        nested.mkdir(parents=True)
        (nested / "output.bin").write_bytes(b"fixture")
        config = SimpleNamespace()
        with (
            patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root), "TMPDIR": "old-location"}, clear=True),
            patch.object(plugin, "exec_capable_tempdir", return_value=scratch),
        ):
            plugin.pytest_configure(config)
            self.assertEqual(os.environ["TMPDIR"], str(scratch))
            self.assertEqual(tempfile.tempdir, str(scratch))
            marker = json.loads((scratch / ".bigcherry-pytest-session.json").read_text())
            self.assertEqual(marker["pid"], os.getpid())
            self.assertEqual(marker["purpose"], "pytest executable temp fallback")
            self.assertTrue(marker["host"])
            self.assertGreater(marker["started_at_unix"], 0)
            plugin.pytest_unconfigure(config)
            self.assertFalse(scratch.exists())
            self.assertEqual(os.environ["TMPDIR"], "old-location")
            self.assertEqual(tempfile.tempdir, self.previous_tempdir)

    def test_cleanup_restores_absent_tmpdir_without_touching_other_fixtures(self):
        scratch = self.fallback()
        other = self.fallback()
        (other / "do-not-remove").write_text("important")
        config = SimpleNamespace()
        with (
            patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}, clear=True),
            patch.object(plugin, "exec_capable_tempdir", return_value=scratch),
        ):
            plugin.pytest_configure(config)
            plugin.pytest_unconfigure(config)
            self.assertNotIn("TMPDIR", os.environ)
            self.assertFalse(scratch.exists())
            self.assertEqual((other / "do-not-remove").read_text(), "important")

    def test_fixture_symlinks_are_not_followed(self):
        scratch = self.fallback()
        external = self.root / "external"
        external.mkdir()
        (external / "important").write_text("survive")
        (scratch / "outside").symlink_to(external, target_is_directory=True)
        (scratch / "inside").mkdir()
        (scratch / "inside" / "pointer").symlink_to(external / "important")
        config = SimpleNamespace()
        with (
            patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}, clear=True),
            patch.object(plugin, "exec_capable_tempdir", return_value=scratch),
        ):
            plugin.pytest_configure(config)
            plugin.pytest_unconfigure(config)
        self.assertFalse(scratch.exists())
        self.assertEqual((external / "important").read_text(), "survive")

    def test_replaced_directory_is_not_deleted(self):
        scratch = self.fallback()
        replacement = self.root / "replacement"
        replacement.mkdir()
        (replacement / "important").write_text("survive")
        config = SimpleNamespace()
        with (
            patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}, clear=True),
            patch.object(plugin, "exec_capable_tempdir", return_value=scratch),
        ):
            plugin.pytest_configure(config)
            original = scratch.with_name("moved-out-original")
            scratch.rename(original)
            replacement.rename(scratch)
            with self.assertWarnsRegex(RuntimeWarning, "replaced"):
                plugin.pytest_unconfigure(config)
            self.assertEqual((scratch / "important").read_text(), "survive")
            self.assertTrue(original.exists())
            self.assertNotIn("TMPDIR", os.environ)
            self.assertEqual(tempfile.tempdir, self.previous_tempdir)

    def test_nested_mount_check_failure_leaves_owned_scratch_and_restores_settings(self):
        scratch = self.fallback()
        config = SimpleNamespace()
        with (
            patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}, clear=True),
            patch.object(plugin, "exec_capable_tempdir", return_value=scratch),
        ):
            plugin.pytest_configure(config)
            with (
                patch.object(plugin, "_no_mounts_under",
                             side_effect=OSError("unreadable mountinfo")),
                self.assertWarnsRegex(RuntimeWarning, "unreadable mountinfo"),
            ):
                plugin.pytest_unconfigure(config)
            self.assertTrue(scratch.is_dir())
            self.assertNotIn("TMPDIR", os.environ)
            self.assertEqual(tempfile.tempdir, self.previous_tempdir)

    def test_unconfigure_is_idempotent_after_cleanup(self):
        scratch = self.fallback()
        config = SimpleNamespace()
        with (
            patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}, clear=True),
            patch.object(plugin, "exec_capable_tempdir", return_value=scratch),
        ):
            plugin.pytest_configure(config)
            plugin.pytest_unconfigure(config)
            plugin.pytest_unconfigure(config)
        self.assertFalse(scratch.exists())

    def test_does_not_overwrite_environment_changes_made_after_configure(self):
        scratch = self.fallback()
        config = SimpleNamespace()
        with (
            patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root), "TMPDIR": "original"}, clear=True),
            patch.object(plugin, "exec_capable_tempdir", return_value=scratch),
        ):
            plugin.pytest_configure(config)
            os.environ["TMPDIR"] = "changed-by-another-plugin"
            tempfile.tempdir = "also