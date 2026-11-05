"""pin-bump resume: a run records every input a bare --resume needs, and keeps the vendored tree on its target."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from bigcherry.release import pin_bump


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


class FreshStateTests(unittest.TestCase):
    def state(self, **overrides):
        arguments = dict(
            from_ref="v1.16.0", to_ref="v1.17.0", to_sha="a" * 40, vendor_root=Path("/vendor"),
            selector_kind="source", selector_name="radiance", selector_patch_ids=("0100_x",),
            build_name="stock", platform_name="linux-radiance", cpu_smoke_build=None,
        )
        arguments.update(overrides)
        return pin_bump.fresh_state(**arguments)

    def test_a_fresh_run_records_its_compile_platform_with_the_other_lane_inputs(self):
        state = self.state()
        self.assertEqual(state.compile_platform, "linux-radiance")
        self.assertEqual(state.lane_build_name, "stock")
        self.assertEqual(state.completed_phases, ["preflight"])
        self.assertEqual(state.next_phase, "declare")

    def test_a_bare_resume_after_an_early_stop_reuses_the_recorded_platform(self):
        # The run stops at drift, before the compile phase: --resume with no --platform must still know it.
        self.assertEqual(pin_bump._resume_platform(self.state(), platform_name=None), "linux-radiance")

    def test_a_resume_naming_another_platform_is_still_refused(self):
        with self.assertRaises(pin_bump.PinBumpStop) as stopped:
            pin_bump._resume_platform(self.state(), platform_name="linux-other")
        self.assertEqual(stopped.exception.code, "RESUME_PLATFORM_MISMATCH")

    def test_a_run_without_a_platform_records_none(self):
        self.assertEqual(self.state(platform_name=None, build_name=None).compile_platform, "")


class VendorTreeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        self.vendor = self.root / "vendor"
        self.vendor.mkdir()
        _git(self.vendor, "init", "-q")
        _git(self.vendor, "config", "user.email", "t@example.invalid")
        _git(self.vendor, "config", "user.name", "t")
        (self.vendor / "a.txt").write_text("one\n")
        _git(self.vendor, "add", "a.txt")
        _git(self.vendor, "commit", "-q", "-m", "one")
        self.old = _git(self.vendor, "rev-parse", "HEAD")
        (self.vendor / "a.txt").write_text("two\n")
        _git(self.vendor, "commit", "-q", "-am", "two")
        self.target = _git(self.vendor, "rev-parse", "HEAD")

    def head(self) -> str:
        return _git(self.vendor, "rev-parse", "HEAD")

    def test_a_tree_already_on_the_target_is_left_alone(self):
        self.assertFalse(pin_bump.ensure_vendor_at_target(self.vendor, to_ref="v2", to_sha=self.target))
        self.assertEqual(self.head(), self.target)

    def test_a_tree_put_back_on_the_old_pin_is_moved_to_the_target(self):
        _git(self.vendor, "checkout", "-q", self.old)
        self.assertTrue(pin_bump.ensure_vendor_at_target(self.vendor, to_ref="v2", to_sha=self.target))
        self.assertEqual(self.head(), self.target)
        self.assertEqual((self.vendor / "a.txt").read_text(), "two\n")

    def test_untracked_files_do_not_stop_the_move(self):
        # the vendored tree carries untracked overlay files from applied patches
        (self.vendor / "overlay.h").write_text("x\n")
        _git(self.vendor, "checkout", "-q", self.old)
        self.assertTrue(pin_bump.ensure_vendor_at_target(self.vendor, to_ref="v2", to_sha=self.target))
        self.assertTrue((self.vendor / "overlay.h").is_file())

    def test_a_tracked_change_that_a_checkout_would_lose_is_refused(self):
        _git(self.vendor, "checkout", "-q", self.old)
        (self.vendor / "a.txt").write_text("edited\n")
        with self.assertRaises(pin_bump.PinBumpStop) as stopped:
            pin_bump.ensure_vendor_at_target(self.vendor, to_ref="v2", to_sha=self.target)
        self.assertEqual(stopped.exception.code, "VENDOR_TREE_NOT_AT_TARGET")
        self.assertEqual(self.head(), self.old)
        self.assertEqual((self.vendor / "a.txt").read_text(), "edited\n")

    def test_a_target_the_tree_never_fetched_is_not_a_checkout_error(self):
        self.assertFalse(pin_bump.ensure_vendor_at_target(self.vendor, to_ref="v9", to_sha="f" * 40))
        self.assertEqual(self.head(), self.target)

    def test_a_path_that_is_not_a_git_checkout_is_a_no_op(self):
        plain = self.root / "plain"
        plain.mkdir()
        self.assertFalse(pin_bump.ensure_vendor_at_target(plain, to_ref="v2", to_sha=self.target))


if __name__ == "__main__":
    unittest.main()
