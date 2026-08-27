"""PA16: patch-rebase-check probing, quarantine fixed point, and the
apply --rebase-report --known-good staleness contract.

Real git fixtures throughout (same convention as
test_patch_source_isolation.py / test_re04_materialization_safety.py):
a synthetic upstream repo bumped from an "old" to a "new" revision, and a
synthetic ``patches/`` root pointed to via ``paths.PATCHES``/
``paths.SRC_OVERLAY`` monkeypatching so the real repository's own patch set
is never touched by these tests.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import paths  # noqa: E402
from bigcherry.patch import rebase  # noqa: E402

CLEAN_PATCH = """\
from bigcherry.patcher import Edit, FilePatch
GROUP = 'core'
STATE = 'validated'
PATCHES = [FilePatch(path='target.c', edits=(
    Edit(id='clean', anchor=r'^clean_anchor$', text='\\nclean_insert',
         guard=r'^clean_insert$', rationale='stable clean anchor'),
))]
"""

NOT_APPLICABLE_PATCH = """\
from bigcherry.patcher import Edit, FilePatch
GROUP = 'core'
STATE = 'validated'
PATCHES = [FilePatch(path='target.c', edits=(
    Edit(id='old-shape-only', anchor=r'^seed$', text='\\nold_only_insert',
         guard=r'^old_only_insert$', applies_if=r'^ONLY_OLD$',
         rationale='old shape only'),
))]
"""

FAILED_PATCH = """\
from bigcherry.patcher import Edit, FilePatch
GROUP = 'core'
STATE = 'validated'
PATCHES = [FilePatch(path='target.c', edits=(
    Edit(id='moved-anchor', anchor=r'^old_failure_anchor$', text='\\nfailed_insert',
         guard=r'^failed_insert$', rationale='anchor renamed by upstream'),
))]
"""

BLOCKED_PATCH = """\
from bigcherry.patcher import Edit, FilePatch
GROUP = 'core'
STATE = 'validated'
REQUIRES = ('0300_failed',)
PATCHES = [FilePatch(path='target.c', edits=(
    Edit(id='blocked', anchor=r'^blocked_anchor$', text='\\nblocked_insert',
         guard=r'^blocked_insert$'),
))]
"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_upstream(directory: Path) -> Path:
    repo = directory / "upstream"
    _git(directory, "init", str(repo))
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "target.c").write_text(
        "seed\n"
        "clean_anchor\n"
        "ONLY_OLD\n"
        "old_failure_anchor\n"
        "blocked_anchor\n",
        encoding="utf-8",
    )
    _git(repo, "add", "target.c")
    _git(repo, "commit", "-m", "old")

    (repo / "target.c").write_text(
        "seed\n"
        "clean_anchor\n"
        "ONLY_NEW\n"
        "renamed_failure_anchor\n"
        "blocked_anchor\n",
        encoding="utf-8",
    )
    _git(repo, "add", "target.c")
    _git(repo, "commit", "-m", "new")
    return repo


class RebaseCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-rebase-test-")
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.upstream = _init_upstream(self.base)

        self.patches_root = self.base / "patches"
        self.patches_root.mkdir()
        (self.patches_root / "0100_clean.py").write_text(CLEAN_PATCH, encoding="utf-8")
        (self.patches_root / "0200_na.py").write_text(NOT_APPLICABLE_PATCH, encoding="utf-8")
        (self.patches_root / "0300_failed.py").write_text(FAILED_PATCH, encoding="utf-8")
        (self.patches_root / "0400_blocked.py").write_text(BLOCKED_PATCH, encoding="utf-8")

        self.overlay_root = self.base / "overlay"
        self.overlay_root.mkdir()

        self._old_patches = paths.PATCHES
        self._old_overlay = paths.SRC_OVERLAY
        paths.PATCHES = self.patches_root
        paths.SRC_OVERLAY = self.overlay_root

    def tearDown(self) -> None:
        _git(self.upstream, "worktree", "prune")
        paths.PATCHES = self._old_patches
        paths.SRC_OVERLAY = self._old_overlay

    def _patch_by_id(self, report: dict, patch_id: str) -> dict:
        return next(p for p in report["patches"] if p["patch_id"] == patch_id)

    def test_clean_not_applicable_failed_and_blocked_by_dependency(self):
        report = rebase.run_rebase_check(self.upstream, all_patches=True)

        self.assertEqual(
            self._patch_by_id(report, "0100_clean")["status"], rebase.STATUS_CLEAN
        )
        self.assertEqual(
            self._patch_by_id(report, "0200_na")["status"], rebase.STATUS_NOT_APPLICABLE
        )
        failed = self._patch_by_id(report, "0300_failed")
        self.assertEqual(failed["status"], rebase.STATUS_FAILED)
        edit = failed["files"][0]["edits"][0]
        self.assertEqual(edit["status"], rebase.EDIT_FAILED)
        self.assertEqual(edit["reason_code"], rebase.REASON_ANCHOR_NO_MATCH)
        self.assertEqual(edit["actual_matches"], 0)

        blocked = self._patch_by_id(report, "0400_blocked")
        self.assertEqual(blocked["status"], rebase.STATUS_BLOCKED)

        known_good = set(report["known_good_patch_ids"])
        self.assertEqual(known_good, {"0100_clean", "0200_na"})
        self.assertTrue(report["summary"]["reconciliation_required"])
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(report["summary"]["blocked_by_dependency"], 1)

        # The vendor checkout itself must be untouched -- the probe runs in
        # an isolated detached worktree, never against `root` directly.
        status = _git(self.upstream, "status", "--porcelain")
        self.assertEqual(status, "")

    def test_render_report_reads_from_the_same_model_as_json(self):
        report = rebase.run_rebase_check(self.upstream, all_patches=True)
        table = rebase.render_report(report)
        self.assertIn("0300_failed", table)
        self.assertIn("FAILED_NEEDS_RECONCILIATION", table)
        self.assertIn("0400_blocked", table)
        self.assertIn("BLOCKED", table)
        self.assertIn("RECONCILIATION REQUIRED", table)

    def test_write_and_load_report_round_trip(self):
        report = rebase.run_rebase_check(self.upstream, all_patches=True)
        report_path = self.base / "report.json"
        rebase.write_report(report_path, report)
        loaded = rebase.load_report(report_path)
        self.assertEqual(loaded, report)


class StaleReportTests(unittest.TestCase):
    """Each bound identity field must independently trigger fail-closed
    rejection -- a report that outlives the state it describes must never
    silently authorize a known-good apply of the wrong subset."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-rebase-stale-")
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.upstream = _init_upstream(self.base)

        self.patches_root = self.base / "patches"
        self.patches_root.mkdir()
        (self.patches_root / "0100_clean.py").write_text(CLEAN_PATCH, encoding="utf-8")

        self.overlay_root = self.base / "overlay"
        self.overlay_root.mkdir()

        self._old_patches = paths.PATCHES
        self._old_overlay = paths.SRC_OVERLAY
        paths.PATCHES = self.patches_root
        paths.SRC_OVERLAY = self.overlay_root

        self.report = rebase.run_rebase_check(self.upstream, all_patches=True)

    def tearDown(self) -> None:
        _git(self.upstream, "worktree", "prune")
        paths.PATCHES = self._old_patches
        paths.SRC_OVERLAY = self._old_overlay

    def test_fresh_report_passes(self):
        known_good = rebase._require_fresh(dict(self.report), self.upstream)
        self.assertEqual(set(known_good), {"0100_clean"})

    def test_schema_version_mismatch_is_stale(self):
        stale = dict(self.report, schema_version=self.report["schema_version"] + 1)
        with self.assertRaises(rebase.StaleRebaseReportError):
            rebase._require_fresh(stale, self.upstream)

    def test_semantics_version_mismatch_is_stale(self):
        stale = dict(
            self.report,
            patch_application_semantics_version=self.report["patch_application_semantics_version"] + 1,
        )
        with self.assertRaises(rebase.StaleRebaseReportError):
            rebase._require_fresh(stale, self.upstream)

    def test_upstream_revision_mismatch_is_stale(self):
        stale = dict(self.report, upstream_revision="0" * 40)
        with self.assertRaises(rebase.StaleRebaseReportError):
            rebase._require_fresh(stale, self.upstream)

    def test_bigcherry_revision_mismatch_is_stale(self):
        stale = dict(self.report, bigcherry_revision="0" * 40)
        with self.assertRaises(rebase.StaleRebaseReportError):
            rebase._require_fresh(stale, self.upstream)

    def test_overlay_digest_mismatch_is_stale(self):
        stale = dict(self.report, overlay_digest="deadbeef")
        with self.assertRaises(rebase.StaleRebaseReportError):
            rebase._require_fresh(stale, self.upstream)

    def test_patch_implementation_digest_mismatch_is_stale(self):
        patches = [dict(p) for p in self.report["patches"]]
        patches[0]["implementation_digest"] = "deadbeef"
        stale = dict(self.report, patches=patches)
        with self.assertRaises(rebase.StaleRebaseReportError):
            rebase._require_fresh(stale, self.upstream)

    def test_apply_known_good_rejects_stale_report(self):
        report_path = self.base / "report.json"
        stale = dict(self.report, upstream_revision="0" * 40)
        rebase.write_report(report_path, stale)
        with self.assertRaises(rebase.StaleRebaseReportError):
            rebase.apply_known_good(self.upstream, report_path, force=True, dry_run=True)


class ApplyKnownGoodTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-rebase-apply-")
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.upstream = _init_upstream(self.base)

        self.patches_root = self.base / "patches"
        self.patches_root.mkdir()
        (self.patches_root / "0100_clean.py").write_text(CLEAN_PATCH, encoding="utf-8")
        (self.patches_root / "0300_failed.py").write_text(FAILED_PATCH, encoding="utf-8")

        self.overlay_root = self.base / "overlay"
        self.overlay_root.mkdir()

        self._old_patches = paths.PATCHES
        self._old_overlay = paths.SRC_OVERLAY
        paths.PATCHES = self.patches_root
        paths.SRC_OVERLAY = self.overlay_root

    def tearDown(self) -> None:
        _git(self.upstream, "worktree", "prune")
        paths.PATCHES = self._old_patches
        paths.SRC_OVERLAY = self._old_overlay

    def test_apply_known_good_writes_only_the_known_good_subset(self):
        from unittest import mock
        from types import SimpleNamespace

        report = rebase.run_rebase_check(self.upstream, all_patches=True)
        self.assertEqual(set(report["known_good_patch_ids"]), {"0100_clean"})
        report_path = self.base / "report.json"
        rebase.write_report(report_path, report)

        fake_record = SimpleNamespace(audit={"passed": True})
        with mock.patch(
            "bigcherry.__main__._record_for", return_value=fake_record,
        ):
            result = rebase.apply_known_good(self.upstream, report_path, force=False, dry_run=False)

        self.assertTrue(result.ok)
        self.assertTrue(result.partial)
        self.assertEqual(result.known_good_patch_ids, ("0100_clean",))
        content = (self.upstream / "target.c").read_text(encoding="utf-8")
        self.assertIn("clean_insert", content)


if __name__ == "__main__":
    unittest.main()
