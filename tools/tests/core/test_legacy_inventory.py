"""WSL06 legacy discovery is read-only, conservative and auditable."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

from bigcherry.cli import workspace_cmds
import os
import shutil
import subprocess

from bigcherry.core import legacy_inventory
from bigcherry.core.legacy_inventory import inventory_legacy


class LegacyInventoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "runs" / "ab" / "legacy-test").mkdir(parents=True)
        (self.root / "runs" / "ab" / "legacy-test" / "capture.bin").write_bytes(b"x" * 33)
        (self.root / "builds" / "recipe" / "build").mkdir(parents=True)
        (self.root / "sources" / "src").mkdir(parents=True)
        (self.root / "worktrees" / "old-checkout").mkdir(parents=True)
        (self.root / "worktrees" / "old-checkout" / ".git").write_text("gitdir: missing")
        (self.root / "legacy-cache").mkdir()

    def test_classifies_every_root_without_cleanup_candidates(self):
        report = inventory_legacy(self.root)
        self.assertTrue(report["read_only"])
        self.assertEqual(report["schema_version"], legacy_inventory.SCHEMA_VERSION)
        self.assertEqual(report["totals"]["entries"], 5)
        self.assertTrue(report["totals"]["incomplete"])
        self.assertTrue(all(e["action"] == "protected" for e in report["entries"]))
        self.assertEqual(
            {e["category"] for e in report["entries"]},
            {"run", "build", "source", "worktree", "unclassified-root"},
        )
        self.assertIn("legacy-cache", next(e["path"] for e in report["entries"]
                                           if e["category"] == "unclassified-root"))

    def test_complete_size_is_payload_not_reclamation_estimate(self):
        report = inventory_legacy(self.root, measure=True)
        run = next(e for e in report["entries"] if e["category"] == "run")
        self.assertTrue(run["size_complete"])
        self.assertEqual(run["size_bytes"], 33)
        self.assertFalse(report["totals"]["incomplete"])
        self.assertTrue(all(e["action"] == "protected" for e in report["entries"]))

    def _snapshot(self):
        return sorted((str(p.relative_to(self.root)), p.stat().st_mtime_ns, p.is_dir() or p.stat().st_size)
                      for p in self.root.rglob("*"))

    def test_human_report_renders_and_nothing_changes(self):
        before = self._snapshot()
        args = Namespace(work_root=str(self.root), legacy=True, builds=False, extra_root=None,
                         no_size=False, json=False, apply=False, min_age_days=7, keep_newest=3, verbose=False)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(workspace_cmds.cmd_clean(args), 0)
        text = out.getvalue()
        self.assertIn("read-only: 5 protected entries, 1 unknown objects", text)
        self.assertIn("unattributed", text)
        self.assertEqual(self._snapshot(), before)

    def test_reconciliation_attributes_every_measured_byte(self):
        (self.root / "runs" / "ab" / "loose.log").write_bytes(b"y" * 7)   # a loose file is an entry too
        report = inventory_legacy(self.root, measure=True)
        rec = report["reconciliation"][0]
        self.assertEqual(rec["attributed_bytes"], report["totals"]["known_measured_bytes"])
        self.assertEqual(rec["measured_bytes"], rec["attributed_bytes"])
        self.assertEqual(rec["unattributed_bytes"], 0)
        self.assertTrue(any(e["path"].endswith("loose.log") for e in report["entries"]))

    def test_hard_link_is_counted_once(self):
        before = inventory_legacy(self.root, measure=True)["totals"]["known_measured_bytes"]
        os.link(self.root / "runs" / "ab" / "legacy-test" / "capture.bin", self.root / "legacy-cache" / "capture-link.bin")
        report = inventory_legacy(self.root, measure=True)
        self.assertEqual(report["totals"]["known_measured_bytes"], before)
        self.assertEqual(report["reconciliation"][0]["measured_bytes"], before)

    def test_repeated_runs_are_identical(self):
        self.assertEqual(json.dumps(inventory_legacy(self.root, measure=True), sort_keys=True),
                         json.dumps(inventory_legacy(self.root, measure=True), sort_keys=True))

    def test_references_come_from_tracked_repository_files(self):
        repo = self.root.parent / (self.root.name + "-repo")
        repo.mkdir()
        self.addCleanup(lambda: shutil.rmtree(repo, ignore_errors=True))
        (repo / "notes.md").write_text("evidence in runs/ab/legacy-test only\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "notes.md"], check=True)
        report = inventory_legacy(self.root, repo_root=repo)
        run = next(e for e in report["entries"] if e["category"] == "run")
        self.assertEqual(run["references"], ["notes.md"])
        self.assertEqual(run["disposition"], "keep-referenced")
        cache = next(e for e in report["entries"] if e["path"].endswith("legacy-cache"))
        self.assertEqual(cache["references"], [])
        self.assertEqual(cache["disposition"], "owner-decision")

    def test_worktree_git_state_decides_recreatability(self):
        remote = self.root.parent / (self.root.name + "-remote.git")
        self.addCleanup(lambda: shutil.rmtree(remote, ignore_errors=True))
        subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
        clone = self.root / "worktrees" / "clean-checkout"
        subprocess.run(["git", "clone", "-q", str(remote), str(clone)], check=True, stderr=subprocess.DEVNULL)
        for k, v in (("user.email", "t@example.com"), ("user.name", "t")):
            subprocess.run(["git", "-C", str(clone), "config", k, v], check=True)
        (clone / "a.txt").write_text("a", encoding="utf-8")
        subprocess.run(["git", "-C", str(clone), "add", "a.txt"], check=True)
        subprocess.run(["git", "-C", str(clone), "commit", "-qm", "a"], check=True)
        subprocess.run(["git", "-C", str(clone), "push", "-q", "origin", "HEAD"], check=True, stderr=subprocess.DEVNULL)
        entry = lambda: next(e for e in inventory_legacy(self.root)["entries"] if e["path"].endswith("clean-checkout"))
        self.assertEqual(entry()["git"]["dirty_paths"], 0)
        self.assertEqual(entry()["rebuildability"], "recreatable-from-remote")
        (clone / "b.txt").write_text("b", encoding="utf-8")
        self.assertEqual(entry()["git"]["dirty_paths"], 1)
        self.assertEqual(entry()["rebuildability"], "unknown")

    def test_extra_root_is_inventoried_and_reconciled_separately(self):
        extra = self.root.parent / (self.root.name + "-vault-work")
        (extra / "rd04-old").mkdir(parents=True)
        (extra / "rd04-old" / "x.bin").write_bytes(b"z" * 5)
        self.addCleanup(lambda: shutil.rmtree(extra, ignore_errors=True))
        report = inventory_legacy(self.root, measure=True, extra_roots=(extra,))
        self.assertEqual(report["roots"], [str(self.root), str(extra)])
        rd = next(e for e in report["entries"] if e["path"].endswith("rd04-old"))
        self.assertEqual((rd["root"], rd["category"], rd["size_bytes"]), (str(extra), "unclassified-root", 5))
        self.assertEqual([r["root"] for r in report["reconciliation"]], [str(self.root), str(extra)])

    @unittest.skipIf(sys.platform == "win32", "symlink permissions vary")
    def test_symlink_is_reported_without_traversal(self):
        external = self.root.parent / (self.root.name + "-outside")
        external.mkdir()
        self.addCleanup(external.rmdir)
        (self.root / "unknown-link").symlink_to(external, target_is_directory=True)
        report = inventory_legacy(self.root, measure=True)
        link = next(e for e in report["entries"] if e["path"].endswith("unknown-link"))
        self.assertTrue(link["is_symlink"])
        self.assertIsNone(link["size_bytes"])
        self.assertEqual(link["action"], "protected")

    def test_cli_json_read_only_apply_refused(self):
        args = Namespace(work_root=str(self.root), legacy=True, builds=False, extra_root=None,
                         no_size=True, json=True, apply=False,
                         min_age_days=7, keep_newest=3, verbose=False)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(workspace_cmds.cmd_clean(args), 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report["totals"]["protected"], 5)
        args.apply = True
        with redirect_stdout(io.StringIO()):
            self.assertEqual(workspace_cmds.cmd_clean(args), 2)
        self.assertTrue((self.root / "legacy-cache").exists())

    @unittest.skipIf(sys.platform == "win32", "symlink permissions vary")
    def test_read_only_refuses_root_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "link"
            path.symlink_to(self.root, target_is_directory=True)
            with self.assertRaises(ValueError):
                inventory_legacy(path)


if __name__ == "__main__":
    unittest.main()
