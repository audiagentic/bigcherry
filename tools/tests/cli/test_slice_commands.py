from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.cli import slice as slice_cli  # noqa: E402


def _run(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _gh(*, state: str, merged_at: str | None = None, branch: str = "feat/pa47-test"):
    def run(args, *, check=True):
        row = {
            "number": 41,
            "state": state,
            "mergedAt": merged_at,
            "headRefName": branch,
        }
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps([row]), stderr="")
    return run


class SliceCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.origin = root / "origin.git"
        self.primary = root / "primary"
        subprocess.run(["git", "init", "--bare", str(self.origin)], check=True, capture_output=True)
        subprocess.run(["git", "init", "-b", "main", str(self.primary)], check=True, capture_output=True)
        _run(self.primary, "config", "user.name", "PA47 Test")
        _run(self.primary, "config", "user.email", "pa47@example.invalid")
        (self.primary / ".gitignore").write_text("/worktrees/\n", encoding="utf-8")
        (self.primary / "README.md").write_text("main\n", encoding="utf-8")
        _run(self.primary, "add", ".gitignore", "README.md")
        _run(self.primary, "commit", "-m", "initial")
        _run(self.primary, "remote", "add", "origin", str(self.origin))
        _run(self.primary, "push", "-u", "origin", "main")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _start(self, branch: str = "feat/pa47-test") -> Path:
        return slice_cli.start_slice(branch, primary_root=self.primary)

    def test_start_uses_origin_main_and_rejects_bad_names_and_duplicates(self):
        remote_head = _run(self.primary, "rev-parse", "origin/main").stdout.strip()
        (self.primary / "LOCAL_ONLY").write_text("local\n", encoding="utf-8")
        _run(self.primary, "add", "LOCAL_ONLY")
        _run(self.primary, "commit", "-m", "local only")

        worktree = self._start()
        self.assertEqual(_run(worktree, "rev-parse", "HEAD").stdout.strip(), remote_head)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self._start()
        with self.assertRaises(ValueError):
            slice_cli.start_slice("Bad Branch", primary_root=self.primary)

    def test_finish_refuses_open_pr(self):
        worktree = self._start()
        with self.assertRaisesRegex(RuntimeError, "requires merged or closed"):
            slice_cli.finish_slice(
                "feat/pa47-test",
                primary_root=self.primary,
                gh_runner=_gh(state="OPEN"),
            )
        self.assertTrue(worktree.exists())

    def test_finish_refuses_dirty_worktree(self):
        worktree = self._start()
        (worktree / "dirty.txt").write_text("dirty\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "worktree is dirty"):
            slice_cli.finish_slice(
                "feat/pa47-test",
                primary_root=self.primary,
                gh_runner=_gh(state="CLOSED"),
            )
        self.assertTrue(worktree.exists())

    def test_finish_closed_pr_removes_worktree_and_local_and_remote_branch(self):
        worktree = self._start()
        _run(worktree, "push", "-u", "origin", "feat/pa47-test")
        slice_cli.finish_slice(
            "feat/pa47-test",
            primary_root=self.primary,
            gh_runner=_gh(state="CLOSED"),
        )
        self.assertFalse(worktree.exists())
        self.assertNotEqual(
            _run(self.primary, "show-ref", "--verify", "--quiet", "refs/heads/feat/pa47-test", check=False).returncode,
            0,
        )
        remote = _run(
            self.primary,
            "ls-remote",
            "--exit-code",
            "--heads",
            "origin",
            "feat/pa47-test",
            check=False,
        )
        self.assertEqual(remote.returncode, 2)
        self.assertEqual(_run(self.primary, "branch", "--show-current").stdout.strip(), "main")

    def test_prune_lists_then_removes_merged_worktree_without_force(self):
        worktree = self._start()
        (worktree / "slice.txt").write_text("slice\n", encoding="utf-8")
        _run(worktree, "add", "slice.txt")
        _run(worktree, "commit", "-m", "slice")
        _run(worktree, "push", "-u", "origin", "feat/pa47-test")
        _run(self.primary, "merge", "--ff-only", "feat/pa47-test")
        _run(self.primary, "push", "origin", "main")

        listed = slice_cli.prune_orphaned_worktrees(
            primary_root=self.primary,
            apply=False,
            gh_runner=_gh(state="MERGED", merged_at="2026-10-08T00:00:00Z"),
        )
        self.assertEqual(listed["eligible"], ["feat/pa47-test"])
        self.assertTrue(worktree.exists())

        applied = slice_cli.prune_orphaned_worktrees(
            primary_root=self.primary,
            apply=True,
            gh_runner=_gh(state="MERGED", merged_at="2026-10-08T00:00:00Z"),
        )
        self.assertEqual(applied["removed"], ["feat/pa47-test"])
        self.assertFalse(worktree.exists())

    def test_prune_handles_squash_merged_pr_not_in_git_ancestry(self):
        worktree = self._start()
        (worktree / "slice.txt").write_text("squash\\n", encoding="utf-8")
        _run(worktree, "add", "slice.txt")
        _run(worktree, "commit", "-m", "slice")
        _run(self.primary, "merge", "--squash", "feat/pa47-test")
        _run(self.primary, "commit", "-m", "squashed slice")
        _run(self.primary, "push", "origin", "main")
        self.assertNotEqual(
            _run(
                self.primary, "merge-base", "--is-ancestor",
                "feat/pa47-test", "origin/main", check=False,
            ).returncode,
            0,
        )
        found = slice_cli.prune_orphaned_worktrees(
            primary_root=self.primary,
            gh_runner=_gh(state="MERGED", merged_at="2026-10-08T00:00:00Z"),
        )
        self.assertEqual(found["eligible"], ["feat/pa47-test"])
        self.assertTrue(worktree.exists())

    def test_prune_never_removes_unpushed_active_slice(self):
        worktree = self._start()
        result = slice_cli.prune_orphaned_worktrees(
            primary_root=self.primary, apply=True,
            gh_runner=_gh(state="OPEN"),
        )
        self.assertEqual(result["removed"], [])
        self.assertTrue(worktree.exists())
        self.assertEqual(
            _run(worktree, "branch", "--show-current").stdout.strip(),
            "feat/pa47-test",
        )

    def test_status_reports_pr_divergence_and_dirty_flag(self):
        worktree = self._start()
        rows = slice_cli.slice_status(
            primary_root=self.primary,
            gh_runner=_gh(state="OPEN"),
        )
        row = next(item for item in rows if item["branch"] == "feat/pa47-test")
        self.assertEqual(row["pr"], 41)
        self.assertEqual(row["state"], "OPEN")
        self.assertEqual(row["ahead"], 0)
        self.assertEqual(row["behind"], 0)
        self.assertFalse(row["dirty"])


if __name__ == "__main__":
    unittest.main()
