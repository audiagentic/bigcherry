"""Detached upstream worktree tests."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.workspace import UpstreamRepository, WorkspaceError  # noqa: E402


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


class WorkspaceTests(unittest.TestCase):
    def test_detached_worktree_is_pinned_and_target_must_be_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            _git(root.parent, "init", str(root))
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "Test")
            (root / "source.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "source.txt")
            _git(root, "commit", "-m", "initial")
            revision = _git(root, "rev-parse", "HEAD")
            worktree = Path(directory) / "worktree"
            upstream = UpstreamRepository(root)
            upstream.add_detached_worktree(revision, worktree)
            self.assertEqual(_git(worktree, "rev-parse", "HEAD"), revision)
            (worktree / "extra").write_text("x", encoding="utf-8")
            with self.assertRaises(WorkspaceError):
                upstream.add_detached_worktree(revision, worktree)
            upstream.remove_worktree(worktree)


if __name__ == "__main__":
    unittest.main()
