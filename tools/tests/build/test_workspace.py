"""Detached upstream worktree tests."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.source.workspace import UpstreamRepository, WorkspaceError  # noqa: E402


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


class FetchRefTests(unittest.TestCase):
    """RE13 follow-up (GPT-auto-agent review, 2026-08-17): fetch_ref()
    resolves via a private ref, not the shared mutable FETCH_HEAD."""

    def test_fetch_ref_resolves_the_correct_commit_and_leaves_no_stray_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            origin = Path(directory) / "origin"
            _git(origin.parent, "init", str(origin))
            _git(origin, "config", "user.email", "test@example.invalid")
            _git(origin, "config", "user.name", "Test")
            (origin / "source.txt").write_text("one\n", encoding="utf-8")
            _git(origin, "add", "source.txt")
            _git(origin, "commit", "-m", "first")
            first_revision = _git(origin, "rev-parse", "HEAD")
            (origin / "source.txt").write_text("two\n", encoding="utf-8")
            _git(origin, "add", "source.txt")
            _git(origin, "commit", "-m", "second")
            second_revision = _git(origin, "rev-parse", "HEAD")

            mirror = Path(directory) / "mirror"
            _git(directory, "clone", str(origin), str(mirror))
            # The clone already has `second_revision` as its default branch
            # tip -- reset origin to first_revision so fetch_ref has to
            # genuinely fetch something new, not just resolve what the
            # clone already had locally.
            _git(origin, "reset", "--hard", first_revision)

            upstream = UpstreamRepository(mirror)
            resolved = upstream.fetch_ref("master")
            self.assertEqual(resolved, first_revision)
            # No stray refs/bigcherry-probe/* ref left behind.
            leftover = _git(mirror, "for-each-ref", "refs/bigcherry-probe")
            self.assertEqual(leftover, "")
            self.assertNotEqual(first_revision, second_revision)

    def test_fetch_ref_updates_on_a_second_call_after_origin_moves(self):
        with tempfile.TemporaryDirectory() as directory:
            origin = Path(directory) / "origin"
            _git(origin.parent, "init", str(origin))
            _git(origin, "config", "user.email", "test@example.invalid")
            _git(origin, "config", "user.name", "Test")
            (origin / "source.txt").write_text("one\n", encoding="utf-8")
            _git(origin, "add", "source.txt")
            _git(origin, "commit", "-m", "first")
            first_revision = _git(origin, "rev-parse", "HEAD")

            mirror = Path(directory) / "mirror"
            _git(directory, "clone", str(origin), str(mirror))
            upstream = UpstreamRepository(mirror)
            self.assertEqual(upstream.fetch_ref("master"), first_revision)

            (origin / "source.txt").write_text("two\n", encoding="utf-8")
            _git(origin, "add", "source.txt")
            _git(origin, "commit", "-m", "second")
            second_revision = _git(origin, "rev-parse", "HEAD")

            self.assertEqual(upstream.fetch_ref("master"), second_revision)
            self.assertNotEqual(first_revision, second_revision)


if __name__ == "__main__":
    unittest.main()
