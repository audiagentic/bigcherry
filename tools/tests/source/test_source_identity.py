"""Git-tree source identity tests."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.source.identity import SourceIdentityError, git_tree_oid, source_slice_id  # noqa: E402


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


class SourceIdentityTests(unittest.TestCase):
    def test_clean_tree_matches_commit_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _git(root.parent, "init", str(root))
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "Test")
            (root / "source.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "source.txt")
            _git(root, "commit", "-m", "initial")
            self.assertEqual(git_tree_oid(root), _git(root, "rev-parse", "HEAD^{tree}"))

    def test_unexpected_untracked_files_fail_and_allowed_files_are_included(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _git(root.parent, "init", str(root))
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "Test")
            (root / "source.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "source.txt")
            _git(root, "commit", "-m", "initial")
            (root / "overlay.txt").write_text("overlay\n", encoding="utf-8")
            with self.assertRaises(SourceIdentityError):
                git_tree_oid(root)
            with_allowed = git_tree_oid(root, allowed_untracked={"overlay.txt"})
            self.assertNotEqual(with_allowed, _git(root, "rev-parse", "HEAD^{tree}"))

    def test_modified_tracked_file_is_not_unexpected(self):
        # workspace.materialize() overlays files and applies patches AFTER
        # add_detached_worktree() checks out a pristine upstream revision --
        # both legitimately modify already-tracked upstream files (that is
        # the entire mechanism of a patch). A materialize() against any
        # source with real patches must not be rejected as "unexpected"
        # for exactly the modifications it is supposed to produce.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _git(root.parent, "init", str(root))
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "Test")
            (root / "source.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "source.txt")
            _git(root, "commit", "-m", "initial")
            # Simulate a patch: modify an already-tracked file in place.
            (root / "source.txt").write_text("one\npatched\n", encoding="utf-8")
            tree_oid = git_tree_oid(root)
            self.assertNotEqual(tree_oid, _git(root, "rev-parse", "HEAD^{tree}"))

    def test_deleted_tracked_file_is_not_unexpected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _git(root.parent, "init", str(root))
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "Test")
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            (root / "b.txt").write_text("b\n", encoding="utf-8")
            _git(root, "add", "a.txt", "b.txt")
            _git(root, "commit", "-m", "initial")
            (root / "b.txt").unlink()
            tree_oid = git_tree_oid(root)
            self.assertNotEqual(tree_oid, _git(root, "rev-parse", "HEAD^{tree}"))

    def test_modification_plus_genuinely_unexpected_untracked_file_still_rejected(self):
        # The fix must not become blanket-permissive: a real stray
        # untracked file alongside a legitimate patch modification is
        # still exactly what this check exists to catch.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _git(root.parent, "init", str(root))
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "Test")
            (root / "source.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "source.txt")
            _git(root, "commit", "-m", "initial")
            (root / "source.txt").write_text("one\npatched\n", encoding="utf-8")
            (root / "stray.txt").write_text("litter\n", encoding="utf-8")
            with self.assertRaises(SourceIdentityError):
                git_tree_oid(root)

    def test_slice_identity_is_content_based(self):
        first = source_slice_id(upstream_revision="a" * 40, tree_oid="b" * 40)
        same = source_slice_id(upstream_revision="a" * 40, tree_oid="b" * 40)
        changed = source_slice_id(upstream_revision="a" * 40, tree_oid="c" * 40)
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()
