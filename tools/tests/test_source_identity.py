"""Git-tree source identity tests."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.source_identity import SourceIdentityError, git_tree_oid, source_slice_id  # noqa: E402


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

    def test_slice_identity_is_content_based(self):
        first = source_slice_id(upstream_revision="a" * 40, tree_oid="b" * 40)
        same = source_slice_id(upstream_revision="a" * 40, tree_oid="b" * 40)
        changed = source_slice_id(upstream_revision="a" * 40, tree_oid="c" * 40)
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()
