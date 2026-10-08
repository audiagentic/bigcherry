from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry.core import paths
from bigcherry.core.context import ProjectContext


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class PrimaryRootTests(unittest.TestCase):
    def _repo_with_worktree(self, directory: str) -> tuple[Path, Path]:
        primary = Path(directory) / "primary"
        primary.mkdir()
        subprocess.run(["git", "init", "-b", "main", str(primary)], check=True, capture_output=True)
        _git(primary, "config", "user.name", "PA47 Test")
        _git(primary, "config", "user.email", "pa47@example.invalid")
        (primary / "README.md").write_text("test\n", encoding="utf-8")
        _git(primary, "add", "README.md")
        _git(primary, "commit", "-m", "initial")
        worktree = Path(directory) / "slice"
        _git(primary, "worktree", "add", "-b", "feat/pa47-test", str(worktree), "HEAD")
        return primary.resolve(), worktree.resolve()

    def test_git_common_dir_resolves_primary_from_linked_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            primary, worktree = self._repo_with_worktree(directory)
            with (
                patch.object(paths, "REPO_ROOT", worktree),
                patch.dict(os.environ, {"BC_PRIMARY_ROOT": ""}),
            ):
                self.assertEqual(paths.primary_root(), primary)
                self.assertEqual(paths.llama_root(), primary / "vendor" / "llama.cpp")

    def test_context_keeps_slice_sources_but_shares_mutable_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            primary, worktree = self._repo_with_worktree(directory)
            with (
                patch.object(paths, "REPO_ROOT", worktree),
                patch.dict(
                    os.environ,
                    {"BC_PRIMARY_ROOT": "", "BIGCHERRY_WORK_ROOT": ""},
                ),
            ):
                context = ProjectContext.resolve(project_root=worktree)
            self.assertEqual(context.project_root, worktree)
            self.assertEqual(context.overlay_root, worktree / "src")
            self.assertEqual(context.patches_root, worktree / "patches")
            self.assertEqual(context.work_root, primary / "work")
            self.assertEqual(
                context.upstream_repo,
                primary / "work" / "upstream" / "llama.cpp.git",
            )

    def test_explicit_primary_override_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            primary, worktree = self._repo_with_worktree(directory)
            with patch.dict(os.environ, {"BC_PRIMARY_ROOT": str(primary)}):
                self.assertEqual(paths.primary_root(), primary)
            with self.assertRaisesRegex(paths.PrimaryRootError, "primary checkout"):
                paths.primary_root(worktree)

    def test_resolution_fails_closed_outside_git(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with (
                patch.object(paths, "REPO_ROOT", root),
                patch.dict(os.environ, {"BC_PRIMARY_ROOT": ""}),
            ):
                with self.assertRaises(paths.PrimaryRootError):
                    paths.primary_root()


if __name__ == "__main__":
    unittest.main()
