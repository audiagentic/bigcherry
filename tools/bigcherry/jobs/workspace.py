"""Attempt-scoped code pinning/worktree creation."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class AttemptWorkspace:
    commit: str
    project_root: Path


class WorkspaceManager(Protocol):
    def create(self, run_id: str, attempt: int, code_ref: str) -> AttemptWorkspace: ...


class GitWorkspaceManager:
    def __init__(self, project_root: Path, worktree_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.worktree_root = worktree_root.resolve()

    def _git(self, *args: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(self.project_root), *args),
            text=True, capture_output=True,
        )
        if completed.returncode != 0:
            raise WorkspaceError(completed.stderr.strip() or f"git {' '.join(args)} failed")
        return completed.stdout.strip()

    def create(self, run_id: str, attempt: int, code_ref: str) -> AttemptWorkspace:
        commit = self._git("rev-parse", "--verify", f"{code_ref}^{{commit}}")
        if len(commit) != 40:
            raise WorkspaceError(f"resolved code ref is not a full commit: {commit!r}")
        target = self.worktree_root / run_id / f"{attempt:03d}"
        if target.exists():
            existing = subprocess.run(("git", "-C", str(target), "rev-parse", "HEAD"), text=True, capture_output=True)
            if existing.returncode == 0 and existing.stdout.strip() == commit:
                return AttemptWorkspace(commit, target.resolve())
            raise WorkspaceError(f"attempt worktree already exists at a different commit: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ("git", "-C", str(self.project_root), "worktree", "add", "--detach", str(target), commit),
            text=True, capture_output=True,
        )
        if completed.returncode != 0:
            raise WorkspaceError(completed.stderr.strip() or "git worktree add failed")
        return AttemptWorkspace(commit, target.resolve())


class PassthroughWorkspaceManager:
    """Deterministic mock workspace manager for service tests."""

    def __init__(self, project_root: Path, commit: str = "0" * 40) -> None:
        self.project_root = project_root.resolve()
        self.commit = commit

    def create(self, run_id: str, attempt: int, code_ref: str) -> AttemptWorkspace:
        return AttemptWorkspace(self.commit, self.project_root)
