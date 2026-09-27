"""Fail-closed git helpers for job evidence harvest.

No helper in this module stashes, resets, or stages an implicit path set.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class GitOpsError(RuntimeError):
    pass


def _run(
    repo: Path,
    *args: str,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        ("git", "-C", str(repo.resolve()), *args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    if check and completed.returncode != 0:
        stderr = completed.stderr if text else completed.stderr.decode("utf-8", "replace")
        raise GitOpsError(f"git {' '.join(args)} failed: {stderr.strip()}")
    return completed


def head(repo: Path) -> str:
    return str(_run(repo, "rev-parse", "HEAD").stdout).strip()


def show_text(repo: Path, revision: str, relpath: str) -> str | None:
    completed = _run(repo, "show", f"{revision}:{relpath}", check=False)
    if completed.returncode == 0:
        return str(completed.stdout)
    stderr = str(completed.stderr)
    missing_markers = (
        "does not exist in",
        "exists on disk, but not in",
        "Path '",
        "fatal: path",
    )
    if any(marker in stderr for marker in missing_markers):
        return None
    raise GitOpsError(f"git show {revision}:{relpath} failed: {stderr.strip()}")


def staged_paths(repo: Path) -> tuple[str, ...]:
    raw = _run(repo, "diff", "--cached", "--name-only", "-z").stdout
    return tuple(sorted(part for part in str(raw).split("\0") if part))


def require_clean_index(repo: Path) -> None:
    staged = staged_paths(repo)
    if staged:
        raise GitOpsError(
            "refusing evidence harvest with pre-existing staged paths: "
            + ", ".join(staged)
        )


def stage_exact(repo: Path, relpaths: tuple[str, ...]) -> tuple[str, ...]:
    if not relpaths:
        return ()
    for path in relpaths:
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise GitOpsError(f"unsafe git path: {path}")
    _run(repo, "add", "--", *relpaths)
    _run(repo, "diff", "--cached", "--check")
    staged = staged_paths(repo)
    expected = tuple(sorted(relpaths))
    if staged != expected:
        raise GitOpsError(
            f"staged path set differs from harvest manifest: {staged!r} != {expected!r}"
        )
    return staged


def commit_staged(repo: Path, *, message: str, expected_paths: tuple[str, ...]) -> str:
    staged = staged_paths(repo)
    if staged != tuple(sorted(expected_paths)):
        raise GitOpsError(
            f"refusing commit with unexpected staged paths: {staged!r}"
        )
    if not staged:
        return head(repo)
    _run(repo, "commit", "-m", message)
    return head(repo)
