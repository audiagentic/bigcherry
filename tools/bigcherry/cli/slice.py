"""Short-lived slice branch maintenance commands (PA46)."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class RemoteBranch:
    name: str
    sha: str
    committed_at: int
    author: str


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _run_git(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _remote_branches(remote: str, *, runner: Runner = _run_git) -> list[RemoteBranch]:
    result = runner(
        [
            "for-each-ref",
            "--format=%(refname:strip=3)%09%(objectname)%09%(committerdate:unix)%09%(authorname)",
            f"refs/remotes/{remote}",
        ]
    )
    branches: list[RemoteBranch] = []
    for raw in result.stdout.splitlines():
        if not raw.strip():
            continue
        fields = raw.split("\t", 3)
        if len(fields) != 4:
            raise RuntimeError(f"unexpected git for-each-ref row: {raw!r}")
        name, sha, committed_at, author = fields
        if name == "HEAD":
            continue
        branches.append(RemoteBranch(name, sha, int(committed_at), author))
    return branches


def _is_merged(branch: RemoteBranch, remote: str, base: str, *, runner: Runner = _run_git) -> bool:
    result = runner(
        ["merge-base", "--is-ancestor", branch.sha, f"{remote}/{base}"],
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git merge-base failed for {branch.name}: {result.stderr.strip()}"
        )
    return result.returncode == 0


def _skip_branch(name: str, base: str) -> bool:
    return (
        name == base
        or name == "HEAD"
        or name.startswith("release-please--")
    )


def prune_remote_branches(
    *,
    remote: str = "origin",
    base: str = "main",
    stale_days: int = 14,
    dry_run: bool = False,
    now: int | None = None,
    runner: Runner = _run_git,
) -> dict[str, list[str]]:
    """Delete only branches fully merged into *base*; report old unmerged branches."""

    if stale_days < 0:
        raise ValueError("stale_days must be non-negative")

    runner(["fetch", "--prune", remote])
    runner(["rev-parse", "--verify", f"{remote}/{base}"])

    current = int(time.time()) if now is None else int(now)
    stale_cutoff = current - stale_days * 24 * 60 * 60
    merged: list[str] = []
    deleted: list[str] = []
    stale_unmerged: list[str] = []

    for branch in _remote_branches(remote, runner=runner):
        if _skip_branch(branch.name, base):
            continue
        if _is_merged(branch, remote, base, runner=runner):
            merged.append(branch.name)
            if not dry_run:
                runner(["push", remote, "--delete", branch.name])
                deleted.append(branch.name)
            continue
        if branch.committed_at <= stale_cutoff:
            stale_unmerged.append(branch.name)
            age_days = max(0, (current - branch.committed_at) // (24 * 60 * 60))
            print(
                f"stale-unmerged {branch.name} age={age_days}d author={branch.author}"
            )

    for name in merged:
        action = "would-delete" if dry_run else "deleted"
        print(f"merged {name} {action}")

    print(
        "slice-prune: "
        f"merged={len(merged)} deleted={len(deleted)} "
        f"stale-unmerged={len(stale_unmerged)}"
    )
    return {
        "merged": merged,
        "deleted": deleted,
        "stale_unmerged": stale_unmerged,
    }


def cmd_slice_prune(args) -> int:
    try:
        prune_remote_branches(
            remote=args.remote,
            base=args.base,
            stale_days=args.stale_days,
            dry_run=args.dry_run,
        )
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        print(f"slice prune: {exc}", file=sys.stderr)
        return 1
    return 0
