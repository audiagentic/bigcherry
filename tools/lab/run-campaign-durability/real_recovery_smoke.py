"""Real-process recovery/commit-pinning smoke for RCD planning.

Uses current BigCherry HI151 code plus real git worktrees. No GPU/Slurm claims.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from bigcherry.core.tree_activity import (
    Lease,
    MaintenanceLock,
    TreeActivityError,
    list_live_leases,
    maintenance_is_held,
    prune_stale_leases,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


def tree_activity_cases(root: Path) -> int:
    checks = 0
    work = root / "work"
    project = root / "project"
    project.mkdir()

    with MaintenanceLock(work, project):
        check(maintenance_is_held(work, project), "maintenance marker visible"); checks += 1
        try:
            with Lease(work, project, command="campaign", run_id="r1"):
                pass
        except TreeActivityError:
            checks += 1
        else:
            raise AssertionError("lease entered while maintenance lock held")
    check(not maintenance_is_held(work, project), "maintenance marker released"); checks += 1

    with Lease(work, project, command="campaign", run_id="r2"):
        check(len(list_live_leases(work, project)) == 1, "live lease published"); checks += 1
        try:
            with MaintenanceLock(work, project):
                pass
        except TreeActivityError:
            checks += 1
        else:
            raise AssertionError("maintenance entered while live lease held")
    check(not list_live_leases(work, project), "lease removed on normal exit"); checks += 1

    # Crash a real child while it holds Lease; stale PID lease must be removable.
    code = """
import os, sys
from pathlib import Path
from bigcherry.core.tree_activity import Lease
work=Path(sys.argv[1]); project=Path(sys.argv[2]); marker=Path(sys.argv[3])
lease=Lease(work, project, command='crash-child', run_id='crash')
lease.__enter__(); marker.write_text('ready'); os._exit(17)
"""
    marker = root / "lease-ready"
    child = subprocess.run(
        [sys.executable, "-c", code, str(work), str(project), str(marker)],
        env=os.environ.copy(),
    )
    check(child.returncode == 17 and marker.exists(), "crash child published lease"); checks += 1
    check(len(list_live_leases(work, project)) == 0, "dead local PID not live"); checks += 1
    removed = prune_stale_leases(work, project)
    check(len(removed) == 1, "stale crash lease pruned explicitly"); checks += 1
    return checks


def git_pinning_cases(root: Path) -> int:
    checks = 0
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "patch-refactor", str(repo)], check=True)
    git(repo, "config", "user.email", "rcd@example.invalid")
    git(repo, "config", "user.name", "RCD Smoke")

    tracked = repo / "version.txt"
    tracked.write_text("A\n", encoding="utf-8")
    git(repo, "add", "version.txt"); git(repo, "commit", "-q", "-m", "A")
    sha_a = git(repo, "rev-parse", "HEAD")

    tracked.write_text("B\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "B")
    sha_b = git(repo, "rev-parse", "HEAD")
    check(sha_a != sha_b, "branch moved to newer commit"); checks += 1

    runner = root / "runner-a"
    git(repo, "worktree", "add", "--detach", str(runner), sha_a)
    check(git(runner, "rev-parse", "HEAD") == sha_a, "attempt worktree pinned exact commit"); checks += 1
    check((runner / "version.txt").read_text(encoding="utf-8") == "A\n", "pinned bytes are old commit"); checks += 1

    tracked.write_text("C\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "C")
    sha_c = git(repo, "rev-parse", "HEAD")
    check(sha_c not in {sha_a, sha_b}, "branch advanced again"); checks += 1
    check(git(runner, "rev-parse", "HEAD") == sha_a, "running attempt unaffected by branch advance"); checks += 1
    check((runner / "version.txt").read_text(encoding="utf-8") == "A\n", "running attempt bytes unchanged"); checks += 1

    # Current evidence writer dirties REPO_ROOT; this demonstrates why v1 keeps
    # a per-attempt runner until evidence is externalized/harvested.
    evidence = runner / "attempt-evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    check("attempt-evidence.json" in git(runner, "status", "--porcelain"), "attempt worktree can carry evidence dirtiness"); checks += 1
    check(git(repo, "status", "--porcelain") == "", "control checkout remains clean"); checks += 1

    git(repo, "worktree", "remove", "--force", str(runner))
    git(repo, "worktree", "prune")
    check(not runner.exists(), "attempt worktree cleanup works"); checks += 1
    return checks


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="bigcherry-rcd-recovery-") as temp:
        root = Path(temp)
        checks = tree_activity_cases(root) + git_pinning_cases(root)
    print(json.dumps({"checks": checks, "ok": True, "scope": "real-hi151-and-git-worktree"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
