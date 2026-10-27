"""WSL07: list and prune the lab host's worktrees (this workspace's lab, not a host-wide sweep)."""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass

from ..core import environment, paths

REMOVABLE, DIRTY, UNIQUE, IN_USE = "removable", "dirty", "unique", "in_use"


@dataclass(frozen=True)
class LabWorktree:
    path: str
    head: str
    branch: str | None
    primary: bool


def parse_porcelain(text: str) -> list[LabWorktree]:
    """``git worktree list --porcelain``; the first entry is the primary checkout."""
    found: list[LabWorktree] = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(" ", 1) if " " in line else (line, "") for line in block.splitlines())
        if "worktree" not in fields:
            continue
        branch = fields.get("branch")
        found.append(LabWorktree(
            path=fields["worktree"], head=fields.get("HEAD", ""),
            branch=branch.removeprefix("refs/heads/") if branch else None,
            primary=not found,
        ))
    return found


def classify(*, dirty: bool, on_origin: bool, in_use: bool = False) -> str:
    """In use beats everything (a running job); dirty wins over unique: all of them mean keep."""
    if in_use:
        return IN_USE
    if dirty:
        return DIRTY
    return REMOVABLE if on_origin else UNIQUE


def _inspect_script(repo_root: str) -> str:
    return "\n".join([
        "set -u",
        f"repo={shlex.quote(repo_root)}",
        'git -C "$repo" fetch --quiet --prune origin || echo "FETCH-FAILED"',
        # a process whose cwd or open file is under the worktree (the slice lab wrapper cd's into its stage)
        'busy=$(for p in /proc/[0-9]*; do readlink "$p/cwd" "$p"/fd/* 2>/dev/null | sed "s|$|/|"; done)',
        'git -C "$repo" worktree list --porcelain | sed -n "s/^worktree //p" | while IFS= read -r wt; do',
        '  head=$(git -C "$wt" rev-parse HEAD 2>/dev/null)',
        '  dirty=0; test -n "$(git -C "$wt" status --porcelain 2>/dev/null)" && dirty=1',
        '  onorigin=0; test -n "$(git -C "$repo" branch -r --contains "$head" 2>/dev/null)" && onorigin=1',
        '  inuse=0; printf "%s\\n" "$busy" | grep -qF -e "$wt/" && inuse=1',
        '  printf "STATE\\t%s\\t%s\\t%s\\t%s\\n" "$wt" "$dirty" "$onorigin" "$inuse"',
        "done",
        'echo "LIST-BEGIN"',
        'git -C "$repo" worktree list --porcelain',
    ]) + "\n"


def parse_inspection(output: str) -> list[tuple[LabWorktree, str]]:
    """Join the per-worktree STATE lines with the porcelain listing; unknown state counts as in use."""
    head, _, listing = output.replace("\r\n", "\n").partition("LIST-BEGIN\n")
    if "FETCH-FAILED" in head:
        raise ValueError("lab fetch failed; refusing to classify against a stale origin")
    flags: dict[str, tuple[bool, bool, bool]] = {}
    for line in head.splitlines():
        parts = line.split("\t")
        if parts[0] == "STATE" and len(parts) == 5:
            flags[parts[1]] = (parts[2] == "1", parts[3] == "1", parts[4] == "1")
    result = []
    for tree in parse_porcelain(listing):
        dirty, on_origin, in_use = flags.get(tree.path, (True, False, True))
        result.append((tree, classify(dirty=dirty, on_origin=on_origin, in_use=in_use)))
    return result


def _ssh(host: environment.Host, script: str) -> str:
    done = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host.hostname, "bash", "-se"],
        input=script.encode("utf-8"), capture_output=True, timeout=300,
    )
    if done.returncode != 0:
        raise ValueError(done.stderr.decode("utf-8", "replace").strip() or f"ssh exited {done.returncode}")
    return done.stdout.decode("utf-8", "replace")


def _rows(host: environment.Host) -> list[tuple[LabWorktree, str]]:
    if not host.hostname or not host.repo_root:
        raise ValueError("lab host needs hostname and repo-root")
    return parse_inspection(_ssh(host, _inspect_script(host.repo_root)))


def _print(rows: list[tuple[LabWorktree, str]]) -> None:
    for tree, state in rows:
        label = "primary" if tree.primary else state
        print(f"{label:10} {tree.branch or 'detached':40} {tree.head[:10]}  {tree.path}")


def cmd_lab_status(args) -> int:
    try:
        host = environment.load_default(repo_root=paths.primary_root()).host(args.host)
        _print(_rows(host))
    except (OSError, ValueError, subprocess.SubprocessError, environment.EnvironmentError_) as exc:
        print(f"slice status --lab: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_lab_prune(args) -> int:
    try:
        host = environment.load_default(repo_root=paths.primary_root()).host(args.host)
        rows = _rows(host)
        _print(rows)
        doomed = [tree for tree, state in rows if state == REMOVABLE and not tree.primary]
        print(f"slice prune --lab: removable={len(doomed)} apply={args.apply}")
        if args.apply and doomed:
            lines = [f"repo={shlex.quote(host.repo_root)}"]
            lines += [f'git -C "$repo" worktree remove {shlex.quote(tree.path)}' for tree in doomed]
            lines.append('git -C "$repo" worktree prune')
            _ssh(host, "\n".join(lines) + "\n")
    except (OSError, ValueError, subprocess.SubprocessError, environment.EnvironmentError_) as exc:
        print(f"slice prune --lab: {exc}", file=sys.stderr)
        return 1
    return 0
