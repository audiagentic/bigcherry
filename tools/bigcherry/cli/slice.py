"""Short-lived slice branch and worktree management (PA46/PA47)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ..core import paths
from typing import Callable, Sequence


_BRANCH_RE = re.compile(
    r"^(?:feat|fix|perf|chore|docs|test|refactor|ci|build)/"
    r"[a-z0-9]+(?:-[a-z0-9]+)+$|^bump/b[0-9]+$"
)


@dataclass(frozen=True)
class RemoteBranch:
    name: str
    sha: str
    committed_at: int
    author: str


@dataclass(frozen=True)
class Worktree:
    path: Path
    head: str
    branch: str | None


@dataclass(frozen=True)
class PullRequest:
    number: int
    state: str
    merged_at: str | None


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _run_git(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _run_gh(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gh", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _at(root: Path | None, args: Sequence[str]) -> list[str]:
    return list(args) if root is None else ["-C", str(root), *args]


def _checked(result: subprocess.CompletedProcess[str], what: str) -> str:
    if result.returncode != 0:
        raise RuntimeError(
            f"{what}: {result.stderr.strip() or result.stdout.strip() or 'command failed'}"
        )
    return result.stdout.strip()


def validate_branch_name(branch: str) -> None:
    if not _BRANCH_RE.fullmatch(branch):
        raise ValueError(
            "branch must match BRANCHING.md: <type>/<item>-<topic> in lower-case "
            "hyphen form (or bump/b<build>)"
        )


def _ref_exists(
    ref: str, *, root: Path | None = None, runner: Runner = _run_git
) -> bool:
    result = runner(
        _at(root, ["show-ref", "--verify", "--quiet", ref]),
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git show-ref failed for {ref}: {result.stderr.strip()}"
        )
    return result.returncode == 0


def _remote_branch_exists(
    branch: str,
    *,
    remote: str,
    root: Path,
    runner: Runner = _run_git,
) -> bool:
    result = runner(
        _at(root, ["ls-remote", "--exit-code", "--heads", remote, branch]),
        check=False,
    )
    if result.returncode not in (0, 2):
        raise RuntimeError(
            f"git ls-remote failed for {remote}/{branch}: {result.stderr.strip()}"
        )
    return result.returncode == 0


def _worktrees(*, root: Path, runner: Runner = _run_git) -> list[Worktree]:
    output = _checked(
        runner(_at(root, ["worktree", "list", "--porcelain"]), check=False),
        "git worktree list",
    )
    rows: list[Worktree] = []
    path: Path | None = None
    head = ""
    branch: str | None = None
    for raw in [*output.splitlines(), ""]:
        if not raw:
            if path is not None:
                rows.append(Worktree(path.resolve(), head, branch))
            path, head, branch = None, "", None
            continue
        key, _, value = raw.partition(" ")
        if key == "worktree":
            path = Path(value)
        elif key == "HEAD":
            head = value
        elif key == "branch":
            prefix = "refs/heads/"
            branch = value[len(prefix):] if value.startswith(prefix) else value
    return rows


def _worktree_for(
    branch: str, *, root: Path, runner: Runner = _run_git
) -> Worktree | None:
    for worktree in _worktrees(root=root, runner=runner):
        if worktree.branch == branch:
            return worktree
    return None


def _remove_empty_parents(path: Path, *, stop: Path) -> None:
    current = path
    while current != stop and stop in current.parents:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _leftover_matches_branch(
    target: Path,
    branch: str,
    *,
    root: Path,
    runner: Runner = _run_git,
) -> bool:
    """Verify a dead-record worktree against its still-present local branch."""

    ref = f"refs/heads/{branch}"
    if not _ref_exists(ref, root=root, runner=runner):
        return False
    listed = _checked(
        runner(_at(root, ["ls-tree", "-r", "-z", ref]), check=False),
        f"read tree for {branch}",
    )
    expected: dict[str, tuple[str, str, str]] = {}
    for row in listed.split("\0"):
        if not row:
            continue
        meta, sep, rel = row.partition("\t")
        fields = meta.split()
        if not sep or len(fields) != 3:
            raise RuntimeError(f"unexpected git ls-tree row for {branch}: {row!r}")
        mode, obj_type, sha = fields
        expected[rel] = (mode, obj_type, sha)

    actual: set[str] = set()
    for path in target.rglob("*"):
        rel = path.relative_to(target).as_posix()
        if rel == ".git" or rel.startswith(".git/"):
            continue
        if path.is_file() or path.is_symlink():
            actual.add(rel)

    expected_files = {
        rel
        for rel, (mode, obj_type, _sha) in expected.items()
        if obj_type == "blob" and mode != "160000"
    }
    if actual != expected_files:
        return False

    for rel in sorted(expected_files):
        mode, _obj_type, sha = expected[rel]
        path = target / rel
        if mode == "120000":
            if not path.is_symlink():
                return False
            link_text = path.readlink().as_posix()
            blob = _checked(
                runner(_at(root, ["cat-file", "-p", sha]), check=False),
                f"read symlink blob {rel}",
            )
            if link_text != blob:
                return False
            continue
        hashed = _checked(
            runner(
                _at(root, ["hash-object", f"--path={rel}", str(path)]),
                check=False,
            ),
            f"hash leftover file {rel}",
        )
        if hashed != sha:
            return False
    return True


def _remove_leftover_worktree_dir(
    target: Path,
    *,
    root: Path,
    branch: str | None = None,
    runner: Runner = _run_git,
) -> bool:
    """Remove a leftover empty/clean slice directory without discarding work."""

    if not target.exists():
        _remove_empty_parents(target.parent, stop=root / "worktrees")
        return True

    worktrees_root = (root / "worktrees").resolve()
    resolved = target.resolve()
    if worktrees_root not in resolved.parents:
        raise RuntimeError(f"refusing to clean path outside worktrees/: {target}")
    if not target.is_dir():
        raise RuntimeError(f"leftover worktree path is not a directory: {target}")

    try:
        next(target.iterdir())
    except StopIteration:
        try:
            target.rmdir()
        except OSError:
            return False
        _remove_empty_parents(target.parent, stop=worktrees_root)
        return True

    status = runner(_at(target, ["status", "--porcelain"]), check=False)
    if status.returncode == 0:
        if status.stdout.strip():
            raise RuntimeError(f"leftover worktree directory is dirty: {target}")
    elif branch is None or not _leftover_matches_branch(
        target, branch, root=root, runner=runner
    ):
        raise RuntimeError(
            f"leftover worktree directory is dirty/untracked or cannot be verified clean: {target}"
        )

    try:
        shutil.rmtree(target)
    except OSError:
        return False
    _remove_empty_parents(target.parent, stop=worktrees_root)
    return True


def _write_patch_file(patch: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".patch",
        delete=False,
    )
    try:
        handle.write(patch)
        return Path(handle.name)
    finally:
        handle.close()


def start_slice(
    branch: str,
    *,
    remote: str = "origin",
    base: str = "main",
    carry: bool = False,
    primary_root: Path | None = None,
    runner: Runner = _run_git,
) -> Path:
    validate_branch_name(branch)
    root = (primary_root or paths.primary_root()).resolve()
    target = root / "worktrees" / branch
    if target.exists():
        raise RuntimeError(f"worktree path already exists: {target}")
    if _ref_exists(f"refs/heads/{branch}", root=root, runner=runner):
        raise RuntimeError(f"local branch already exists: {branch}")

    _checked(
        runner(_at(root, ["fetch", "--prune", remote]), check=False),
        f"fetch {remote}",
    )
    if _remote_branch_exists(branch, remote=remote, root=root, runner=runner):
        raise RuntimeError(f"remote branch already exists: {remote}/{branch}")
    if not _ref_exists(f"refs/remotes/{remote}/{base}", root=root, runner=runner):
        raise RuntimeError(f"base branch is unavailable after fetch: {remote}/{base}")

    patch_path: Path | None = None
    if carry:
        current = _checked(
            runner(_at(root, ["branch", "--show-current"]), check=False),
            "read primary checkout branch",
        )
        if current != base:
            raise RuntimeError(
                f"--carry requires the primary checkout on {base}; current branch is {current!r}"
            )
        untracked = _checked(
            runner(
                _at(root, ["ls-files", "--others", "--exclude-standard"]),
                check=False,
            ),
            "read untracked primary files",
        )
        if untracked:
            raise RuntimeError(
                "--carry refuses untracked files; add/commit/remove them first: "
                + ", ".join(untracked.splitlines())
            )
        diff = runner(_at(root, ["diff", "--binary", "HEAD"]), check=False)
        patch = _checked(diff, "capture primary changes") if diff.returncode else diff.stdout
        if patch:
            patch_path = _write_patch_file(patch)
            _checked(
                runner(_at(root, ["reset", "--hard", "HEAD"]), check=False),
                "clean primary checkout for --carry",
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        _checked(
            runner(
                _at(
                    root,
                    ["worktree", "add", "-b", branch, str(target), f"{remote}/{base}"],
                ),
                check=False,
            ),
            f"create worktree for {branch}",
        )
        if patch_path is not None:
            _checked(
                runner(
                    _at(target, ["apply", "--whitespace=nowarn", str(patch_path)]),
                    check=False,
                ),
                f"apply carried changes in {branch}",
            )
    except Exception as exc:
        worktree = _worktree_for(branch, root=root, runner=runner)
        if worktree is not None:
            runner(
                _at(root, ["worktree", "remove", str(worktree.path)]),
                check=False,
            )
        if _ref_exists(f"refs/heads/{branch}", root=root, runner=runner):
            runner(
                _at(root, ["update-ref", "-d", f"refs/heads/{branch}"]),
                check=False,
            )
        _remove_leftover_worktree_dir(target, root=root, branch=branch, runner=runner)
        if patch_path is not None:
            restore = runner(
                _at(root, ["apply", "--whitespace=nowarn", str(patch_path)]),
                check=False,
            )
            if restore.returncode != 0:
                raise RuntimeError(
                    f"{exc}; additionally failed to restore primary changes: "
                    f"{restore.stderr.strip() or restore.stdout.strip()}"
                ) from exc
        raise
    finally:
        if patch_path is not None:
            patch_path.unlink(missing_ok=True)

    print(target)
    return target


def _pr_for_branch(
    branch: str,
    *,
    gh_runner: Runner = _run_gh,
    required: bool = True,
) -> PullRequest | None:
    result = gh_runner(
        [
            "pr",
            "list",
            "--head",
            branch,
            "--state",
            "all",
            "--limit",
            "1",
            "--json",
            "number,state,mergedAt",
        ],
        check=False,
    )
    raw = _checked(result, f"gh pr list --head {branch}")
    try:
        rows = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gh returned invalid JSON for {branch}: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        if required:
            raise RuntimeError(f"no pull request found for branch {branch}")
        return None
    row = rows[0]
    return PullRequest(
        number=int(row["number"]),
        state=str(row["state"]).upper(),
        merged_at=row.get("mergedAt"),
    )


def _require_primary_main_clean(
    root: Path, *, base: str, runner: Runner = _run_git
) -> None:
    current = _checked(
        runner(_at(root, ["branch", "--show-current"]), check=False),
        "read primary checkout branch",
    )
    if current != base:
        raise RuntimeError(
            f"primary checkout must stay on {base}; current branch is {current!r}"
        )
    dirty = _checked(
        runner(_at(root, ["status", "--porcelain"]), check=False),
        "read primary checkout status",
    )
    if dirty:
        raise RuntimeError("primary checkout is dirty; refusing slice finish")


def finish_slice(
    branch: str,
    *,
    remote: str = "origin",
    base: str = "main",
    primary_root: Path | None = None,
    runner: Runner = _run_git,
    gh_runner: Runner = _run_gh,
) -> PullRequest | None:
    validate_branch_name(branch)
    root = (primary_root or paths.primary_root()).resolve()
    target = root / "worktrees" / branch
    _require_primary_main_clean(root, base=base, runner=runner)

    pr = _pr_for_branch(branch, gh_runner=gh_runner, required=False)
    if pr is not None and pr.state not in {"CLOSED", "MERGED"}:
        raise RuntimeError(
            f"PR #{pr.number} for {branch} is {pr.state}; finish requires merged or closed"
        )

    _checked(
        runner(_at(root, ["fetch", "--prune", remote]), check=False),
        f"fetch {remote}",
    )
    if not _ref_exists(f"refs/remotes/{remote}/{base}", root=root, runner=runner):
        raise RuntimeError(f"base branch is unavailable after fetch: {remote}/{base}")
    _checked(
        runner(_at(root, ["merge", "--ff-only", f"{remote}/{base}"]), check=False),
        f"fast-forward {base}",
    )

    local_exists = _ref_exists(f"refs/heads/{branch}", root=root, runner=runner)
    merged = (
        local_exists
        and _branch_merged(
            branch,
            remote=remote,
            base=base,
            root=root,
            runner=runner,
        )
    )
    eligible = merged or (pr is not None and pr.state in {"CLOSED", "MERGED"})
    if not eligible:
        raise RuntimeError(
            f"{branch} is neither merged into {remote}/{base} nor backed by a closed/merged PR"
        )

    worktree = _worktree_for(branch, root=root, runner=runner)
    if worktree is not None:
        dirty = _checked(
            runner(_at(worktree.path, ["status", "--porcelain"]), check=False),
            f"read worktree status for {branch}",
        )
        if dirty:
            raise RuntimeError(f"worktree is dirty: {worktree.path}")

        removed = runner(
            _at(root, ["worktree", "remove", str(worktree.path)]),
            check=False,
        )
        if removed.returncode != 0:
            # Windows can drop the worktree record before directory deletion
            # fails because another process has its cwd inside the path.
            if _worktree_for(branch, root=root, runner=runner) is not None:
                raise RuntimeError(
                    f"remove worktree for {branch}: "
                    f"{removed.stderr.strip() or removed.stdout.strip() or 'command failed'}"
                )

    if not _remove_leftover_worktree_dir(
        target, root=root, branch=branch, runner=runner
    ):
        raise RuntimeError(
            f"leftover worktree directory could not be removed; retry after releasing it: {target}"
        )

    if _ref_exists(f"refs/heads/{branch}", root=root, runner=runner):
        _checked(
            runner(
                _at(root, ["update-ref", "-d", f"refs/heads/{branch}"]),
                check=False,
            ),
            f"delete local branch {branch}",
        )
    if _remote_branch_exists(branch, remote=remote, root=root, runner=runner):
        _checked(
            runner(_at(root, ["push", remote, "--delete", branch]), check=False),
            f"delete remote branch {remote}/{branch}",
        )

    _checked(
        runner(_at(root, ["worktree", "prune"]), check=False),
        "prune worktree metadata",
    )
    _remove_empty_parents(target.parent, stop=root / "worktrees")

    if worktree is None and not local_exists and not target.exists():
        print(f"already finished {branch}")
    elif pr is not None:
        print(f"finished {branch} via PR #{pr.number} ({pr.state.lower()})")
    else:
        print(f"finished merged branch {branch}")
    return pr


def _remote_branches(
    remote: str,
    *,
    root: Path | None = None,
    runner: Runner = _run_git,
) -> list[RemoteBranch]:
    result = runner(
        _at(
            root,
            [
                "for-each-ref",
                "--format=%(refname:strip=3)%09%(objectname)%09%(committerdate:unix)%09%(authorname)",
                f"refs/remotes/{remote}",
            ],
        )
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


def _is_merged(
    branch: RemoteBranch,
    remote: str,
    base: str,
    *,
    root: Path | None = None,
    runner: Runner = _run_git,
) -> bool:
    result = runner(
        _at(root, ["merge-base", "--is-ancestor", branch.sha, f"{remote}/{base}"]),
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git merge-base failed for {branch.name}: {result.stderr.strip()}"
        )
    return result.returncode == 0


def _skip_branch(name: str, base: str) -> bool:
    return name == base or name == "HEAD" or name.startswith("release-please--")


def prune_remote_branches(
    *,
    remote: str = "origin",
    base: str = "main",
    stale_days: int = 14,
    apply: bool = False,
    now: int | None = None,
    root: Path | None = None,
    runner: Runner = _run_git,
) -> dict[str, list[str]]:
    """List merged/stale remote branches; delete merged branches only with apply."""

    if stale_days < 0:
        raise ValueError("stale_days must be non-negative")

    runner(_at(root, ["fetch", "--prune", remote]))
    runner(_at(root, ["rev-parse", "--verify", f"{remote}/{base}"]))

    current = int(time.time()) if now is None else int(now)
    stale_cutoff = current - stale_days * 24 * 60 * 60
    merged: list[str] = []
    deleted: list[str] = []
    stale_unmerged: list[str] = []

    for branch in _remote_branches(remote, root=root, runner=runner):
        if _skip_branch(branch.name, base):
            continue
        if _is_merged(branch, remote, base, root=root, runner=runner):
            merged.append(branch.name)
            if apply:
                runner(_at(root, ["push", remote, "--delete", branch.name]))
                deleted.append(branch.name)
            continue
        if branch.committed_at <= stale_cutoff:
            stale_unmerged.append(branch.name)
            age_days = max(0, (current - branch.committed_at) // (24 * 60 * 60))
            print(
                f"stale-unmerged {branch.name} age={age_days}d author={branch.author}"
            )

    for name in merged:
        action = "deleted" if apply else "would-delete"
        print(f"merged {name} {action}")

    return {
        "merged": merged,
        "deleted": deleted,
        "stale_unmerged": stale_unmerged,
    }


def _branch_merged(
    branch: str,
    *,
    remote: str,
    base: str,
    root: Path,
    runner: Runner = _run_git,
) -> bool:
    result = runner(
        _at(root, ["merge-base", "--is-ancestor", branch, f"{remote}/{base}"]),
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git merge-base failed for {branch}: {result.stderr.strip()}"
        )
    return result.returncode == 0


def prune_orphaned_worktrees(
    *,
    remote: str = "origin",
    base: str = "main",
    apply: bool = False,
    primary_root: Path | None = None,
    runner: Runner = _run_git,
    gh_runner: Runner = _run_gh,
) -> dict[str, list[str]]:
    root = (primary_root or paths.primary_root()).resolve()
    eligible: list[str] = []
    removed: list[str] = []
    dirty: list[str] = []

    for worktree in _worktrees(root=root, runner=runner):
        if worktree.path == root or worktree.branch in (None, base):
            continue
        branch = worktree.branch
        local_exists = _ref_exists(f"refs/heads/{branch}", root=root, runner=runner)
        if local_exists:
            # Squash-merged branches are not ancestors of origin/main. The
            # PR verdict is authoritative for clean-up, not Git ancestry.
            # A never-pushed branch with no closed/merged PR is still active
            # even when its HEAD equals main; never discard it.
            try:
                pr = _pr_for_branch(branch, gh_runner=gh_runner)
            except RuntimeError:
                continue
            if pr.state not in {"CLOSED", "MERGED"}:
                continue

        eligible.append(branch)
        status = _checked(
            runner(_at(worktree.path, ["status", "--porcelain"]), check=False),
            f"read worktree status for {branch}",
        )
        if status:
            dirty.append(branch)
            print(f"orphan-worktree {branch} dirty-skip {worktree.path}")
            continue
        action = "would-remove"
        if apply:
            _checked(
                runner(
                    _at(root, ["worktree", "remove", str(worktree.path)]),
                    check=False,
                ),
                f"remove orphaned worktree {branch}",
            )
            if _ref_exists(f"refs/heads/{branch}", root=root, runner=runner):
                _checked(
                    runner(
                        _at(root, ["update-ref", "-d", f"refs/heads/{branch}"]),
                        check=False,
                    ),
                    f"delete orphaned local branch {branch}",
                )
            removed.append(branch)
            action = "removed"
        print(f"orphan-worktree {branch} {action} {worktree.path}")

    return {"eligible": eligible, "removed": removed, "dirty": dirty}


def slice_status(
    *,
    remote: str = "origin",
    base: str = "main",
    primary_root: Path | None = None,
    runner: Runner = _run_git,
    gh_runner: Runner = _run_gh,
) -> list[dict[str, object]]:
    root = (primary_root or paths.primary_root()).resolve()
    gh = gh_runner(
        [
            "pr",
            "list",
            "--state",
            "all",
            "--limit",
            "100",
            "--json",
            "number,state,mergedAt,headRefName",
        ],
        check=False,
    )
    raw = _checked(gh, "gh pr list")
    try:
        pr_rows = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gh returned invalid JSON: {exc}") from exc
    prs = {
        str(row.get("headRefName")): row
        for row in pr_rows
        if isinstance(row, dict) and row.get("headRefName")
    }

    rows: list[dict[str, object]] = []
    for worktree in _worktrees(root=root, runner=runner):
        branch = worktree.branch
        dirty = bool(
            _checked(
                runner(_at(worktree.path, ["status", "--porcelain"]), check=False),
                f"read worktree status for {worktree.path}",
            )
        )
        ahead: int | str = "-"
        behind: int | str = "-"
        if branch:
            result = runner(
                _at(
                    root,
                    [
                        "rev-list",
                        "--left-right",
                        "--count",
                        f"{remote}/{base}...{branch}",
                    ],
                ),
                check=False,
            )
            if result.returncode == 0:
                fields = result.stdout.strip().split()
                if len(fields) == 2:
                    behind, ahead = int(fields[0]), int(fields[1])
        pr = prs.get(branch or "")
        state = "-"
        number: int | str = "-"
        if pr:
            number = int(pr["number"])
            state = "MERGED" if pr.get("mergedAt") else str(pr.get("state", "-")).upper()
        rows.append(
            {
                "worktree": str(worktree.path),
                "branch": branch or "(detached)",
                "pr": number,
                "state": state,
                "ahead": ahead,
                "behind": behind,
                "dirty": dirty,
            }
        )

    print("WORKTREE\tBRANCH\tPR\tSTATE\tAHEAD\tBEHIND\tDIRTY")
    for row in rows:
        print(
            f"{row['worktree']}\t{row['branch']}\t{row['pr']}\t{row['state']}\t"
            f"{row['ahead']}\t{row['behind']}\t{'yes' if row['dirty'] else 'no'}"
        )
    return rows


def cmd_slice_start(args) -> int:
    try:
        start_slice(
            args.branch,
            remote=args.remote,
            base=args.base,
            carry=args.carry,
        )
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        print(f"slice start: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_slice_finish(args) -> int:
    try:
        finish_slice(args.branch, remote=args.remote, base=args.base)
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        print(f"slice finish: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_slice_status(args) -> int:
    try:
        slice_status(remote=args.remote, base=args.base)
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        print(f"slice status: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_slice_prune(args) -> int:
    try:
        root = paths.primary_root()
        remote_result = prune_remote_branches(
            remote=args.remote,
            base=args.base,
            stale_days=args.stale_days,
            apply=args.apply,
            root=root,
        )
        worktree_result = prune_orphaned_worktrees(
            remote=args.remote,
            base=args.base,
            apply=args.apply,
            primary_root=root,
        )
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        print(f"slice prune: {exc}", file=sys.stderr)
        return 1
    print(
        "slice-prune: "
        f"merged={len(remote_result['merged'])} "
        f"deleted={len(remote_result['deleted'])} "
        f"stale-unmerged={len(remote_result['stale_unmerged'])} "
        f"orphaned={len(worktree_result['eligible'])} "
        f"removed={len(worktree_result['removed'])} "
        f"dirty={len(worktree_result['dirty'])}"
    )
    return 0
