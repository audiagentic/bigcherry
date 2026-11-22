"""Reconcile a release PR with main when only the shared release-please manifest conflicts.

All release lines share `.release-please-manifest.json`. After one line releases, main's manifest moved and every other
open release PR (whose base is older) conflicts in that one file. The resolution is mechanical: the package the PR
releases keeps the version from the PR head, every other package takes the version from main.

`merge_manifest_text` is that rule, pure. `reconcile_head` applies it to a PR head in a temporary worktree and pushes a
plain merge commit (never forced) only while the head is still the commit the caller inspected. CLI (CI):
`PYTHONPATH=tools python3 -m bigcherry.release.manifest_reconcile <repo> <remote>` with GH_TOKEN set (the
remote URL carries it). It never merges a pull request.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

MANIFEST = ".release-please-manifest.json"
CONFIG = "release-please-config.json"
LABEL = "autorelease: pending"
_ROOT_PACKAGE = "."

#: git runner: (cwd, args) -> (returncode, stdout, stderr). Tests inject a fake.
Git = Callable[[Path, Sequence[str]], tuple[int, str, str]]


class ManifestReconcileError(RuntimeError):
    """The manifest (or the merge) cannot be resolved by the ownership rule; nothing was changed."""


def _version(text: object, key: str) -> tuple[int, ...]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", text) if isinstance(text, str) else None
    if match is None:
        raise ManifestReconcileError(f"{MANIFEST} key {key!r} has version {text!r}, not major.minor.patch")
    return tuple(int(part) for part in match.groups())


def merge_manifest_text(head_text: str, main_text: str, package: str) -> str:
    """The manifest with ``package`` from the PR head and every other package from main, formatting preserved.

    Refuses when a side is not a JSON object of version strings, the key sets differ, ``package`` is missing, or the
    result would take any package below its version on either side.
    """
    try:
        head, main = json.loads(head_text), json.loads(main_text)
    except json.JSONDecodeError as exc:
        raise ManifestReconcileError(f"cannot resolve {MANIFEST}: a side is not valid JSON") from exc
    if not isinstance(head, dict) or not isinstance(main, dict):
        raise ManifestReconcileError(f"cannot resolve {MANIFEST}: both sides must be objects")
    if package not in head or package not in main:
        raise ManifestReconcileError(f"cannot resolve {MANIFEST}: package {package!r} is missing on a side")
    if set(head) != set(main):
        raise ManifestReconcileError(
            f"cannot resolve {MANIFEST}: package keys differ between the PR head and main; refusing to invent them")

    desired = dict(main)
    desired[package] = head[package]
    for key, value in desired.items():
        chosen = _version(value, key)
        if chosen < _version(head[key], key) or chosen < _version(main[key], key):
            raise ManifestReconcileError(
                f"cannot resolve {MANIFEST}: {key!r} would go down to {value!r} "
                f"(PR head {head[key]!r}, main {main[key]!r})")

    updated = head_text
    for key, value in desired.items():
        pattern = re.compile(
            rf'(?m)^([ \t]*){re.escape(json.dumps(key))}([ \t]*:[ \t]*)'
            r'(?P<value>"(?:\\.|[^"\\])*")(?P<tail>[ \t]*,?[ \t]*(?:\r?\n|$))')
        match = pattern.search(updated)
        if match is None:
            raise ManifestReconcileError(f"cannot preserve formatting while resolving {MANIFEST} key {key!r}")
        updated = updated[:match.start("value")] + json.dumps(value) + updated[match.end("value"):]
    try:
        resolved = json.loads(updated)
    except json.JSONDecodeError as exc:
        raise ManifestReconcileError(f"resolved {MANIFEST} is invalid JSON") from exc
    if resolved != desired:
        raise ManifestReconcileError(f"resolved {MANIFEST} does not match the ownership rule")
    return updated


def system_git(cwd: Path, args: Sequence[str]) -> tuple[int, str, str]:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=600)
    return done.returncode, done.stdout, done.stderr


def _ok(git: Git, cwd: Path, *args: str) -> str:
    code, out, err = git(cwd, args)
    if code != 0:
        raise ManifestReconcileError(f"git {args[0]} failed: {(err or out).strip()[-300:]}")
    return out


def reconcile_head(repo: Path, remote: str, base: str, head: str, expected_sha: str, package: str,
                   git: Git = system_git) -> str:
    """Merge ``base`` into the PR branch ``head``; returns "up-to-date", "head-moved" or "pushed".

    ``remote`` is a remote name or URL (it must not carry credentials). ``expected_sha`` is the head the caller
    inspected. The merge is always built with --no-commit and its manifest always comes from the pure rule applied to the
    two sides' manifests, whether git merged it cleanly or not; any other unmerged file refuses. The push is a
    compare-and-swap on the expected old head (`--force-with-lease=<ref>:<expected>`, checked by the server when it
    applies the update), used only on a commit that descends from both the expected head and main, so it can only
    fast-forward that exact head; a branch that moved meanwhile, even to an ancestor of the candidate, is rejected and
    reported as "head-moved". Raises ManifestReconcileError when the merge cannot be resolved by the rule.
    """
    base_ref, head_ref = "refs/reconcile/base", "refs/reconcile/head"
    _ok(git, repo, "fetch", "-q", remote, f"+refs/heads/{base}:{base_ref}", f"+refs/heads/{head}:{head_ref}")
    if _ok(git, repo, "rev-parse", head_ref).strip() != expected_sha:
        return "head-moved"
    if git(repo, ["merge-base", "--is-ancestor", base_ref, head_ref])[0] == 0:
        return "up-to-date"

    parent = Path(tempfile.mkdtemp(prefix="release-manifest-reconcile-", dir=repo.parent))
    tree = parent / "tree"
    try:
        _ok(git, repo, "worktree", "add", "-q", "--detach", str(tree), head_ref)
        git(tree, ["merge", "--no-commit", "--no-ff", base_ref])  # exit status is read from the index below
        try:
            unmerged = [line for line in _ok(git, tree, "diff", "--name-only", "--diff-filter=U").splitlines() if line]
            others = sorted(path for path in unmerged if path != MANIFEST)
            if others:
                raise ManifestReconcileError(
                    f"{head} conflicts with {base} in {', '.join(others)}; only {MANIFEST} is resolved automatically")
            resolved = merge_manifest_text(
                _ok(git, tree, "show", f"{head_ref}:{MANIFEST}"), _ok(git, tree, "show", f"{base_ref}:{MANIFEST}"), package)
            (tree / MANIFEST).write_text(resolved, encoding="utf-8", newline="")
            _ok(git, tree, "add", MANIFEST)
            if _ok(git, tree, "diff", "--name-only", "--diff-filter=U").strip():
                raise ManifestReconcileError("unresolved paths remain after the manifest was resolved")
            _ok(git, tree, "commit", "-q", "-m", f"merge {base} into {head}")
        except ManifestReconcileError:
            git(tree, ["merge", "--abort"])
            raise
        pushed = git(tree, ["push", "-q", f"--force-with-lease=refs/heads/{head}:{expected_sha}", remote,
                            f"HEAD:refs/heads/{head}"])
        return "pushed" if pushed[0] == 0 else "head-moved"
    finally:
        git(repo, ["worktree", "remove", "--force", str(tree)])
        shutil.rmtree(parent, ignore_errors=True)


BASE = "main"
#: how long a PR's mergeability may stay UNKNOWN (GitHub computes it lazily), and how often a moved head is retried
POLL_ATTEMPTS = 8
POLL_SECONDS = 15


def platform_target(config: dict) -> tuple[str, str]:
    """(package path, component) of the platform line: the package at the repository root."""
    entry = (config.get("packages") or {}).get(_ROOT_PACKAGE)
    if not isinstance(entry, dict) or not entry.get("component"):
        raise ManifestReconcileError(f"{CONFIG} has no component for the root package")
    return _ROOT_PACKAGE, str(entry["component"])


def canonical_branch(component: str) -> str:
    """The head branch release-please uses for one component's PR."""
    return f"release-please--branches--{BASE}--components--{component}"


def refusal(pr: dict, branch: str) -> str | None:
    """Why a PR is not the open, labelled platform release PR (None: it is)."""
    labels = {str(item.get("name")) for item in pr.get("labels") or [] if isinstance(item, dict)}
    checks = (
        (str(pr.get("state")).upper() == "OPEN", "not open"),
        (pr.get("baseRefName") == BASE, f"base is not {BASE}"),
        (LABEL in labels, f"missing label {LABEL!r}"),
        (pr.get("headRefName") == branch, "head branch is not the canonical release branch"),
    )
    return next((why for ok, why in checks if not ok), None)


_FIELDS = "number,state,baseRefName,labels,headRefName,headRefOid,mergeable,mergeStateStatus"
GhJson = Callable[..., object]


def _gh_json(*args: str) -> object:
    done = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise ManifestReconcileError(f"gh {args[0]} {args[1]} failed: {done.stderr.strip()[-300:]}")
    return json.loads(done.stdout or "null")


def reconcile_platform(repo: Path, remote: str, config: dict, gh_json: GhJson = _gh_json,
                       git: Git = system_git, sleep: Callable[[float], None] = time.sleep,
                       attempts: int = POLL_ATTEMPTS, seconds: float = POLL_SECONDS) -> list[str]:
    """One report line per candidate: the platform release PR is reconciled, left alone, refused or deferred.

    Candidates are listed by the canonical branch; each is re-read by number before a write, so the head SHA written
    against is the one just read. Deferred states (``deferred-mergeability-unknown``, ``deferred-head-moved``) leave the PR untouched; running
    the workflow again retries.
    """
    package, component = platform_target(config)
    branch = canonical_branch(component)
    listed = gh_json("pr", "list", "--state", "open", "--label", LABEL, "--base", BASE, "--head", branch, "--json", _FIELDS)
    report: list[str] = []
    for item in listed or []:
        number, line = item["number"], None
        if refusal(item, branch):
            continue
        for _ in range(attempts):
            pr = gh_json("pr", "view", str(number), "--json", _FIELDS)
            why = refusal(pr, branch)
            if why:
                line = f"PR #{number}: refused: {why}"
                break
            if str(pr["mergeable"]).upper() == "UNKNOWN":
                line = f"PR #{number}: deferred-mergeability-unknown"
                sleep(seconds)
                continue
            if str(pr["mergeable"]).upper() != "CONFLICTING" and str(pr["mergeStateStatus"]).upper() != "DIRTY":
                line = f"PR #{number}: no conflict, nothing to do"
                break
            try:
                result = reconcile_head(repo, remote, BASE, branch, pr["headRefOid"], package, git)
            except ManifestReconcileError as exc:
                line = f"PR #{number}: not reconciled: {exc}"
                break
            line = f"PR #{number}: {result}" if result != "head-moved" else f"PR #{number}: deferred-head-moved"
            if result != "head-moved":
                break
            sleep(seconds)
        report.append(line or f"PR #{number}: deferred-no-attempt")
    return report


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: manifest_reconcile <repo> <remote-url>", file=sys.stderr)
        return 2
    repo = Path(args[0]).resolve()
    config = json.loads((repo / CONFIG).read_text(encoding="utf-8"))
    for line in reconcile_platform(repo, args[1], config):
        # a refusal or deferral is a warning, never a failed release workflow; the PR stays as it was
        print(f"::warning::{line}" if ("refused" in line or "deferred" in line or "not reconciled" in line) else line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
