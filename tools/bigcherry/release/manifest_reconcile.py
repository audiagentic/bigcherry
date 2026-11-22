"""Reconcile a release PR with main when only the shared release-please manifest conflicts.

All release lines share `.release-please-manifest.json`. After one line releases, main's manifest moved and every other
open release PR (whose base is older) conflicts in that one file. The resolution is mechanical: the package the PR
releases keeps the version from the PR head, every other package takes the version from main.

`merge_manifest_text` is that rule, pure. `reconcile_head` applies it to a PR head in a temporary worktree and pushes a
plain merge commit (never forced) only while the head is still the commit the caller inspected. CLI (CI):
`PYTHONPATH=tools python3 -m bigcherry.release.manifest_reconcile <repo> <remote-url-or-name>` with GH_TOKEN set.
It never merges a pull request.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
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

    ``remote`` is a remote name or URL. ``expected_sha`` is the head the caller inspected: nothing is pushed unless the
    branch is still there, and the push is a plain fast-forward (a concurrent update makes it fail, reported as
    "head-moved"). Raises ManifestReconcileError when anything but the manifest conflicts or the rule refuses.
    """
    if head == base:
        raise ManifestReconcileError(f"refusing to update {base!r} itself")
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
        code, _, _ = git(tree, ["merge", "--no-edit", "-m", f"merge {base} into {head}", base_ref])
        if code != 0:
            conflicts = [line for line in _ok(git, tree, "diff", "--name-only", "--diff-filter=U").splitlines() if line]
            if conflicts != [MANIFEST]:
                git(tree, ["merge", "--abort"])
                raise ManifestReconcileError(
                    f"{head} conflicts with {base} in {', '.join(sorted(conflicts)) or 'unknown files'}; "
                    f"only {MANIFEST} is resolved automatically")
            try:
                resolved = merge_manifest_text(
                    _ok(git, tree, "show", f":2:{MANIFEST}"), _ok(git, tree, "show", f":3:{MANIFEST}"), package)
            except ManifestReconcileError:
                git(tree, ["merge", "--abort"])
                raise
            (tree / MANIFEST).write_text(resolved, encoding="utf-8", newline="")
            _ok(git, tree, "add", MANIFEST)
            _ok(git, tree, "commit", "-q", "--no-edit")
        # still the head we started from? then a plain (non-forced) push is a fast-forward of exactly that commit
        _ok(git, repo, "fetch", "-q", remote, f"+refs/heads/{head}:{head_ref}")
        if _ok(git, repo, "rev-parse", head_ref).strip() != expected_sha:
            return "head-moved"
        if git(tree, ["push", "-q", remote, f"HEAD:refs/heads/{head}"])[0] != 0:
            return "head-moved"
        return "pushed"
    finally:
        git(repo, ["worktree", "remove", "--force", str(tree)])
        shutil.rmtree(parent, ignore_errors=True)


def platform_target(config: dict) -> tuple[str, str]:
    """(package path, component) of the platform line: the package at the repository root."""
    entry = (config.get("packages") or {}).get(_ROOT_PACKAGE)
    if not isinstance(entry, dict) or not entry.get("component"):
        raise ManifestReconcileError(f"{CONFIG} has no component for the root package")
    return _ROOT_PACKAGE, str(entry["component"])


def _gh_json(*args: str) -> object:
    done = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise ManifestReconcileError(f"gh {args[0]} {args[1]} failed: {done.stderr.strip()[-300:]}")
    return json.loads(done.stdout or "null")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: manifest_reconcile <repo> <remote-url-or-name>", file=sys.stderr)
        return 2
    repo, remote = Path(args[0]).resolve(), args[1]
    package, component = platform_target(json.loads((repo / CONFIG).read_text(encoding="utf-8")))
    prs = _gh_json("pr", "list", "--state", "open", "--label", LABEL, "--base", "main",
                   "--json", "number,headRefName,headRefOid,mergeable,mergeStateStatus")
    status = 0
    for pr in prs:
        if not str(pr["headRefName"]).endswith(f"--components--{component}"):
            continue
        if str(pr["mergeable"]).upper() != "CONFLICTING" and str(pr["mergeStateStatus"]).upper() != "DIRTY":
            print(f"PR #{pr['number']}: no conflict, nothing to do")
            continue
        try:
            result = reconcile_head(repo, remote, "main", pr["headRefName"], pr["headRefOid"], package)
        except ManifestReconcileError as exc:
            # never fatal for the release workflow; the PR stays as it was for a human
            print(f"::warning::PR #{pr['number']} not reconciled: {exc}")
            continue
        print(f"PR #{pr['number']}: {result}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
