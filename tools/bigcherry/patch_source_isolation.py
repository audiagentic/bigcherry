"""HI82: content-addressed source isolation for patch-validation campaigns.

Fixes a real contamination risk found this session: patch_validation_
campaign.py's --build-root reuses ONE mutable vendor/llama.cpp git checkout
across sequential patch applications with no provenance check. Applying
patch A then patch B to the same tree silently composes both patches into
one build rather than testing B in isolation -- this session did exactly
that (1204 RD08 then 1002 unsafe-math, then 1005, then 1205/1206) and only
avoided a real false-attribution result because none of those patches
happened to share touched files. That must not be the ongoing pattern.

Design: GPT (gpt-auto-agent), requests req_51838ef1ea5f4086 (design +
git_worktree_tree/stable_digest/source_identity sketch),
req_2c85f337296e4c30 and req_17ab02fd3af2449a (implementation, both
truncated mid-response by the chat interface -- the identity/key/path
helpers below are GPT's text verbatim; materialize_source/
materialize_stock_source are written directly against GPT's documented
contract from the design review rather than waiting on a third truncated
attempt). Applied per plan item HI82.

Each materialized source lives at <worktree_root>/<source_key>/, a real
`git worktree add` checkout (never the shared vendor/llama.cpp working
tree), with a manifest.json recording the git tree hash the CONTENT was
at when this module last touched it. Reuse is only permitted when the
on-disk tree hash still matches the manifest -- any mutation after the
fact (by this module's own bug, an operator, or another process) is
detected and rejected rather than silently trusted.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHES_ROOT = REPO_ROOT / "patches"
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1


class PatchSourceIsolationError(RuntimeError):
    """Raised when a patch-validation source cannot be materialized safely."""


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def framework_baseline_digest() -> str:
    """sha256 over the sorted bytes of every currently-validated patch file.

    Folded into materialize_source()'s identity so a worktree materialized
    against an OLDER framework-validated patchset is correctly treated as
    stale (different source_key) once the baseline changes -- otherwise a
    patch validated today could silently keep reusing an isolated worktree
    that no longer reflects what `bigcherry apply` would actually produce.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from bigcherry import patchset  # noqa: E402  (path inserted above)

    hasher = hashlib.sha256()
    for path in sorted(patchset.discover_modules(PATCHES_ROOT)):
        if patchset.module_state(path) != "validated":
            continue
        hasher.update(str(path.relative_to(PATCHES_ROOT)).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _make_source_identity(
    base_revision: str, patch_name: str, implementation_digest: str,
) -> dict[str, str]:
    payload = {
        "schema": "bigcherry-patch-source-v1",
        "base_revision": base_revision,
        "framework_baseline_digest": framework_baseline_digest(),
        "patch_name": patch_name,
        "patch_implementation_digest": implementation_digest,
    }
    return {**payload, "source_key": _stable_digest(payload)}


def source_key(base_revision: str, patch_name: str, implementation_digest: str) -> str:
    """Return the content-addressed key for one materialized source identity."""
    return _make_source_identity(
        base_revision=base_revision, patch_name=patch_name,
        implementation_digest=implementation_digest,
    )["source_key"]


def _stock_source_identity(base_revision: str) -> dict[str, str]:
    payload = {
        "schema": "bigcherry-patch-source-v1",
        "base_revision": base_revision,
        "patch_name": None,
        "patch_implementation_digest": None,
    }
    return {**payload, "source_key": _stable_digest(payload)}


def _patch_path(patch_name: str) -> Path:
    """Resolve a flat patches/<name>.py module without permitting path escape."""
    if not patch_name:
        raise PatchSourceIsolationError("patch name must not be empty")

    if patch_name.endswith(".py"):
        patch_name = patch_name[:-3]

    if (
        "/" in patch_name or "\\" in patch_name
        or patch_name in {".", ".."} or ".." in Path(patch_name).parts
    ):
        raise PatchSourceIsolationError(
            f"patch name must be a flat module name under {PATCHES_ROOT}: {patch_name!r}"
        )

    root = PATCHES_ROOT.resolve()
    candidate = (PATCHES_ROOT / f"{patch_name}.py").resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PatchSourceIsolationError(
            f"patch module escapes patches directory: {patch_name!r}"
        ) from exc

    if not candidate.is_file():
        raise PatchSourceIsolationError(f"patch module does not exist: {candidate}")

    return candidate


def patch_implementation_digest(patch_name: str) -> str:
    """sha256 of the patch module's own bytes.

    Deliberately does NOT also hash tools/bigcherry/patcher.py (the shared
    edit engine) -- that would force a rebuild on any unrelated patcher.py
    docstring tweak. A stricter version could add it if patcher.py's own
    edit semantics start changing in ways that matter; note the tradeoff
    rather than silently picking one side of it.
    """
    return hashlib.sha256(_patch_path(patch_name).read_bytes()).hexdigest()


def _run(argv: list[str], *, cwd: Path, check: bool = True) -> str:
    result = subprocess.run(
        argv, cwd=cwd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    if check and result.returncode != 0:
        raise PatchSourceIsolationError(
            f"{' '.join(argv)} (cwd={cwd}) failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def git_head(repo: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=repo)


def git_head_tree(repo: Path) -> str:
    return _run(["git", "rev-parse", "HEAD^{tree}"], cwd=repo)


def git_worktree_tree(repo: Path) -> str:
    """The real git tree hash of the current working-tree CONTENT.

    Uses a temporary index (GIT_INDEX_FILE) so this never stages or
    modifies the repo's own index -- `git add -A` here only affects the
    throwaway index, not the operator's real staging area. Captures
    tracked modifications, deletions, and untracked non-ignored additions
    (exactly what a patch's edits + any new files it adds would produce).
    Ignored files (build output) are deliberately excluded.
    """
    with tempfile.TemporaryDirectory(prefix="bigcherry-git-index-") as td:
        index = Path(td) / "index"
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(index)
        subprocess.run(
            ["git", "read-tree", "HEAD"], cwd=repo, env=env, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        subprocess.run(
            ["git", "add", "-A", "--", "."], cwd=repo, env=env, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        result = subprocess.run(
            ["git", "write-tree"], cwd=repo, env=env, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        return result.stdout.strip()


def _manifest_path(source_dir: Path) -> Path:
    # A SIBLING of the worktree, not inside it: writing manifest.json into
    # the worktree itself would make it show up as an untracked file the
    # very next time git_worktree_tree() runs, changing the computed hash
    # out from under the manifest that just recorded it (confirmed by a
    # real reuse-path test failing this way before this fix).
    return source_dir.parent / f"{source_dir.name}.{MANIFEST_NAME}"


def _write_manifest(source_dir: Path, manifest: dict[str, Any]) -> None:
    path = _manifest_path(source_dir)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_manifest(source_dir: Path) -> dict[str, Any] | None:
    path = _manifest_path(source_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_reuse(source_dir: Path, expected_identity: dict[str, str]) -> bool:
    """True if source_dir can be reused as-is for expected_identity.

    Fails CLOSED (raises, does not silently return False and let the
    caller quietly rebuild over a suspicious directory) when a manifest
    exists but disagrees with reality -- that state means something wrote
    into this identity's directory without going through this module.
    """
    manifest = _read_manifest(source_dir)
    if manifest is None:
        # No manifest at all in an existing directory is not safely
        # reusable -- treat as "not materialized yet" so the caller
        # rebuilds from scratch rather than trusting an unmanaged tree.
        return False

    # Iterate the identity's own keys (not a hardcoded tuple) so this check
    # naturally covers both the stock identity (no framework_baseline_digest
    # key at all) and the patched identity (which has one) without drifting
    # out of sync if either identity shape changes later.
    for key in expected_identity:
        if manifest.get(key) != expected_identity.get(key):
            raise PatchSourceIsolationError(
                f"source provenance mismatch for {key!r} at {source_dir}: "
                f"manifest={manifest.get(key)!r}, expected={expected_identity.get(key)!r}"
            )

    actual_head = git_head(source_dir)
    if actual_head != manifest.get("head"):
        raise PatchSourceIsolationError(
            f"source HEAD changed after materialization at {source_dir}: "
            f"manifest={manifest.get('head')!r}, actual={actual_head!r}"
        )

    actual_tree = git_worktree_tree(source_dir)
    if actual_tree != manifest.get("patched_tree"):
        raise PatchSourceIsolationError(
            f"patched source tree was modified after materialization at "
            f"{source_dir}: manifest tree={manifest.get('patched_tree')!r}, "
            f"actual tree={actual_tree!r}"
        )

    return True


def _add_worktree(base_repo: Path, worktree_dir: Path, base_revision: str) -> None:
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    if worktree_dir.exists():
        # A directory with no valid manifest (caught above) or a leftover
        # partial worktree from a killed prior run -- prune registrations
        # for missing worktrees, then remove the stale directory outright
        # rather than trying to patch up unknown state.
        _run(["git", "worktree", "prune"], cwd=base_repo, check=False)
        shutil.rmtree(worktree_dir, ignore_errors=True)
    _run(
        ["git", "worktree", "add", "--detach", str(worktree_dir), base_revision],
        cwd=base_repo,
    )


def materialize_source(
    *, base_repo: Path, worktree_root: Path, patch_module: str, base_revision: str,
) -> Path:
    """Return an isolated, patched worktree for exactly this one patch.

    Reuses an existing worktree only when its manifest AND its actual
    on-disk tree hash both match this exact (base_revision, patch_name,
    patch_implementation_digest) identity. Otherwise materializes fresh:
    `git worktree add --detach` at base_revision (never the shared vendor/
    llama.cpp working tree, and never a branch ref that could move under
    it -- --detach pins to the resolved commit), applies patch_module via
    patcher.apply_all, records the resulting tree hash, and returns the
    worktree path.
    """
    digest = patch_implementation_digest(patch_module)
    identity = _make_source_identity(
        base_revision=base_revision, patch_name=patch_module, implementation_digest=digest,
    )
    source_dir = worktree_root / identity["source_key"]

    if source_dir.exists() and _verify_reuse(source_dir, identity):
        return source_dir

    _add_worktree(base_repo, source_dir, base_revision)

    sys.path.insert(0, str(REPO_ROOT))
    from bigcherry import patcher, patchset, paths  # noqa: E402  (path inserted above)

    # A patch under test is validated as it will actually SHIP: on top of
    # the real framework-validated baseline, not bare upstream. Confirmed
    # necessary by a real failure: RD08's own test-backend-ops.cpp anchor
    # only exists after other validated patches (the HI70 direct-op
    # corpus) have already added it -- applying RD08 alone against raw
    # upstream in an isolated worktree failed with a real "anchor matched
    # 0 times" error until this baseline step was added.
    written = []
    for source in sorted(paths.SRC_OVERLAY.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(paths.SRC_OVERLAY)
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8", newline="")
        written.append(str(relative))

    baseline_patches = patchset.load_patches(groups=None, states=frozenset({"validated"}))
    baseline_results = patcher.apply_all(baseline_patches, source_dir, dry_run=False)
    for result in baseline_results:
        if result.failed:
            raise PatchSourceIsolationError(
                f"framework baseline patch failed to apply to {result.path} in "
                f"isolated worktree {source_dir} (before the patch under test "
                f"was even applied): {[r.detail for r in result.failed]}"
            )

    mod = importlib.import_module(f"patches.{patch_module}")
    results = patcher.apply_all(mod.PATCHES, source_dir, dry_run=False)
    for result in results:
        if result.failed:
            raise PatchSourceIsolationError(
                f"patch {patch_module} failed to apply to {result.path} in "
                f"isolated worktree {source_dir}: {[r.detail for r in result.failed]}"
            )

    manifest = {
        **identity,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "head": git_head(source_dir),
        "patched_tree": git_worktree_tree(source_dir),
    }
    _write_manifest(source_dir, manifest)
    return source_dir


def materialize_stock_source(*, base_repo: Path, worktree_root: Path, base_revision: str) -> Path:
    """Same worktree mechanism as materialize_source(), but no patch applied --
    a genuinely pristine baseline at base_revision for stock bench comparison."""
    identity = _stock_source_identity(base_revision)
    source_dir = worktree_root / identity["source_key"]

    if source_dir.exists() and _verify_reuse(source_dir, identity):
        return source_dir

    _add_worktree(base_repo, source_dir, base_revision)

    manifest = {
        **identity,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "head": git_head(source_dir),
        "patched_tree": git_worktree_tree(source_dir),
    }
    _write_manifest(source_dir, manifest)
    return source_dir


def remove_worktree(base_repo: Path, source_dir: Path) -> None:
    """Explicit teardown -- worktrees are cheap to keep around for reuse, but
    callers doing disk-space cleanup should use this rather than a raw
    rmtree so `git worktree list` in base_repo doesn't accumulate stale
    registrations for a directory that no longer exists."""
    if source_dir.exists():
        _run(["git", "worktree", "remove", "--force", str(source_dir)], cwd=base_repo, check=False)
    shutil.rmtree(source_dir, ignore_errors=True)
    _manifest_path(source_dir).unlink(missing_ok=True)
    _run(["git", "worktree", "prune"], cwd=base_repo, check=False)
