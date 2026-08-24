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
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

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


def _patch_registry():
    """The registry module, importable whether this file is loaded as a
    package member or standalone (its existing pattern: REPO_ROOT-derived
    paths + function-level import)."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from bigcherry import patch_registry  # noqa: E402
    return patch_registry


def framework_baseline_digest() -> str:
    """sha256 over the sorted bytes of every currently-validated patch file.

    Folded into materialize_source()'s identity so a worktree materialized
    against an OLDER framework-validated patchset is correctly treated as
    stale (different source_key) once the baseline changes -- otherwise a
    patch validated today could silently keep reusing an isolated worktree
    that no longer reflects what `bigcherry apply` would actually produce.

    RS05: the validated set is the registry's validated descriptors (root
    relative implementation paths + bytes). For today's flat tree the
    relative paths and bytes are exactly what the old discover_modules() +
    module_state() walk produced, so existing source_key digests are stable.
    """
    registry = _patch_registry().load_registry(PATCHES_ROOT)
    validated = sorted(
        (d for d in registry.descriptors if d.state == "validated"),
        key=lambda d: d.implementation_path.as_posix(),
    )
    hasher = hashlib.sha256()
    for descriptor in validated:
        relative = descriptor.implementation_path.as_posix()
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update((registry.root / descriptor.implementation_path).read_bytes())
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


def patch_implementation_digest(patch_name: str) -> str:
    """sha256 of the patch implementation's own bytes (flat OR packaged).

    RS05: resolved through the registry descriptor -- no flat-only
    f"{name}.py" construction left in this module (the old _patch_path() is
    gone; the registry's own discovery is the path-resolution authority and
    enforces the same escape rules).

    Deliberately does NOT also hash tools/bigcherry/patcher.py (the shared
    edit engine) -- that would force a rebuild on any unrelated patcher.py
    docstring tweak. A stricter version could add it if patcher.py's own
    edit semantics start changing in ways that matter; note the tradeoff
    rather than silently picking one side of it.
    """
    registry = _patch_registry().load_registry(PATCHES_ROOT)
    try:
        descriptor = registry.get(patch_name)
    except Exception as exc:
        raise PatchSourceIsolationError(f"patch module does not exist: {patch_name!r}") from exc
    return hashlib.sha256(
        (registry.root / descriptor.implementation_path).read_bytes()
    ).hexdigest()


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
        # partial worktree from a killed prior run -- remove the stale
        # directory outright rather than trying to patch up unknown state.
        shutil.rmtree(worktree_dir, ignore_errors=True)
    # prune AFTER the rmtree (and even when the directory is already gone):
    # a registration whose directory no longer exists is "missing but
    # already registered" -- a prune that ran BEFORE the directory was
    # removed leaves it in place, and the re-add below then fails with
    # exit 128 (caught by a real RS05 rebuild test).
    _run(["git", "worktree", "prune"], cwd=base_repo, check=False)
    _run(
        ["git", "worktree", "add", "--detach", str(worktree_dir), base_revision],
        cwd=base_repo,
    )


def _apply_baseline_and_stack(
    source_dir: Path, patch_modules: Sequence[str], *, root: Path | None = None
) -> None:
    # Call-time resolution (not a def-time default) so tests can redirect
    # the patches root by monkeypatching PATCHES_ROOT.
    root = root or PATCHES_ROOT
    """Overlay bigcherry's own source additions, apply the validated
    framework baseline, then apply patch_modules IN ORDER.

    A patch under test is validated as it will actually SHIP: on top of the
    real framework-validated baseline, not bare upstream. Confirmed
    necessary by a real failure: RD08's own test-backend-ops.cpp anchor only
    exists after other validated patches (the HI70 direct-op corpus) have
    already added it -- applying RD08 alone against raw upstream in an
    isolated worktree failed with a real "anchor matched 0 times" error
    until this baseline step was added.

    Shared by materialize_source() (single patch) and
    materialize_source_variant() (an ordered stack) so the overlay/baseline/
    apply sequence has exactly one implementation.

    RS05: BOTH the baseline and the stack load through the registry
    (descriptor + byte-compile load_implementation) -- the old
    ``importlib.import_module(f"patches.{module}")`` line is gone, so
    packaged patches materialize the same way flat ones do.
    """
    registry = _patch_registry().load_registry(root)
    from bigcherry import patcher, paths  # noqa: E402

    for source in sorted(paths.SRC_OVERLAY.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(paths.SRC_OVERLAY)
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8", newline="")

    baseline = [
        d for d in registry.descriptors if d.state == "validated"
    ]
    baseline_patches = [
        p for d in baseline for p in _patch_registry().load_implementation(d, root=root)
    ]
    baseline_results = patcher.apply_all(baseline_patches, source_dir, dry_run=False)
    for result in baseline_results:
        if result.failed:
            raise PatchSourceIsolationError(
                f"framework baseline patch failed to apply to {result.path} in "
                f"isolated worktree {source_dir} (before the patch stack was "
                f"even applied): {[r.detail for r in result.failed]}"
            )

    for patch_module in patch_modules:
        try:
            descriptor = registry.get(patch_module)
        except Exception as exc:
            raise PatchSourceIsolationError(
                f"patch module does not exist: {patch_module!r}"
            ) from exc
        implementation = _patch_registry().load_implementation(descriptor, root=root)
        results = patcher.apply_all(implementation, source_dir, dry_run=False)
        for result in results:
            if result.failed:
                raise PatchSourceIsolationError(
                    f"patch {patch_module} failed to apply to {result.path} in "
                    f"isolated worktree {source_dir}: {[r.detail for r in result.failed]}"
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
    _apply_baseline_and_stack(source_dir, (patch_module,))

    manifest = {
        **identity,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "head": git_head(source_dir),
        "patched_tree": git_worktree_tree(source_dir),
    }
    _write_manifest(source_dir, manifest)
    return source_dir


def _make_variant_identity(
    base_revision: str, patch_modules: Sequence[str], variant_name: str, variant_digest: str,
) -> dict[str, Any]:
    payload = {
        "schema": "bigcherry-patch-source-variant-v1",
        "base_revision": base_revision,
        "framework_baseline_digest": framework_baseline_digest(),
        "patch_stack": [
            {"patch": module, "implementation_digest": patch_implementation_digest(module)}
            for module in patch_modules
        ],
        "variant_name": variant_name,
        "variant_digest": variant_digest,
    }
    return {**payload, "source_key": _stable_digest(payload)}


def materialize_source_variant(
    *,
    base_repo: Path,
    worktree_root: Path,
    base_revision: str,
    patch_modules: tuple[str, ...],
    variant_name: str,
    variant_digest: str,
    apply_variant: Callable[[Path], None] | None = None,
) -> Path:
    """Materialize an isolated worktree for an ORDERED patch stack, plus an
    optional post-stack variant transform.

    Generalizes materialize_source() for "subject vs control" evidence,
    where two worktrees need the SAME patch stack but differ by an
    explicitly identified, content-addressed transform applied on top of it
    (e.g. rd08_correctness_evidence.py's VDR1-control / VDR2-subject pair,
    which differ only by two checked source-line reversions applied after
    the identical 1204+1222+1223 stack). `variant_digest` MUST be a stable
    digest over the actual transform content (e.g. sha256 of the ordered
    (path, old, new) triples an `apply_variant` callback performs) -- never
    a bare name -- so two different transforms can never collide on the
    same source_key, and the same transform's own identity is independently
    checkable outside this module.

    Deliberately separate from materialize_source()'s own identity/manifest
    shape (schema "bigcherry-patch-source-v1"): changing that shape to fold
    in a patch_stack list would silently invalidate every already-cached
    single-patch worktree's manifest. Both functions share the actual
    worktree-construction sequence via _apply_baseline_and_stack().
    """
    if not patch_modules:
        raise PatchSourceIsolationError(
            "materialize_source_variant requires at least one patch module"
        )

    identity = _make_variant_identity(
        base_revision=base_revision, patch_modules=patch_modules,
        variant_name=variant_name, variant_digest=variant_digest,
    )
    source_dir = worktree_root / identity["source_key"]

    if source_dir.exists() and _verify_reuse(source_dir, identity):
        return source_dir

    _add_worktree(base_repo, source_dir, base_revision)
    _apply_baseline_and_stack(source_dir, patch_modules)

    if apply_variant is not None:
        apply_variant(source_dir)

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
