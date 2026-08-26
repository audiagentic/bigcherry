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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PATCHES_ROOT = REPO_ROOT / "patches"
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1


class PatchSourceIsolationError(RuntimeError):
    """Raised when a patch-validation source cannot be materialized safely."""


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _patch_registry():
    """The registry module, importable whether this file is loaded as a
    package member or standalone (its existing pattern: REPO_ROOT-derived
    paths + function-level import)."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from bigcherry.patch import registry as patch_registry # noqa: E402

    return patch_registry


def resolve_base_revision(ref: str, *, repo: Path) -> str:
    """Resolve a git ref (branch/tag/HEAD/full-SHA) to the IMMUTABLE commit SHA
    it points at (peels annotated tags via ``^{commit}``). Fails closed on any
    unresolvable ref. The RESOLVED SHA -- not the requested ref -- is what
    enters the source identity and what ``git worktree add --detach`` receives,
    so a moved ref yields a new identity, never a reused stale worktree.
    (RV80/B2: replaces the implicit ``git_head`` + state-scan baseline.)"""
    return _run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=repo)


def _overlay_files(overlay_root: Path | None) -> list[tuple[str, str]]:
    """Deterministic ``(relpath, sha256)`` enumeration of every file under the
    source overlay. Shared by :func:`overlay_digest` and
    :func:`_apply_composition` so the EXACT bytes that enter the identity are
    the EXACT bytes applied. POSIX relative paths, sorted. A missing/absent
    overlay (or ``None``) yields ``[]`` (the empty digest)."""
    if overlay_root is None or not overlay_root.is_dir():
        return []
    entries = []
    for source in sorted(overlay_root.rglob("*")):
        if not source.is_file():
            continue
        relpath = source.relative_to(overlay_root).as_posix()
        entries.append((relpath, hashlib.sha256(source.read_bytes()).hexdigest()))
    return sorted(entries)


def overlay_digest(overlay_root: Path | None) -> str:
    """sha256 over the sorted ``(relpath, sha256)`` of every file under the
    source overlay (bigcherry's ``src/`` additions). State-independent and
    deterministic. ``overlay_root=None`` (stock) hashes the empty set."""
    encoded = json.dumps(_overlay_files(overlay_root), separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def composition_digest(composition: Sequence[tuple[str, str]]) -> str:
    """sha256 over the EXACT ORDERED ``[(patch_id, implementation_digest)]``
    list as supplied (which must be the topologically-ordered output of
    ``patchset.resolve_exact``). We do NOT re-sort: a lexicographic sort could
    give two DIFFERENT application orders the same key when a packaged
    ``patch.toml`` changes requires/order with unchanged ``patch.py`` digests.
    The empty composition (stock) hashes the empty list."""
    encoded = json.dumps(
        [list(entry) for entry in composition],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_source_identity_v2(
    *,
    resolved_revision: str,
    composition: Sequence[tuple[str, str]],
    overlay_root: Path | None,
    variant_name: str | None = None,
    variant_digest: str | None = None,
) -> dict[str, Any]:
    """``bigcherry-patch-source-v2`` (variants: ``-variant-v2``). The payload is
    STATE-INDEPENDENT (RV80/B2): no lifecycle-state scan -- only (a) the
    immutable resolved base SHA, (b) the overlay-bytes digest, (c) the EXPLICIT
    ordered composition, and (d) for variants, the content-addressed transform.
    The requested ref is recorded in the manifest separately and is NOT part of
    the hashed payload."""
    schema = (
        "bigcherry-patch-source-variant-v2"
        if variant_name is not None
        else "bigcherry-patch-source-v2"
    )
    payload = {
        "schema": schema,
        "resolved_revision": resolved_revision,
        "overlay_digest": overlay_digest(overlay_root),
        "composition": [list(entry) for entry in composition],
    }
    if variant_name is not None:
        payload["variant_name"] = variant_name
        payload["variant_digest"] = variant_digest
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
        raise PatchSourceIsolationError(
            f"patch module does not exist: {patch_name!r}"
        ) from exc
    return hashlib.sha256(
        (registry.root / descriptor.implementation_path).read_bytes()
    ).hexdigest()


def _run(argv: list[str], *, cwd: Path, check: bool = True) -> str:
    result = subprocess.run(argv, cwd=cwd, check=check, capture_output=True, text=True)
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
            ["git", "read-tree", "HEAD"],
            cwd=repo,
            env=env,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "add", "-A", "--", "."],
            cwd=repo,
            env=env,
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "write-tree"],
            cwd=repo,
            env=env,
            check=True,
            capture_output=True,
            text=True,
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
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
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


def _apply_composition(
    source_dir: Path,
    composition: Sequence[tuple[str, str]],
    *,
    overlay_root: Path | None,
    root: Path | None = None,
) -> None:
    """Overlay bigcherry's own source additions (EXACTLY the files whose bytes
    are in the identity's overlay_digest), then apply the EXPLICIT composition
    IN THE GIVEN ORDER via the registry loader (B3 digest re-check applies).

    No lifecycle-state scan anywhere: the composition is exactly what the
    caller resolved (RV80/B6). A patch under test is materialized as it will
    actually SHIP: on top of the explicitly resolved named composition, not
    bare upstream (RD08's anchor-dependency failure is why the composition
    step exists at all).
    """
    registry = _patch_registry().load_registry(root or PATCHES_ROOT)
    from bigcherry import patcher  # noqa: E402

    if overlay_root is not None:
        overlay_entries = _overlay_files(overlay_root)
    else:
        overlay_entries = []
    for relative, file_digest in overlay_entries:
        # overlay_entries is non-empty only when overlay_root is present.
        assert overlay_root is not None
        source = overlay_root / relative
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != file_digest:
            raise PatchSourceIsolationError(
                f"overlay file {relative} changed between identity computation "
                f"and application: {actual} != {file_digest}"
            )
        target.write_bytes(data)

    for patch_id, expected_digest in composition:
        try:
            descriptor = registry.get(patch_id)
        except Exception as exc:
            raise PatchSourceIsolationError(
                f"patch module does not exist: {patch_id!r}"
            ) from exc
        actual_digest = hashlib.sha256(
            (registry.root / descriptor.implementation_path).read_bytes()
        ).hexdigest()
        if actual_digest != expected_digest:
            raise PatchSourceIsolationError(
                f"patch {patch_id} implementation changed between identity "
                f"computation and application: {actual_digest} != {expected_digest}"
            )
        implementation = _patch_registry().load_implementation(
            descriptor, root=registry.root
        )
        results = patcher.apply_all(implementation, source_dir, dry_run=False)
        for result in results:
            if result.failed:
                raise PatchSourceIsolationError(
                    f"patch {patch_id} failed to apply to {result.path} in "
                    f"isolated worktree {source_dir}: "
                    f"{[r.detail for r in result.failed]}"
                )


def _materialize_v2(
    *,
    base_repo: Path,
    worktree_root: Path,
    resolved_revision: str,
    composition: Sequence[tuple[str, str]],
    overlay_root: Path | None,
    requested_revision: str | None = None,
    variant_name: str | None = None,
    variant_digest: str | None = None,
    apply_variant: Callable[[Path], None] | None = None,
) -> Path:
    """The single v2 worktree-construction path (RV80/B2).

    `git worktree add --detach` at the RESOLVED SHA (a moved ref is a new
    identity, never a reused stale worktree), overlay + explicit ordered
    composition applied, optional content-addressed variant transform, then a
    state-independent v2 manifest. Reuse is only permitted when manifest AND
    actual on-disk tree hash both match this exact identity.
    """
    identity = _make_source_identity_v2(
        resolved_revision=resolved_revision,
        composition=composition,
        overlay_root=overlay_root,
        variant_name=variant_name,
        variant_digest=variant_digest,
    )
    source_dir = worktree_root / identity["source_key"]

    if source_dir.exists() and _verify_reuse(source_dir, identity):
        return source_dir

    _add_worktree(base_repo, source_dir, resolved_revision)
    _apply_composition(
        source_dir,
        composition,
        overlay_root=overlay_root,
        root=PATCHES_ROOT,
    )

    if apply_variant is not None:
        apply_variant(source_dir)

    manifest = {
        **identity,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "requested_revision": requested_revision,
        "head": git_head(source_dir),
        "patched_tree": git_worktree_tree(source_dir),
    }
    _write_manifest(source_dir, manifest)
    return source_dir


def verify_composition_idempotent(
    *,
    base_repo: Path,
    source: Path,
    worktree_root: Path,
    resolved_revision: str,
    composition: Sequence[tuple[str, str]],
    overlay_root: Path | None = None,
    requested_revision: str | None = None,
) -> bool:
    """Re-run exact materialization and prove it reused the same tree.

    The second invocation must pass the v2 manifest/tree reuse check; equality
    of the resulting path and git tree then proves a no-op reapplication.
    """
    before = git_worktree_tree(source)
    again = materialize_composition(
        base_repo=base_repo,
        worktree_root=worktree_root,
        resolved_revision=resolved_revision,
        composition=composition,
        overlay_root=overlay_root,
        requested_revision=requested_revision,
    )
    return again == source and git_worktree_tree(again) == before


def materialize_composition(
    *,
    base_repo: Path,
    worktree_root: Path,
    resolved_revision: str,
    composition: Sequence[tuple[str, str]],
    overlay_root: Path | None = None,
    requested_revision: str | None = None,
) -> Path:
    """Materialize an isolated worktree for an EXPLICIT composition.

    ``composition`` is the EXACT ORDERED ``[(patch_id, implementation_digest)]``
    list (topologically ordered, e.g. from :func:`resolve_source_composition`)
    -- its entries are re-verified against the live registry before
    application and the whole list (in order) is what enters the v2 source
    identity. An empty composition + ``overlay_root=None`` is the stock
    (pristine upstream) case. ``materialize_source()`` (implicit state-scan
    baseline) is RETIRED (runbook 12 / RV80-B6).
    """
    return _materialize_v2(
        base_repo=base_repo,
        worktree_root=worktree_root,
        resolved_revision=resolved_revision,
        composition=tuple(composition),
        overlay_root=overlay_root,
        requested_revision=requested_revision,
    )


def materialize_source_variant(
    *,
    base_repo: Path,
    worktree_root: Path,
    resolved_revision: str,
    composition: Sequence[tuple[str, str]],
    variant_name: str,
    variant_digest: str,
    overlay_root: Path | None = None,
    requested_revision: str | None = None,
    apply_variant: Callable[[Path], None] | None = None,
) -> Path:
    """Materialize an isolated worktree for an EXPLICIT composition plus a
    post-composition variant transform ("subject vs control" evidence).

    The two worktrees differ ONLY by the content-addressed transform applied
    on top of the identical composition (e.g. the RD08 package-local correctness producer's
    VDR1-control / VDR2-subject pair). ``variant_digest`` MUST be a stable
    digest over the actual transform content (never a bare name) so two
    different transforms can never collide on the same source_key.

    RV80/B2: the identity is ``bigcherry-patch-source-variant-v2`` -- the
    same state-independent payload as :func:`materialize_composition` plus
    ``variant_name`` + ``variant_digest``. The v1 state-scan identity is
    retired; the composition comes explicitly from the caller (e.g. via
    :func:`resolve_source_composition`), never from a lifecycle-state scan.
    """
    return _materialize_v2(
        base_repo=base_repo,
        worktree_root=worktree_root,
        resolved_revision=resolved_revision,
        composition=tuple(composition),
        overlay_root=overlay_root,
        requested_revision=requested_revision,
        variant_name=variant_name,
        variant_digest=variant_digest,
        apply_variant=apply_variant,
    )


def materialize_stock_source(
    *,
    base_repo: Path,
    worktree_root: Path,
    base_revision: str,
) -> Path:
    """A genuinely pristine pinned-upstream worktree for stock bench
    comparison: v2 identity with the EMPTY composition and no overlay. The
    base ref is resolved to its immutable commit SHA first, so a moved ref
    yields a new identity (and the requested ref is recorded in the manifest
    informationally only)."""
    resolved_revision = resolve_base_revision(base_revision, repo=base_repo)
    return _materialize_v2(
        base_repo=base_repo,
        worktree_root=worktree_root,
        resolved_revision=resolved_revision,
        composition=(),
        overlay_root=None,
        requested_revision=base_revision,
    )


def resolve_source_composition(
    source_name: str,
    *,
    focal: str | None = None,
    extra_patches: tuple[str, ...] = (),
    base_ref: str = "HEAD",
    base_repo: Path | None = None,
    recipes: Path | None = None,
    patches_root: Path | None = None,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Resolve a source's EXPLICIT named composition for identity purposes.

    ``source_name``'s ``[patch-set.*]`` sets from config/recipes.toml (plus
    an optional single ``focal`` patch and/or an explicit ``extra_patches``
    stack, e.g. the RD08 evidence stack) are resolved through
    ``campaign_resolution.resolve_lane`` + ``patchset.resolve_exact`` -- the
    authoritative exact-composition validator (fail-closed on unknown IDs,
    rejected members, missing requires, internal conflicts) in TRUE
    topological order.

    Returns ``(resolved_base_sha, ordered [(patch_id, impl_digest)])``. This
    is the ONLY sanctioned way the identity/materialization path obtains a
    composition -- RV80/B6 forbids lifecycle-state scans here.
    """
    if base_repo is None:
        raise PatchSourceIsolationError(
            "resolve_source_composition: base_repo is required"
        )
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from bigcherry.campaign import resolution as campaign_resolution # noqa: E402
    from bigcherry.core import config as campaign_config # noqa: E402
    from bigcherry.core import paths # noqa: E402
    from bigcherry.patch import patchset # noqa: E402

    resolved_revision = resolve_base_revision(base_ref, repo=base_repo)
    cfg = campaign_config.load(recipes or paths.RECIPES)
    patches_root = patches_root or PATCHES_ROOT
    catalog = patchset.catalog(directory=patches_root)
    lane = campaign_resolution.resolve_lane(source_name, cfg, catalog)
    # lane.patch_set.module_ids is the FULL named composition (the merged
    # multi-set result already contains the validated-enhancements members;
    # lane.promoted_enhancements is a declared-config view of them, not an
    # extra). Extras/focal go on top and are deduplicated above.
    ids = list(lane.patch_set.module_ids)
    extras = list(extra_patches)
    if focal is not None:
        if focal in ids:
            raise PatchSourceIsolationError(
                f"focal patch {focal!r} is already in source {source_name!r}'s "
                "named composition"
            )
        extras.append(focal)
    if len(set(extras)) != len(extras):
        raise PatchSourceIsolationError("explicit composition contains duplicates")
    ids = [*ids, *extras]
    resolved = patchset.resolve_exact(tuple(ids), directory=patches_root)
    registry = _patch_registry().load_registry(patches_root)
    composition = tuple(
        (
            module.patch_id,
            hashlib.sha256(
                (
                    registry.root / registry.get(module.patch_id).implementation_path
                ).read_bytes()
            ).hexdigest(),
        )
        for module in resolved.modules
    )
    return resolved_revision, composition


def remove_worktree(base_repo: Path, source_dir: Path) -> None:
    """Explicit teardown -- worktrees are cheap to keep around for reuse, but
    callers doing disk-space cleanup should use this rather than a raw
    rmtree so `git worktree list` in base_repo doesn't accumulate stale
    registrations for a directory that no longer exists."""
    if source_dir.exists():
        _run(
            ["git", "worktree", "remove", "--force", str(source_dir)],
            cwd=base_repo,
            check=False,
        )
    shutil.rmtree(source_dir, ignore_errors=True)
    _manifest_path(source_dir).unlink(missing_ok=True)
    _run(["git", "worktree", "prune"], cwd=base_repo, check=False)
