"""Host-local upstream repository and immutable source materialisation."""

from __future__ import annotations

import shutil
import subprocess
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..patch import apply as patcher
from ..patch.apply import PatchError, resolve_contained_target
from ..patch import patchset
from ..patch import registry as patch_registry
from ..core.context import ProjectContext
from .identity import describe


class WorkspaceError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise WorkspaceError(detail.strip()) from exc
    return result.stdout.strip()


class UpstreamRepository:
    def __init__(self, path: Path):
        self.path = path.resolve()

    def resolve_ref(self, ref: str) -> str:
        return _git(self.path, "rev-parse", f"{ref}^{{commit}}")

    def fetch_ref(self, ref: str) -> str:
        """Fetch ``ref`` from ``origin`` into this repo and return its
        resolved commit sha.

        RE13: the only place in the codebase that fetches an arbitrary,
        not-yet-local ref -- canonical production builds only ever resolve
        refs already present (the pin, updated by a separate, deliberate
        process), by design (RE13's own detailed_solution: do not
        reintroduce a general --ref override into canonical builds). A
        release-compatibility probe against an upstream ref ahead of the
        pin (typically ``master``) is the one legitimate exception, and
        needs this repo's ``origin`` remote to already exist -- it never
        clones one.

        GPT-auto-agent review (2026-08-17): fetches into a private,
        uniquely-named ref rather than resolving the shared, mutable
        ``FETCH_HEAD`` -- this mirror is shared with real production
        materialisation and can legitimately see concurrent fetches (e.g.
        two probes at once); resolving ``FETCH_HEAD`` after a plain
        ``git fetch`` is a real race where a concurrent fetch on the same
        mirror can flip it before this call reads it back, resolving the
        WRONG commit for the ref actually requested here.
        """
        private_ref = f"refs/bigcherry-probe/{uuid.uuid4().hex}"
        try:
            _git(self.path, "fetch", "--no-tags", "origin", f"{ref}:{private_ref}")
            return self.resolve_ref(private_ref)
        finally:
            _git(self.path, "update-ref", "-d", private_ref)

    def ensure_commit(self, revision: str) -> None:
        _git(self.path, "cat-file", "-e", f"{revision}^{{commit}}")

    def add_detached_worktree(self, revision: str, path: Path) -> None:
        self.ensure_commit(revision)
        path = path.resolve()
        if path.exists() and any(path.iterdir()):
            raise WorkspaceError(f"source worktree target is not empty: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        _git(self.path, "worktree", "add", "--detach", str(path), revision)
        actual = _git(path, "rev-parse", "HEAD")
        if actual != revision:
            raise WorkspaceError(f"worktree HEAD {actual} does not equal {revision}")

    def remove_worktree(self, path: Path) -> None:
        _git(self.path, "worktree", "remove", "--force", str(path.resolve()))


@dataclass(frozen=True)
class SourcePlan:
    upstream_revision: str
    overlay_enabled: bool
    patch_ids: tuple[str, ...]
    required_state: str | None = None
    #: RE03 (RV48 audit): the reviewed logical-composition identity
    #: campaign_resolution.resolve_lane already computes (patch_set_id --
    #: distinct from the byte-level source_slice_id two different logical
    #: compositions could still legitimately share) -- optional so direct
    #: construction (most existing tests) is unaffected; source_plan_for()
    #: is the one real caller that populates it, closing the gap where this
    #: identity was computed then silently discarded before reaching
    #: materialize().
    patch_set_id: str | None = None
    #: The declared patch-set classification (e.g. "base"/"experimental" --
    #: see campaign_resolution.ResolvedPatchSet); which named [patch-set.*]
    #: a module belongs to (framework vs validated-enhancements) is the
    #: actual human review boundary and is already explicit in recipes.toml
    #: itself, not re-decided here.
    classification: str | None = None
    resolved_patch_inputs: tuple["ResolvedPatchInput", ...] = ()
    resolved_overlay_inputs: tuple["ResolvedOverlayInput", ...] = ()
    inputs_resolved: bool = False


@dataclass(frozen=True)
class ResolvedPatchInput:
    patch_id: str
    implementation_path: Path
    implementation_digest: str


@dataclass(frozen=True)
class ResolvedOverlayInput:
    relative_path: str
    content_digest: str


def bigcherry_revision(context: ProjectContext) -> str:
    """RE25.2: the BigCherry project repository's own HEAD (the tooling
    revision this execution ran under), as a full commit SHA.

    Distinct from every upstream/source revision in the same execution:
    source provenance records which upstream tree was materialised, while
    project.provenance_class + project.bigcherry_revision record which
    BigCherry checkout PRODUCED the artifacts -- a release consumer
    auditing a runtime bundle needs both, and today's call sites pass an
    empty project namespace (no revision at all), so this was previously
    unrecoverable.
    """
    return _git(context.project_root, "rev-parse", "HEAD")


def require_clean_bigcherry(
    context: ProjectContext, *, allow_dirty_bigcherry: bool
) -> None:
    """RE04 (RV48 audit fix): the dirty-BigCherry-tree check as its own
    reusable function, so a cache-hit path (campaign_build.materialize_source)
    can enforce it too -- it used to live only inside ``materialize()``,
    which a cache hit never called, so dirty-tree execution was implicit on
    reuse regardless of what a caller intended.
    """
    if allow_dirty_bigcherry:
        return
    status = subprocess.run(
        ["git", "-C", str(context.project_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise WorkspaceError(
            "BigCherry repository is dirty; use explicit development override"
        )


def materialize(
    context: ProjectContext,
    plan: SourcePlan,
    destination: Path,
    *,
    allow_dirty_bigcherry: bool = False,
) -> dict[str, object]:
    """Materialise one source plan and return external identity metadata."""
    require_clean_bigcherry(context, allow_dirty_bigcherry=allow_dirty_bigcherry)
    repository = UpstreamRepository(context.upstream_repo)
    repository.add_detached_worktree(plan.upstream_revision, destination)
    allowed_untracked: set[str] = set()
    if plan.overlay_enabled:
        for resolved in plan.resolved_overlay_inputs:
            source = context.overlay_root / resolved.relative_path
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != resolved.content_digest:
                raise WorkspaceError(
                    f"overlay bytes changed after plan resolution: {resolved.relative_path}"
                )
            relative = Path(resolved.relative_path)
            try:
                target = resolve_contained_target(destination, relative.as_posix())
            except PatchError as exc:
                raise WorkspaceError(f"overlay target escapes source root: {exc}") from exc
            was_present = target.exists()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            if not was_present:
                allowed_untracked.add(relative.as_posix())
    if not plan.inputs_resolved:
        raise WorkspaceError("source plan has no resolved patch/overlay inputs")
    registry = patch_registry.load_registry(context.patches_root)
    loaded = []
    for resolved in plan.resolved_patch_inputs:
        descriptor = registry.get(resolved.patch_id)
        if descriptor.implementation_path != resolved.implementation_path:
            raise WorkspaceError(f"patch path changed after plan resolution: {resolved.patch_id}")
        loaded.extend(patch_registry.load_implementation(
            descriptor, root=context.patches_root,
            expected_digest=resolved.implementation_digest,
        ))
    results = patcher.apply_all(loaded, destination)
    if not all(result.ok for result in results):
        raise WorkspaceError("source patch application failed")
    metadata = describe(
        root=destination,
        upstream_revision=plan.upstream_revision,
        allowed_untracked=allowed_untracked,
    )
    metadata["plan"] = {
        "upstream_revision": plan.upstream_revision,
        "overlay_enabled": plan.overlay_enabled,
        "patches": [
            {"patch_id": item.patch_id, "content_hash": item.implementation_digest}
            for item in plan.resolved_patch_inputs
        ],
        "required_state": plan.required_state,
        # RE03 (RV48 audit): the reviewed logical-composition identity,
        # carried through into materialised source provenance instead of
        # being discarded after campaign_resolution.resolve_lane computes
        # it -- verifiable from this metadata without recomputing patch
        # resolution. None for a caller that constructed SourcePlan
        # directly rather than through source_plan_for().
        "patch_set_id": plan.patch_set_id,
        "classification": plan.classification,
    }
    # RE04 (RV48 audit fix): stored so a later cache-hit re-verification can
    # recompute this same destination's git_tree_oid with the SAME allowed-
    # untracked set used when it was first materialised, without having to
    # re-walk overlay_root (which may have changed since -- caught instead
    # by the overlay content hash in resolve_materialization_identity()).
    metadata["allowed_untracked"] = sorted(allowed_untracked)
    return metadata
