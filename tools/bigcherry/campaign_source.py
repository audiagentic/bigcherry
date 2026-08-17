"""Bridge from v2 campaign config to the isolated-source primitive (RE14).

``config.Source``/``config.PatchSet`` describe *what* a source variant is;
``workspace.SourcePlan`` is what ``workspace.materialize`` actually consumes
to produce an isolated, content-identified worktree. Nothing wired these
together before RE14: campaign config parsing and source materialisation
existed as two separate, independently-tested primitives with no bridge.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import campaign_resolution
from . import config as campaign_config
from . import patchset
from . import workspace
from .context import ProjectContext


class CampaignSourceError(ValueError):
    pass


def _overlay_content_hash(overlay_root: Path) -> str | None:
    """A single hash over every overlay file's relative path and bytes,
    order-independent (sorted by relative path first). ``None`` when there
    is nothing to hash, so an overlay-disabled plan's identity is not
    perturbed by an unrelated overlay_root that happens to exist on disk.
    """
    if not overlay_root.is_dir():
        return None
    digest = hashlib.blake2b(b"bigcherry/overlay-content/v1\0")
    found = False
    for source in sorted(overlay_root.rglob("*")):
        if not source.is_file():
            continue
        found = True
        relative = source.relative_to(overlay_root).as_posix()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative.encode("utf-8"))
        content = source.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    if not found:
        return None
    return digest.hexdigest()


def resolve_materialization_identity(
    context: ProjectContext, plan: workspace.SourcePlan,
) -> dict[str, object]:
    """RE04 (RV48 audit fix): the CONTENT-aware identity a materialisation
    destination is actually keyed by -- resolves ``plan.patch_ids`` against
    the real patch catalog to get each module's own ``content_hash`` (an
    in-place edit to a patch file under its unchanged canonical ID changes
    this), and hashes the overlay tree's actual file bytes (an overlay edit
    changes this too). ``source_plan_id()`` alone (IDs/flags only) is not
    safe to key a reused destination directory by -- see its own docstring.
    """
    selection = patchset.resolve_exact(
        plan.patch_ids, directory=context.patches_root,
        required_state=plan.required_state,
    )
    return {
        "upstream_revision": plan.upstream_revision,
        "overlay_enabled": plan.overlay_enabled,
        "overlay_content_hash": (
            _overlay_content_hash(context.overlay_root) if plan.overlay_enabled else None
        ),
        "patches": [
            {"patch_id": module.patch_id, "content_hash": module.content_hash}
            for module in selection.modules
        ],
        "required_state": plan.required_state,
    }


def materialization_plan_id(identity: dict[str, object]) -> str:
    """Deterministic id for a content-resolved materialisation identity --
    what the destination directory is actually keyed by (RE04). Distinct
    from ``source_plan_id()``, which is cheap/context-free and therefore
    necessarily ID-only.
    """
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(
        b"bigcherry/materialization-plan/v1\0" + encoded, digest_size=16
    ).hexdigest()


def source_plan_id(plan: workspace.SourcePlan) -> str:
    """Deterministic id for a SourcePlan, independent of materialising it.

    ID-only (patch IDs, not their content) and therefore NOT safe to key a
    reused materialisation destination directory by -- an in-place edit to
    a patch module under its unchanged canonical ID would collide with the
    old materialisation. Use ``materialization_plan_id(resolve_materialization_
    identity(context, plan))`` for that. This function remains for cheap,
    context-free comparisons where content-safety is not the concern (e.g.
    logging, request-shape equality checks before a context even exists).

    ``source_slice_id`` (source_identity.source_slice_id) can only be known
    AFTER materialisation -- it hashes the actual post-overlay-and-patch git
    tree OID.
    """
    payload = {
        "upstream_revision": plan.upstream_revision,
        "overlay_enabled": plan.overlay_enabled,
        "patch_ids": sorted(plan.patch_ids),
        "required_state": plan.required_state,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(
        b"bigcherry/source-plan/v1\0" + encoded, digest_size=16
    ).hexdigest()


def source_plan_for(
    cfg: campaign_config.Config,
    source_name: str,
    *,
    catalog: list[patchset.PatchModule] | None = None,
    experiment: str | None = None,
) -> workspace.SourcePlan:
    """Resolve one ``config.Source`` into a ``workspace.SourcePlan``.

    Patch composition is delegated to ``campaign_resolution.resolve_lane``
    rather than reimplemented here: a first version of this function
    flattened ``source.patch_sets`` by hand, which is weaker than what
    already existed -- ``resolve_lane`` additionally cross-checks every
    resolved patch module's content hash against the physical patch
    catalog (``patchset.catalog()``), which a hand-rolled flatten silently
    skips. ``resolve_lane`` was already implemented and unit-tested
    (BC04); this function was simply not using it.

    ``upstream_revision`` is passed through as ``source.ref`` (or
    ``cfg.pinned`` for the ``"pinned"`` alias) WITHOUT resolving a tag or
    short ref to its full commit SHA. Two callers spelling the same commit
    two different ways would therefore get two different
    ``source_slice_id``s for identical content -- resolving that requires
    ``workspace.UpstreamRepository(context.upstream_repo).resolve_ref(...)``,
    which needs a real local clone with the ref already fetched, so it is
    left to the caller (materialisation time), not done here.
    """
    if source_name not in cfg.sources:
        raise CampaignSourceError(
            f"unknown source {source_name!r}; valid choices: "
            f"{', '.join(sorted(cfg.sources))}"
        )
    source = cfg.sources[source_name]
    revision = cfg.pinned if source.ref == "pinned" else source.ref
    resolved_catalog = catalog if catalog is not None else patchset.catalog()

    try:
        lane = campaign_resolution.resolve_lane(
            source_name, cfg, resolved_catalog, experiment=experiment
        )
    except campaign_resolution.ResolutionError as exc:
        raise CampaignSourceError(str(exc)) from exc

    return workspace.SourcePlan(
        upstream_revision=revision,
        overlay_enabled=source.overlay,
        patch_ids=lane.patch_set.module_ids,
        required_state=lane.patch_set.required_state,
    )
