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
from dataclasses import replace
from pathlib import Path

from . import resolution as campaign_resolution
from ..core import config as campaign_config
from ..patch import patchset
from ..patch.apply import PATCH_APPLICATION_SEMANTICS_VERSION
from ..source import workspace
from ..core.context import ProjectContext


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


def resolve_materialization_inputs(
    context: ProjectContext, plan: workspace.SourcePlan,
) -> workspace.SourcePlan:
    """Resolve patch paths/digests and overlay paths/digests exactly once."""
    selection = patchset.resolve_exact(
        plan.patch_ids, directory=context.patches_root,
        required_state=plan.required_state,
    )
    patches = tuple(
        workspace.ResolvedPatchInput(
            patch_id=module.patch_id,
            implementation_path=module.relative_path or module.path.relative_to(
                context.patches_root
            ),
            implementation_digest=module.content_hash,
        )
        for module in selection.modules
    )
    overlays: list[workspace.ResolvedOverlayInput] = []
    if plan.overlay_enabled and context.overlay_root.is_dir():
        for source in sorted(context.overlay_root.rglob("*")):
            if source.is_file():
                overlays.append(workspace.ResolvedOverlayInput(
                    relative_path=source.relative_to(context.overlay_root).as_posix(),
                    content_digest=hashlib.sha256(source.read_bytes()).hexdigest(),
                ))
    return replace(
        plan, resolved_patch_inputs=patches,
        resolved_overlay_inputs=tuple(overlays),
        inputs_resolved=True,
    )


def _overlay_hash_from_inputs(
    inputs: tuple[workspace.ResolvedOverlayInput, ...],
) -> str | None:
    if not inputs:
        return None
    digest = hashlib.blake2b(b"bigcherry/overlay-content/v2\0")
    for item in inputs:
        digest.update(len(item.relative_path).to_bytes(4, "big"))
        digest.update(item.relative_path.encode("utf-8"))
        digest.update(item.content_digest.encode("ascii"))
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

    GPT-auto-agent review (RE03/RE05 follow-up, 2026-08-17): also includes
    ``plan.patch_set_id``/``classification`` -- the reviewed LOGICAL
    composition identity, not just the byte-producing inputs above. Without
    this, two sources with different logical patch-set membership that
    happen to resolve to byte-identical trees (e.g. ``bigcherry-native``'s
    [framework] vs ``bigcherry``'s [framework, validated-enhancements]
    while validated-enhancements is empty) would share one materialisation
    directory, and materialize_source()'s cache-hit path returns the
    FIRST request's persisted patch_set_id/classification verbatim for
    the SECOND, differently-composed request -- a real cache-aliasing bug
    in the standard campaign profile, not a hypothetical: bigcherry-native
    runs before bigcherry in campaign.standard's lane order. Folding these
    into the destination key trades away byte-identical dedup between
    differently-composed sources in exchange for correct per-request
    provenance -- the simpler of the two fixes discussed, not the
    architecturally larger physical-cache/logical-provenance split.
    """
    if not plan.inputs_resolved:
        plan = resolve_materialization_inputs(context, plan)
    return {
        "upstream_revision": plan.upstream_revision,
        "overlay_enabled": plan.overlay_enabled,
        "overlay_content_hash": (
            _overlay_hash_from_inputs(plan.resolved_overlay_inputs)
        ),
        "patches": [
            {"patch_id": item.patch_id, "content_hash": item.implementation_digest}
            for item in plan.resolved_patch_inputs
        ],
        "required_state": plan.required_state,
        "patch_set_id": plan.patch_set_id,
        "classification": plan.classification,
        # PA07 (source/patch identity hardening L1.2): a patch's own content
        # digest cannot see a change to the SHARED application semantics
        # (apply.py/core/csource.py) that also determines its output bytes.
        "patch_application_semantics_version": PATCH_APPLICATION_SEMANTICS_VERSION,
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
    catalog_directory: Path | None = None,
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
            source_name, cfg, resolved_catalog, experiment=experiment,
            catalog_directory=catalog_directory,
        )
    except campaign_resolution.ResolutionError as exc:
        raise CampaignSourceError(str(exc)) from exc

    # Experimental lanes: resolve_lane() already enforced the base patch set's
    # declared state at planning time (validated) and admitted the experiment
    # module(s) with required_state=None (they are typically 'untested' --
    # that is the whole point of the --experiment path). Re-applying the base
    # set's state to the whole merged module list at materialize time would
    # reject the lane moments after planning approved it, so the strict
    # state gate is relaxed to None for experimental classification only.
    # Content identity is still provenance-enforced: materialize re-derives
    # each module's content hash from the catalog and records it in the
    # source metadata. Base (non-experimental) lanes keep the strict gate.
    plan_required_state = (
        None if lane.patch_set.classification == "experimental"
        else lane.patch_set.required_state
    )
    return workspace.SourcePlan(
        upstream_revision=revision,
        overlay_enabled=source.overlay,
        patch_ids=lane.patch_set.module_ids,
        required_state=plan_required_state,
        # RE03 (RV48 audit): resolve_lane already computed this reviewed
        # logical-composition identity -- carry it through instead of
        # discarding it here, so materialize() can persist it into source
        # provenance.
        patch_set_id=lane.patch_set.patch_set_id,
        classification=lane.patch_set.classification,
    )
