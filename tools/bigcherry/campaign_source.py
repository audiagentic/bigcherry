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

from . import campaign_resolution
from . import config as campaign_config
from . import patchset
from . import workspace


class CampaignSourceError(ValueError):
    pass


def source_plan_id(plan: workspace.SourcePlan) -> str:
    """Deterministic id for a SourcePlan, independent of materialising it.

    ``source_slice_id`` (source_identity.source_slice_id) can only be known
    AFTER materialisation -- it hashes the actual post-overlay-and-patch git
    tree OID. A materialise destination path has to be chosen BEFORE that,
    so callers need a stable identifier derived purely from the plan's own
    fields to pick (and later recognise/reuse) that destination.
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
