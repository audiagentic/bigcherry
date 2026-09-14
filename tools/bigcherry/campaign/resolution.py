"""Pure exact patch-set and campaign-lane resolution (BC04 boundary)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..core import config
from ..patch import patchset


class ResolutionError(ValueError):
    pass


# ---------------------------------------------------------------------------
# SelectorIdentity (PA34): the single canonical selector-identity authority.
#
# One immutable value object owns the complete serializable identity of an
# exact selection -- kind, deterministic name, source name/ref, patch-set id,
# and the exact ordered patch ids with their module content hashes. Resolution,
# rebase freshness, gates, validation, and receipts all compare/select through
# THIS object: its ``to_payload()``/``from_payload()`` are the one canonical
# serializer/validator, and exact equality (``==``) is the one shared
# comparison path. Consumers must not hand-assemble their own selector
# identity dictionaries (PA34 measurable target; enforced by a structural
# test).
# ---------------------------------------------------------------------------

SELECTOR_IDENTITY_SCHEMA_VERSION = 2

#: The closed set of selector kinds. Each kind's applicable fields are
#: validated in ``SelectorIdentity.__post_init__`` -- non-applicable fields
#: are explicitly ``None`` in the payload, never variant ad-hoc schemas.
SELECTOR_KIND_SOURCE = "source"
SELECTOR_KIND_EXPERIMENT = "experiment"
SELECTOR_KIND_FOCAL_OVERLAY = "focal-overlay"
SELECTOR_KIND_ALL_PATCHES = "all-patches"

_SELECTOR_KINDS = frozenset(
    (
        SELECTOR_KIND_SOURCE,
        SELECTOR_KIND_EXPERIMENT,
        SELECTOR_KIND_FOCAL_OVERLAY,
        SELECTOR_KIND_ALL_PATCHES,
    )
)

#: Kinds that resolve over a named source (and therefore carry source name,
#: resolved ref, and a patch-set composition identity).
_SOURCE_BOUNDED_KINDS = frozenset(
    (
        SELECTOR_KIND_SOURCE,
        SELECTOR_KIND_EXPERIMENT,
        SELECTOR_KIND_FOCAL_OVERLAY,
    )
)

_PAYLOAD_KEYS = frozenset(
    (
        "schema_version",
        "selector_kind",
        "selector_name",
        "source_name",
        "source_ref",
        "patch_set_id",
        "patch_ids",
        "module_hashes",
    )
)


def selector_name_for(kind: str, name: str) -> str:
    """The deterministic selector name for a kind: never free-form text.

    source -> the source name; experiment -> ``experiment:<name>``;
    focal-overlay -> ``focal:<patch-id>``; all-patches -> ``all-patches``.
    """
    if kind == SELECTOR_KIND_SOURCE:
        return name
    if kind == SELECTOR_KIND_EXPERIMENT:
        return f"experiment:{name}"
    if kind == SELECTOR_KIND_FOCAL_OVERLAY:
        return f"focal:{name}"
    if kind == SELECTOR_KIND_ALL_PATCHES:
        return SELECTOR_KIND_ALL_PATCHES
    raise ResolutionError(f"unknown selector kind {kind!r}")


def _split_selector_name(kind: str, selector_name: str) -> str:
    prefix = {"experiment": "experiment:", "focal-overlay": "focal:"}[kind]
    if not selector_name.startswith(prefix):
        raise ResolutionError(
            f"selector name {selector_name!r} does not carry the {kind!r} "
            f"{prefix!r} prefix"
        )
    suffix = selector_name[len(prefix) :]
    if not suffix:
        raise ResolutionError(
            f"selector name {selector_name!r} carries an empty "
            f"{prefix!r} qualifier -- experiment/focal names must be "
            "non-empty"
        )
    return suffix


@dataclass(frozen=True)
class SelectorIdentity:
    """The complete, immutable identity of one exact selection (PA34).

    ``selector_kind`` + ``selector_name`` name the selection; ``source_name``
    / ``source_ref`` / ``patch_set_id`` carry the source-bounded identity
    (explicitly ``None`` for coverage kinds); ``patch_ids`` is the exact
    ORDERED composition and ``module_hashes`` its paired ordered
    ``(patch_id, content_hash)`` list. Equality is exact-match semantics
    over all of these: a difference in kind, name, ref, order, or any
    module hash is a different identity.
    """

    selector_kind: str
    selector_name: str
    source_name: str | None
    source_ref: str | None
    patch_set_id: str | None
    patch_ids: tuple[str, ...]
    module_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.selector_kind not in _SELECTOR_KINDS:
            raise ResolutionError(f"unknown selector kind {self.selector_kind!r}")
        if not isinstance(self.selector_name, str) or not self.selector_name:
            raise ResolutionError("selector name must be a non-empty string")
        # PA34 adversarial-review fix (dev-gpt-agent req_b6af12ef4ad34bad
        # P3 #7): frozen=True only stops attribute re-assignment -- it does
        # not stop a caller passing a mutable list or non-string elements.
        # The identity is the shared serialization authority, so its shape
        # is validated here, at construction, not at payload time.
        if not isinstance(self.patch_ids, tuple) or not all(
            isinstance(patch_id, str) and patch_id for patch_id in self.patch_ids
        ):
            raise ResolutionError(
                "selector identity patch_ids must be a tuple of non-empty strings"
            )
        if not isinstance(self.module_hashes, tuple) or not all(
            isinstance(entry, tuple)
            and len(entry) == 2
            and all(isinstance(item, str) and item for item in entry)
            for entry in self.module_hashes
        ):
            raise ResolutionError(
                "selector identity module_hashes must be a tuple of "
                "(patch_id, content_hash) non-empty string pairs"
            )
        if self.selector_kind in _SOURCE_BOUNDED_KINDS:
            if (
                self.source_name is None
                or self.source_ref is None
                or self.patch_set_id is None
                or not all(
                    isinstance(value, str) and value
                    for value in (self.source_name, self.source_ref, self.patch_set_id)
                )
            ):
                raise ResolutionError(
                    f"{self.selector_kind!r} selector identity requires "
                    "non-empty string source_name, source_ref, and "
                    "patch_set_id"
                )
            if self.selector_kind == SELECTOR_KIND_SOURCE:
                expected = selector_name_for(SELECTOR_KIND_SOURCE, self.source_name)
            elif self.selector_kind == SELECTOR_KIND_EXPERIMENT:
                expected = selector_name_for(
                    SELECTOR_KIND_EXPERIMENT,
                    _split_selector_name(SELECTOR_KIND_EXPERIMENT, self.selector_name),
                )
            else:
                expected = selector_name_for(
                    SELECTOR_KIND_FOCAL_OVERLAY,
                    _split_selector_name(
                        SELECTOR_KIND_FOCAL_OVERLAY, self.selector_name
                    ),
                )
            if self.selector_name != expected:
                raise ResolutionError(
                    f"selector name {self.selector_name!r} is not the "
                    f"deterministic name {expected!r} for this selection"
                )
        else:
            if (
                self.source_name is not None
                or self.source_ref is not None
                or self.patch_set_id is not None
            ):
                raise ResolutionError(
                    "all-patches coverage identity carries no source-bounded "
                    "fields -- it is coverage identity, not a production "
                    "composition claim"
                )
            if self.selector_name != SELECTOR_KIND_ALL_PATCHES:
                raise ResolutionError(
                    f"all-patches selector name must be {SELECTOR_KIND_ALL_PATCHES!r}"
                )
        if len(self.patch_ids) != len(self.module_hashes):
            raise ResolutionError(
                "selector identity patch_ids/module_hashes cardinality "
                f"mismatch ({len(self.patch_ids)} vs {len(self.module_hashes)})"
            )
        for patch_id, (hash_id, _hash) in zip(
            self.patch_ids, self.module_hashes, strict=True
        ):
            if patch_id != hash_id:
                raise ResolutionError(
                    f"selector identity module_hashes entry {hash_id!r} "
                    f"does not align with patch_ids {patch_id!r}"
                )
        if len(set(self.patch_ids)) != len(self.patch_ids):
            raise ResolutionError("selector identity contains duplicate patch ids")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SelectorIdentity):
            return NotImplemented
        return (
            self.selector_kind == other.selector_kind
            and self.selector_name == other.selector_name
            and self.source_name == other.source_name
            and self.source_ref == other.source_ref
            and self.patch_set_id == other.patch_set_id
            and self.patch_ids == other.patch_ids
            and self.module_hashes == other.module_hashes
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.selector_kind,
                self.selector_name,
                self.source_name,
                self.source_ref,
                self.patch_set_id,
                self.patch_ids,
                self.module_hashes,
            )
        )

    def describe_diff(self, other: SelectorIdentity) -> tuple[str, ...]:
        """The named dimensions on which ``self`` and ``other`` differ --
        staleness messages must say WHICH identity fact moved (order-only
        drift? one module hash?), not merely that something did."""
        differing = [
            field
            for field in (
                "selector_kind",
                "selector_name",
                "source_name",
                "source_ref",
                "patch_set_id",
            )
            if getattr(self, field) != getattr(other, field)
        ]
        if self.patch_ids != other.patch_ids:
            differing.append("patch_ids")
        if self.module_hashes != other.module_hashes:
            differing.append("module_hashes")
        return tuple(differing)

    def to_payload(self) -> dict[str, object]:
        """The one canonical serializer (PA34). Fixed key set, stable order
        preserved, non-applicable fields explicitly null."""
        return {
            "schema_version": SELECTOR_IDENTITY_SCHEMA_VERSION,
            "selector_kind": self.selector_kind,
            "selector_name": self.selector_name,
            "source_name": self.source_name,
            "source_ref": self.source_ref,
            "patch_set_id": self.patch_set_id,
            "patch_ids": list(self.patch_ids),
            "module_hashes": [
                [patch_id, digest] for patch_id, digest in self.module_hashes
            ],
        }

    @classmethod
    def from_payload(cls, payload: object) -> SelectorIdentity:
        """The one canonical validator/deserializer (PA34). Any deviation
        from the exact payload shape -- extra/missing keys, wrong types, wrong
        schema version -- fails closed via the same ``__post_init__`` rules.
        """
        if not isinstance(payload, dict):
            raise ResolutionError("selector identity payload must be a mapping")
        if set(payload) != _PAYLOAD_KEYS:
            raise ResolutionError(
                "selector identity payload keys do not match the canonical "
                f"shape (got {sorted(payload)}, want {sorted(_PAYLOAD_KEYS)})"
            )
        if payload["schema_version"] != SELECTOR_IDENTITY_SCHEMA_VERSION:
            raise ResolutionError(
                f"selector identity schema_version {payload['schema_version']!r} "
                f"!= current {SELECTOR_IDENTITY_SCHEMA_VERSION!r}"
            )
        for key in ("selector_kind", "selector_name"):
            if not isinstance(payload[key], str):
                raise ResolutionError(f"selector identity {key!r} must be a string")
        for key in ("source_name", "source_ref", "patch_set_id"):
            if payload[key] is not None and not isinstance(payload[key], str):
                raise ResolutionError(
                    f"selector identity {key!r} must be a string or null"
                )
        patch_ids = payload["patch_ids"]
        module_hashes = payload["module_hashes"]
        if not isinstance(patch_ids, list) or not all(
            isinstance(i, str) for i in patch_ids
        ):
            raise ResolutionError(
                "selector identity patch_ids must be a list of strings"
            )
        if (
            not isinstance(module_hashes, list)
            or len(module_hashes) != len(patch_ids)
            or not all(
                isinstance(entry, list)
                and len(entry) == 2
                and all(isinstance(part, str) for part in entry)
                for entry in module_hashes
            )
        ):
            raise ResolutionError(
                "selector identity module_hashes must be a list of "
                "[patch_id, content_hash] pairs aligned with patch_ids"
            )
        return cls(
            selector_kind=payload["selector_kind"],  # type: ignore[arg-type]
            selector_name=payload["selector_name"],  # type: ignore[arg-type]
            source_name=payload["source_name"],  # type: ignore[arg-type]
            source_ref=payload["source_ref"],  # type: ignore[arg-type]
            patch_set_id=payload["patch_set_id"],  # type: ignore[arg-type]
            patch_ids=tuple(patch_ids),
            module_hashes=tuple(
                (entry[0], entry[1])
                for entry in module_hashes  # type: ignore[index]
            ),
        )


def build_all_patches_identity(
    catalog: list[patchset.PatchModule],
) -> SelectorIdentity:
    """The all-patches coverage identity (PA34): every non-rejected registry
    module in catalog order, carried with the SAME payload type as every
    other selector kind. It is coverage identity only -- a claim that these
    modules were each probed, NOT that the mutually conflicting ones among
    them form one production composition (rebase --all splits them into
    conflict-free probe groups; this identity never claims otherwise)."""
    from ..patch.patchset import RETIRED_STATES

    live = [m for m in catalog if m.state not in RETIRED_STATES]
    return SelectorIdentity(
        selector_kind=SELECTOR_KIND_ALL_PATCHES,
        selector_name=SELECTOR_KIND_ALL_PATCHES,
        source_name=None,
        source_ref=None,
        patch_set_id=None,
        patch_ids=tuple(m.patch_id for m in live),
        module_hashes=tuple((m.patch_id, m.content_hash) for m in live),
    )


@dataclass(frozen=True)
class ResolvedPatchSet:
    name: str
    module_ids: tuple[str, ...]
    module_hashes: tuple[tuple[str, str], ...]
    classification: str
    required_state: str | None
    patch_set_id: str


@dataclass(frozen=True)
class ResolvedLane:
    name: str
    source_name: str
    patch_set: ResolvedPatchSet
    promoted_enhancements: tuple[str, ...]


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(
        b"bigcherry/patch-set/v1\0" + encoded, digest_size=16
    ).hexdigest()


def resolve_patch_set(
    name: str,
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    *,
    extra_patch_ids: tuple[str, ...] = (),
    required_state_override: str | None = None,
    classification: str = "base",
    catalog_directory: object = None,
    composition_names: tuple[str, ...] | None = None,
    modules: dict[str, patchset.PatchModule] | None = None,
) -> ResolvedPatchSet:
    if name == "all":
        raise ResolutionError("'all' is not a valid production patch-set")
    if name not in cfg.patch_sets:
        raise ResolutionError(f"unknown patch set {name!r}")
    declared = cfg.patch_sets[name]
    ids = tuple(declared.patches) + tuple(extra_patch_ids)
    if len(set(ids)) != len(ids):
        raise ResolutionError("resolved patch set contains duplicate module IDs")
    # GPT-auto-agent review (RE03/RE05 follow-up, 2026-08-17): this used to
    # call patchset.catalog() with no directory override, defaulting to
    # paths.PATCHES (the real project's patches/) regardless of what
    # directory `catalog` was actually resolved against -- two authorities
    # in one execution, since materialisation/patch application elsewhere
    # in this same call chain correctly use context.patches_root. A caller
    # whose context is rooted at a different checkout (any isolated test,
    # or a real non-default patches_root in production) would have its
    # otherwise-correct catalog rejected here.
    #
    # Prefer an explicitly supplied ``catalog_directory`` (source_plan_for()
    # passes context.patches_root through resolve_lane() to here); fall
    # back to inferring it from the supplied catalog's own first entry only
    # when the caller didn't supply one -- inference alone loses the
    # directory entirely for a genuinely EMPTY custom catalog (a context
    # with zero patch files), which would otherwise silently fall back to
    # the wrong global default.
    if modules is not None:
        # PA34 (adversarial re-review, dev-gpt-agent req_1d02cb052310446c P1 Q2):
        # the caller bound this resolution to ONE catalog snapshot (the rebase
        # identity snapshot). Consume it via the pure validator -- never
        # re-read the registry, which would admit a concurrent
        # state/REQUIRES/CONFLICTS edit between identity resolution and
        # probing. The physical-consistency check below is skipped because
        # the supplied snapshot IS the authority on this path.
        selected = patchset.resolve_exact_from_catalog(
            ids,
            modules=modules,
            required_state=required_state_override or declared.required_state,
        )
        by_id = modules
    else:
        resolved_catalog_directory = (
            catalog_directory
            if isinstance(catalog_directory, Path)
            else (
                (catalog[0].catalog_root or catalog[0].path.parent) if catalog else None
            )
        )
        selected = patchset.resolve_exact(
            ids,
            directory=resolved_catalog_directory,
            required_state=required_state_override or declared.required_state,
        )
        by_id = {module.patch_id: module for module in catalog}
        if set(by_id) != {
            module.patch_id
            for module in patchset.catalog(directory=resolved_catalog_directory)
        }:
            raise ResolutionError(
                "catalog argument does not match the physical patch catalog"
            )
        # resolve_exact uses the project catalog; ensure the passed catalog
        # supplies identical content identities before exposing the result.
        for module in selected.modules:
            supplied = by_id.get(module.patch_id)
            if supplied is None or supplied.content_hash != module.content_hash:
                raise ResolutionError(
                    f"catalog identity mismatch for {module.patch_id}"
                )
    module_ids = tuple(module.patch_id for module in selected.modules)
    module_hashes = tuple(
        (module.patch_id, module.content_hash) for module in selected.modules
    )
    identity = {
        "schema_version": 1,
        "name": name,
        "required_state": required_state_override or declared.required_state,
        "modules": module_hashes,
        "classification": classification,
        # GPT-auto-agent review (RE03 comprehensive follow-up, 2026-08-17):
        # RE03's own architectural contract says two byte-identical sources
        # can have distinct REVIEWED logical compositions and must get
        # distinct patch_set_ids -- but resolve_lane()'s multi-patch-set
        # case collapsed every named-set combination into one synthetic
        # "__merged__" name before calling this function, so [A, B] and
        # [C, D] resolving to the same modules/state/classification got the
        # SAME patch_set_id; the ordered constituent set names were gone by
        # the time this identity was hashed. Include them explicitly
        # (defaults to (name,) for the single-set/no-composition case, so
        # this is not a behaviour change there).
        "composition_names": list(composition_names or (name,)),
    }
    return ResolvedPatchSet(
        name=name,
        module_ids=module_ids,
        module_hashes=module_hashes,
        classification=classification,
        required_state=identity["required_state"],
        patch_set_id=_digest(identity),
    )


def resolve_lane_overlay(
    source_name: str,
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    *,
    overlay_patch_ids: tuple[str, ...],
    overlay_name: str,
    catalog_directory: object = None,
    modules: dict[str, patchset.PatchModule] | None = None,
) -> ResolvedLane:
    """Resolve ``source_name``'s base lane with an EXACT extra module set
    layered on top, under ``resolve_exact``'s ``context_ids`` semantics (the
    overlay's own REQUIRES may be satisfied by the base lane without the
    base lane's modules being re-added to the overlay itself).

    This is the generic seam VA26's qualification-matrix planner composes
    "release" vs "release + candidate patch" through (dev-gpt-agent design
    review, req_5b7d0dacef604446): PQM's overlay is an unvalidated candidate
    module, not a named, pre-declared ``cfg.experiments`` entry, so it must
    not be forced through experiment-name identity. ``resolve_lane``'s own
    ``experiment=`` path is now a thin wrapper around this primitive (below)
    so both callers share one implementation of the overlay-composition
    identity rules.

    ``overlay_name`` is an opaque label folded into the resulting
    ``patch_set_id``'s identity payload (via ``composition_names``) --
    distinct overlays over the same base+module-set must not collide; it is
    not itself validated against ``cfg``.
    """
    if source_name not in cfg.sources:
        raise ResolutionError(f"unknown source {source_name!r}")
    base = resolve_lane(
        source_name,
        cfg,
        catalog,
        catalog_directory=catalog_directory,
        modules=modules,
    )
    if modules is not None:
        # PA34 (adversarial re-review, dev-gpt-agent req_1d02cb052310446c P1 Q2):
        # consume the caller's single catalog snapshot -- never re-read the
        # registry for the overlay either.
        overlay_selection = patchset.resolve_exact_from_catalog(
            tuple(overlay_patch_ids),
            modules=modules,
            required_state=None,
            context_ids=frozenset(base.patch_set.module_ids),
        )
        by_id = modules
    else:
        resolved_catalog_directory = (
            catalog_directory
            if isinstance(catalog_directory, Path)
            else (
                (catalog[0].catalog_root or catalog[0].path.parent) if catalog else None
            )
        )
        overlay_selection = patchset.resolve_exact(
            tuple(overlay_patch_ids),
            directory=resolved_catalog_directory,
            required_state=None,
            context_ids=frozenset(base.patch_set.module_ids),
        )
        by_id = {module.patch_id: module for module in catalog}
    if set(base.patch_set.module_ids) & {m.patch_id for m in overlay_selection.modules}:
        raise ResolutionError(f"overlay {overlay_name!r} repeats a base patch module")
    # PA34 adversarial-review fix (dev-gpt-agent req_b6af12ef4ad34bad P1 #3):
    # resolve_exact() with context_ids only checks the OVERLAY's conflicts
    # against the base (base IDs enter as context_ids, and only modules in
    # the resolved set are conflict-checked). The reverse direction -- a
    # base module declaring CONFLICTS on an overlay module -- is invisible
    # there, so the merged composition would silently contain a declared
    # conflict. Check it explicitly at the single merge site (both the
    # experiment and focal-overlay paths flow through here).
    overlay_id_set = {m.patch_id for m in overlay_selection.modules}
    for base_pid in base.patch_set.module_ids:
        base_conflicts = set(by_id[base_pid].conflicts) & overlay_id_set
        if base_conflicts:
            raise ResolutionError(
                f"base module {base_pid!r} conflicts with overlay "
                f"{overlay_name!r} module(s): "
                f"{', '.join(sorted(base_conflicts))}"
            )
    merged_ids = [
        *base.patch_set.module_ids,
        *(m.patch_id for m in overlay_selection.modules),
    ]
    merged_modules = tuple(
        by_id[pid] for pid in patchset.topological_order(merged_ids, modules=by_id)
    )
    module_ids = tuple(module.patch_id for module in merged_modules)
    module_hashes = tuple(
        (module.patch_id, module.content_hash) for module in merged_modules
    )
    identity = {
        "schema_version": 1,
        "name": base.patch_set.name,
        "required_state": base.patch_set.required_state,
        "modules": module_hashes,
        "classification": "experimental",
        "composition_names": list(cfg.sources[source_name].patch_sets) + [overlay_name],
    }
    resolved = ResolvedPatchSet(
        name=base.patch_set.name,
        module_ids=module_ids,
        module_hashes=module_hashes,
        classification="experimental",
        required_state=base.patch_set.required_state,
        patch_set_id=_digest(identity),
    )
    return ResolvedLane(
        name=f"{source_name}+{overlay_name}",
        source_name=source_name,
        patch_set=resolved,
        promoted_enhancements=base.promoted_enhancements,
    )


def resolve_lane(
    source_name: str,
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    *,
    experiment: str | None = None,
    catalog_directory: object = None,
    modules: dict[str, patchset.PatchModule] | None = None,
) -> ResolvedLane:
    if source_name not in cfg.sources:
        raise ResolutionError(f"unknown source {source_name!r}")
    source = cfg.sources[source_name]
    # `experiment` is validated up front, before the empty-patch-sets branch
    # below, so a clean/empty-base source (e.g. a stock Vulkan lane with no
    # framework patch-set yet) still resolves a requested experiment instead
    # of silently dropping it (bug found via external patch-management
    # review, 2026-08-20: this used to return unconditionally a few lines
    # below, before `experiment` was ever consulted).
    if experiment is not None and experiment not in cfg.experiments:
        raise ResolutionError(f"unknown experiment {experiment!r}")
    if not source.patch_sets:
        resolved = ResolvedPatchSet(
            "empty", (), (), "upstream", None, _digest({"modules": []})
        )
    elif len(source.patch_sets) == 1:
        base_name = source.patch_sets[0]
        resolved = resolve_patch_set(
            base_name,
            cfg,
            catalog,
            classification="experimental" if experiment else "base",
            catalog_directory=catalog_directory,
            composition_names=(base_name,),
            modules=modules,
        )
    else:
        # Each named set is resolved under its OWN required_state policy,
        # not just the first set's (bug found via external patch-management
        # review, 2026-08-20: flattening every set into one synthetic
        # "__merged__" PatchSet before resolving silently applied only
        # source.patch_sets[0]'s policy to every module -- correct today
        # only because every current multi-set source happens to share one
        # policy, and would be silently wrong the moment two named sets in
        # the same source genuinely diverge).
        per_set = [
            resolve_patch_set(
                name,
                cfg,
                catalog,
                classification="base",
                catalog_directory=catalog_directory,
                composition_names=(name,),
                modules=modules,
            )
            for name in source.patch_sets
        ]
        claimed_by: dict[str, str] = {}
        for resolved_set in per_set:
            for patch_id in resolved_set.module_ids:
                if patch_id in claimed_by:
                    raise ResolutionError(
                        f"patch {patch_id!r} is claimed by more than one patch-set "
                        f"in source {source_name!r} ({claimed_by[patch_id]!r} and "
                        f"{resolved_set.name!r})"
                    )
                claimed_by[patch_id] = resolved_set.name
        by_id = (
            modules
            if modules is not None
            else {module.patch_id: module for module in catalog}
        )
        # RV80 follow-up (GPT deep review, systemic): a GLOBAL (order, patch_id)
        # re-sort here would destroy the dependency order that
        # resolve_patch_set()/resolve_exact() already established whenever
        # numbering and REQUIRES disagree -- reintroducing the B4 class of bug
        # one layer above patchset. Use the true topological picker instead
        # (identical output for every currently-configured source, whose
        # numbering is consistent with REQUIRES).
        merged_ids = [
            patch_id for resolved_set in per_set for patch_id in resolved_set.module_ids
        ]
        merged_modules = tuple(
            by_id[pid] for pid in patchset.topological_order(merged_ids, modules=by_id)
        )
        module_ids = tuple(module.patch_id for module in merged_modules)
        module_hashes = tuple(
            (module.patch_id, module.content_hash) for module in merged_modules
        )
        # When every constituent set was resolved under the SAME
        # required_state policy (today's only real case: `bigcherry`'s
        # framework + validated-enhancements both require "validated"),
        # the merged result still has exactly one true policy -- represent
        # it as that shared value, not None, so patch_set_id is unchanged
        # from before this fix for every currently-configured source.
        # required_state only becomes None (genuinely ambiguous) once two
        # sets in the same source actually diverge.
        distinct_policies = {resolved_set.required_state for resolved_set in per_set}
        merged_required_state = (
            next(iter(distinct_policies)) if len(distinct_policies) == 1 else None
        )
        classification = "experimental" if experiment else "base"
        identity = {
            "schema_version": 1,
            "name": "__merged__",
            "required_state": merged_required_state,
            "modules": module_hashes,
            "classification": classification,
            "composition_names": list(source.patch_sets),
        }
        resolved = ResolvedPatchSet(
            name="__merged__",
            module_ids=module_ids,
            module_hashes=module_hashes,
            classification=classification,
            required_state=merged_required_state,
            patch_set_id=_digest(identity),
        )
    resolved_catalog_directory = catalog_directory or (
        (catalog[0].catalog_root or catalog[0].path.parent) if catalog else None
    )
    if experiment:
        # Delegate to the generic overlay primitive (VA26 design review,
        # req_5b7d0dacef604446) so a named experiment and an ad-hoc PQM
        # candidate overlay share one implementation of the composition
        # identity rules, rather than drifting independently.
        overlaid = resolve_lane_overlay(
            source_name,
            cfg,
            catalog,
            overlay_patch_ids=cfg.experiments[experiment].patches,
            overlay_name=f"experiment:{experiment}",
            catalog_directory=resolved_catalog_directory,
            modules=modules,
        )
        resolved = overlaid.patch_set
    return ResolvedLane(
        name=f"{source_name}+{experiment}" if experiment else source_name,
        source_name=source_name,
        patch_set=resolved,
        # A view of the promoted enhancements THIS lane actually composes --
        # not the global contents of validated-enhancements. patch/source.py
        # already documents the intent ("a declared-config view of them, not
        # an extra"), but the implementation returned the whole set for every
        # source, including ones whose patch-sets do not include it.
        #
        # Latent while validated-enhancements was empty; populating it made
        # bigcherry-native -- the framework-only CONTROL source -- report a
        # promoted enhancement it does not build. Actual patch selection
        # (patch_set.module_ids) was always correct, so no build shipped the
        # wrong thing, but any consumer reading this field for a native lane
        # would have been told the control arm carries an enhancement. That is
        # precisely the baseline-contamination story a control exists to rule
        # out, so it must not be merely conventionally true.
        promoted_enhancements=tuple(
            patch_id
            for patch_id in cfg.patch_sets["validated-enhancements"].patches
            if patch_id in set(resolved.module_ids)
        )
        if "validated-enhancements" in cfg.patch_sets
        else (),
    )


@dataclass(frozen=True)
class CanonicalSelection:
    """One exact selection, carried as its ``SelectorIdentity`` (PA34).

    The identity object is the single authority -- the source-name/ref,
    patch-set id, and patch-id projections below are read-only views of
    ``identity`` (bounded compatibility for live consumers), never a second,
    separately stored copy. Introduced to remove ``recipes.py``'s retired
    compatibility bridge without smuggling ``Recipe``'s unrelated
    build/platform/groups/states baggage back in through a different name.
    """

    identity: SelectorIdentity

    @property
    def source_name(self) -> str | None:
        return self.identity.source_name

    @property
    def source_ref(self) -> str | None:
        return self.identity.source_ref

    @property
    def patch_set_id(self) -> str | None:
        return self.identity.patch_set_id

    @property
    def patch_ids(self) -> tuple[str, ...]:
        return self.identity.patch_ids

    @property
    def module_hashes(self) -> tuple[tuple[str, str], ...]:
        return self.identity.module_hashes


def resolve_canonical_selection(
    source_name: str,
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    *,
    experiment: str | None = None,
    focal_overlay_patch_id: str | None = None,
    catalog_directory: object = None,
) -> CanonicalSelection:
    """Resolve ``source_name`` to its real upstream ref and exact patch-set
    identity, in one call -- the two things every migrated legacy command
    (``pull``, ``patch-rebase-check``, ``patch-doc``, ``pin-bump``,
    ``probe-release``/``validate-ref``) needs.

    ``experiment`` and ``focal_overlay_patch_id`` are mutually exclusive
    (PA34): a named experiment delegates to the existing
    ``resolve_lane(..., experiment=...)`` path; a focal overlay resolves the
    base source, expands the focal's REQUIRES closure, removes the ids the
    base already carries, and layers ONLY the remainder through
    ``resolve_lane_overlay`` -- a deterministic ``focal:<patch-id>``
    selection, never an arbitrary user patch list.

    Ref resolution matches ``campaign/source.py``'s own
    ``revision = cfg.pinned if source.ref == "pinned" else source.ref``
    convention exactly, so a migrated caller sees the identical ref it
    would have seen under the old ``Recipe.follows_pin`` semantics.
    """
    if experiment is not None and focal_overlay_patch_id is not None:
        raise ResolutionError(
            "experiment and focal_overlay_patch_id are mutually exclusive"
        )
    if source_name not in cfg.sources:
        raise ResolutionError(f"unknown source {source_name!r}")
    source = cfg.sources[source_name]
    resolved_ref = cfg.pinned if source.ref == "pinned" else source.ref

    # PA34 (adversarial re-review, dev-gpt-agent req_1d02cb052310446c P1 Q2):
    # bind the ENTIRE canonical resolution to the caller's supplied catalog
    # snapshot. Every sub-resolution (patch-set, overlay, focal closure) is
    # validated against these exact modules -- the registry is never
    # re-read on this path, so a concurrent state/REQUIRES/CONFLICTS edit
    # cannot change what is resolved between identity resolution and probing.
    # The 3 live callers (rebase identity/snapshot, cli patch-gates, patch
    # selection) all pass a real catalog, so this is always-consume.
    by_id = {module.patch_id: module for module in catalog}

    if focal_overlay_patch_id is not None:
        if focal_overlay_patch_id not in by_id:
            raise ResolutionError(f"unknown focal patch {focal_overlay_patch_id!r}")
        base = resolve_lane(
            source_name,
            cfg,
            catalog,
            catalog_directory=catalog_directory,
            modules=by_id,
        )
        closure = patchset.expand_composition_from_modules(
            (focal_overlay_patch_id,),
            modules=by_id,
        ).expanded
        base_ids = set(base.patch_set.module_ids)
        overlay_ids = tuple(
            patch_id for patch_id in closure if patch_id not in base_ids
        )
        lane = resolve_lane_overlay(
            source_name,
            cfg,
            catalog,
            overlay_patch_ids=overlay_ids,
            overlay_name=selector_name_for(
                SELECTOR_KIND_FOCAL_OVERLAY, focal_overlay_patch_id
            ),
            catalog_directory=catalog_directory,
            modules=by_id,
        )
        kind = SELECTOR_KIND_FOCAL_OVERLAY
    else:
        lane = resolve_lane(
            source_name,
            cfg,
            catalog,
            experiment=experiment,
            catalog_directory=catalog_directory,
            modules=by_id,
        )
        kind = SELECTOR_KIND_EXPERIMENT if experiment else SELECTOR_KIND_SOURCE

    return CanonicalSelection(
        identity=SelectorIdentity(
            selector_kind=kind,
            selector_name=selector_name_for(
                kind,
                experiment if experiment else (focal_overlay_patch_id or source_name),
            ),
            source_name=source_name,
            source_ref=resolved_ref,
            patch_set_id=lane.patch_set.patch_set_id,
            patch_ids=lane.patch_set.module_ids,
            module_hashes=lane.patch_set.module_hashes,
        ),
    )
