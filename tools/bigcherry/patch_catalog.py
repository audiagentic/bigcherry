"""Declarative patch-stream metadata (RE30 phase 1).

``patches/catalog.toml`` is orthogonal metadata layered on top of each patch
module's existing ``GROUP``/``STATE`` Python constants (see ``patchset.py``).
It answers questions those two labels cannot: whether a patch is BigCherry's
own instrumentation, a backport of specific upstream work, or a port from an
external fork; which backend (hip/vulkan/agnostic) it targets; and, for
backports, where the fix came from and when it should be retired.

GROUP/STATE, recipes.toml membership, and patchset.py remain authoritative
for patch COMPOSITION: this module never adds, removes, substitutes, or
silently skips patches in an already-resolved patch set.

Catalog metadata may, however, be consumed by explicit ADMISSION policy
after composition has been resolved (``resolve_for_context``/
``applicable`` already do this for backend/option applicability). An
admission check may reject a resolved patch set because its backend,
options, or (HI83) validation-evidence obligations are not satisfied.
Rejection is not selection: the resolved composition remains unchanged and
the operation fails closed rather than silently substituting or dropping
a patch.

This module remains read-only with respect to patch lifecycle state and
catalog contents. The HI83 evidence functions below
(``validation_evidence_statuses``/``require_validation_evidence``) are
verifiers, not the production admission boundary -- as of this writing
nothing calls them from a live ``bigcherry apply``/``bigcherry build``
path; wiring a hard gate is a deliberate, separately-adjudicated decision
(see plan item HI83's notes for the current status and why).

Revised 2026-08-22 per GPT review (req_b87ea92609fa45fe): the previous
wording ("read-only, additive metadata -- it does not change what apply/
build actually select") overstated the invariant relative to
resolve_for_context()'s pre-existing hard admission behavior, and did not
leave room for HI83 evidence to ever become a real (but still
post-selection, fail-closed) admission check.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import config as campaign_config
from . import patch_validation_evidence
from . import paths, patchset

KINDS = ("framework", "upstream-backport", "enhancement")
ORIGINS = ("local", "upstream-commit", "upstream-pr", "external-fork")
BACKENDS = ("hip", "vulkan", "agnostic")


@dataclass(frozen=True)
class CatalogEntry:
    patch_id: str
    kind: str
    origin: str
    backend: str
    state: str
    upstream_ref: str | None = None
    retirement: str | None = None
    external_source: str | None = None
    plan_item: str | None = None
    requires_options: tuple[str, ...] = ()
    forbids_options: tuple[str, ...] = ()
    # RE40 (external patch-management review, 2026-08-20): genuinely new
    # descriptive fields -- NOT `requires`/`conflicts`. Those already exist
    # as a real, ENFORCED mechanism (patchset.PatchModule.requires/conflicts,
    # read from each patch module's own REQUIRES/CONFLICTS constants and
    # validated by patchset.resolve_exact(), which resolve_lane's
    # experiment= path already calls) -- adding a second, metadata-only
    # requires/conflicts pair here would create two disagreeing sources of
    # truth for the same relationship. This catalog stays purely
    # descriptive/additive, per its own module docstring.
    #
    # `plan_ids` is plural and additive alongside the older singular
    # `plan_item` (kept for backward compat -- every existing entry that set
    # `plan-item` still works unchanged); a patch can genuinely serve more
    # than one RD/EX/HI plan item. `backends` is plural alongside the older
    # singular `backend` for the same reason (a future patch may span
    # hip+vulkan). `subsystems`/`hardware` are free-form descriptive tags,
    # not validated against a closed vocabulary -- explicitly NOT a folder
    # axis (RE41: patches/ stays flat, this is browsability metadata only).
    plan_ids: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()
    subsystems: tuple[str, ...] = ()
    hardware: tuple[str, ...] = ()
    # HI83: which GPU architecture(s) must have a current, qualifying
    # patch_validation_evidence record before this patch may claim global
    # STATE="validated" -- a distinct policy field from `hardware` (free-form
    # descriptive metadata, not validated against a closed vocabulary,
    # explicitly not a correctness obligation per that field's own docstring
    # above). Empty means no architecture-coverage requirement is enforced.
    validation_architectures: tuple[str, ...] = ()


@dataclass(frozen=True)
class PatchContext:
    """The lane a patch is being resolved for (RE30 phase 1).

    Backend-aware resolution needs to know more than a patch-set name: which
    backend the lane targets and which cmake options the resolved build will
    actually pass, so a patch declaring ``requires-options`` can be rejected
    up front rather than silently miscompiling.
    """

    backend: str
    source: str | None = None
    build: str | None = None
    platform: str | None = None
    options: tuple[str, ...] = ()


def applicable(entry: CatalogEntry, context: PatchContext) -> tuple[bool, str | None]:
    """Whether ``entry`` may be selected for ``context``. Returns
    ``(True, None)`` or ``(False, reason)`` -- never silently ambiguous."""
    if entry.backend != "agnostic" and entry.backend != context.backend:
        return False, f"backend mismatch: patch is {entry.backend!r}, context is {context.backend!r}"
    missing = [opt for opt in entry.requires_options if opt not in context.options]
    if missing:
        return False, f"missing required option(s): {', '.join(missing)}"
    conflicting = [opt for opt in entry.forbids_options if opt in context.options]
    if conflicting:
        return False, f"forbidden option(s) present: {', '.join(conflicting)}"
    return True, None


def resolve_for_context(
    patch_ids: tuple[str, ...] | list[str],
    context: PatchContext,
    *,
    catalog_path: Path | None = None,
) -> tuple[str, ...]:
    """Check an explicit patch selection against ``context``.

    A selected-but-inapplicable patch is a hard error, never a silent skip
    (RE30's own design requirement) -- callers that want backend filtering
    must filter BEFORE calling this, e.g. via ``patches_for_backend``.
    """
    entries = load_catalog(catalog_path)
    errors: list[str] = []
    for patch_id in patch_ids:
        entry = entries.get(patch_id)
        if entry is None:
            errors.append(f"{patch_id}: no catalog entry, cannot check applicability")
            continue
        ok, reason = applicable(entry, context)
        if not ok:
            errors.append(f"{patch_id}: {reason}")
    if errors:
        raise ValueError(
            f"patch selection is not applicable to backend {context.backend!r}: "
            + "; ".join(errors)
        )
    return tuple(patch_ids)


def patches_for_backend(backend: str, *, catalog_path: Path | None = None) -> tuple[str, ...]:
    """Every catalog patch ID whose backend is ``backend`` or 'agnostic'."""
    entries = load_catalog(catalog_path)
    return tuple(sorted(
        patch_id for patch_id, entry in entries.items()
        if entry.backend in (backend, "agnostic")
    ))


def load_catalog(path: Path | None = None) -> dict[str, CatalogEntry]:
    """Load and validate ``patches/catalog.toml``, keyed by patch ID."""
    path = path or paths.PATCH_CATALOG
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if raw.get("version") != 1:
        raise ValueError(f"{path}: unsupported catalog version {raw.get('version')!r}")

    entries: dict[str, CatalogEntry] = {}
    for record in raw.get("patch") or []:
        patch_id = record.get("id")
        if not patch_id:
            raise ValueError(f"{path}: a [[patch]] record is missing 'id'")
        if patch_id in entries:
            raise ValueError(f"{path}: duplicate catalog entry for {patch_id!r}")
        kind = record.get("kind")
        if kind not in KINDS:
            raise ValueError(f"{path}: {patch_id}: kind must be one of {KINDS}, got {kind!r}")
        origin = record.get("origin")
        if origin not in ORIGINS:
            raise ValueError(f"{path}: {patch_id}: origin must be one of {ORIGINS}, got {origin!r}")
        backend = record.get("backend")
        if backend not in BACKENDS:
            raise ValueError(f"{path}: {patch_id}: backend must be one of {BACKENDS}, got {backend!r}")
        state = record.get("state")
        if state not in patchset.STATES:
            raise ValueError(f"{path}: {patch_id}: state must be one of {patchset.STATES}, got {state!r}")
        for field, valid in (("backends", BACKENDS),):
            for value in record.get(field) or ():
                if value not in valid:
                    raise ValueError(
                        f"{path}: {patch_id}: {field} entries must be one of "
                        f"{valid}, got {value!r}"
                    )
        validation_architectures_raw = record.get("validation-architectures") or []
        if not isinstance(validation_architectures_raw, list) or not all(
            isinstance(value, str) and value for value in validation_architectures_raw
        ):
            raise ValueError(
                f"{path}: {patch_id}: validation-architectures must be a list of non-empty strings"
            )
        if len(set(validation_architectures_raw)) != len(validation_architectures_raw):
            raise ValueError(f"{path}: {patch_id}: validation-architectures contains duplicates")

        entries[patch_id] = CatalogEntry(
            patch_id=patch_id,
            kind=kind,
            origin=origin,
            backend=backend,
            state=state,
            upstream_ref=record.get("upstream-ref"),
            retirement=record.get("retirement"),
            external_source=record.get("external-source"),
            plan_item=record.get("plan-item"),
            requires_options=tuple(record.get("requires-options") or ()),
            forbids_options=tuple(record.get("forbids-options") or ()),
            plan_ids=tuple(record.get("plan-ids") or ()),
            backends=tuple(record.get("backends") or ()),
            subsystems=tuple(record.get("subsystems") or ()),
            hardware=tuple(record.get("hardware") or ()),
            validation_architectures=tuple(validation_architectures_raw),
        )
    return entries


# ------------------------------------------------------------ HI83 evidence
#
# Purely observational functions -- NOT wired into any selection/build/apply
# path. patch_catalog.py's own module docstring above states it is
# "read-only, additive metadata... does not change what bigcherry apply/
# bigcherry build actually select"; hard enforcement (refusing to apply/
# build a STATE="validated" patch with stale evidence) is a deliberate,
# separate, deferred decision -- see plan item HI83's notes for why. These
# functions exist so `patch-verify-evidence` and cross_check() (also only
# a diagnostic, not called from any production path today) can report the
# same thing a future enforcement point would check.


def validation_evidence_statuses(
    patch_ids: tuple[str, ...] | list[str],
    *,
    catalog_path: Path | None = None,
    patches_dir: Path | None = None,
    pinned_ref: str | None = None,
    evidence_root: Path | None = None,
    allow_legacy_grandfather: bool = True,
) -> dict[str, patch_validation_evidence.EvidenceCheck]:
    entries = load_catalog(catalog_path)
    modules = {module.patch_id: module for module in patchset.catalog(patches_dir)}

    if pinned_ref is None:
        pinned_ref = campaign_config.load(paths.RECIPES).pinned

    result: dict[str, patch_validation_evidence.EvidenceCheck] = {}
    for patch_id in patch_ids:
        entry = entries.get(patch_id)
        if entry is None:
            result[patch_id] = patch_validation_evidence.EvidenceCheck(
                status="missing-or-stale", problems=("no patches/catalog.toml entry",),
            )
            continue

        module = modules.get(patch_id)
        if module is None:
            result[patch_id] = patch_validation_evidence.EvidenceCheck(
                status="missing-or-stale", problems=("no matching patch module",),
            )
            continue

        result[patch_id] = patch_validation_evidence.verify_validated_patch(
            module, pinned_ref=pinned_ref,
            required_architectures=entry.validation_architectures,
            root=evidence_root, allow_legacy_grandfather=allow_legacy_grandfather,
        )
    return result


def require_validation_evidence(
    patch_ids: tuple[str, ...] | list[str],
    *,
    catalog_path: Path | None = None,
    patches_dir: Path | None = None,
    pinned_ref: str | None = None,
    evidence_root: Path | None = None,
    allow_legacy_grandfather: bool = True,
) -> None:
    """Raise ValueError describing every STATE='validated' patch (among
    patch_ids) whose evidence is missing or stale. Callers decide whether
    and where to treat that as fatal -- this function itself does not gate
    anything on its own."""
    statuses = validation_evidence_statuses(
        patch_ids, catalog_path=catalog_path, patches_dir=patches_dir, pinned_ref=pinned_ref,
        evidence_root=evidence_root, allow_legacy_grandfather=allow_legacy_grandfather,
    )
    failures = [(patch_id, status) for patch_id, status in statuses.items() if not status.ok]
    if not failures:
        return
    detail = "; ".join(
        f"{patch_id}: " + ("; ".join(status.problems) or status.status)
        for patch_id, status in failures
    )
    raise ValueError(f"validated patch evidence check failed: {detail}")


def cross_check(
    catalog_path: Path | None = None,
    patches_dir: Path | None = None,
    *,
    verify_validation_evidence: bool | None = None,
    pinned_ref: str | None = None,
    evidence_root: Path | None = None,
    allow_legacy_grandfather: bool = True,
) -> list[str]:
    """Verify one-to-one coverage between the catalog and discovered patch
    modules, and that each entry's ``state`` matches its module's real STATE
    constant. Returns a list of problem descriptions (empty = clean).

    HI83: when verify_validation_evidence is left at its default (None), it
    auto-enables only for the real repository catalog (catalog_path and
    patches_dir both omitted) -- synthetic temp-catalog unit tests keep
    their pre-HI83 narrow semantics unless they explicitly opt in. This
    function is still only a diagnostic (see the module note above it) --
    nothing currently calls it from a production apply/build path."""
    entries = load_catalog(catalog_path)
    modules = {module.patch_id for module in patchset.catalog(patches_dir)}
    problems: list[str] = []

    orphan_records = sorted(set(entries) - modules)
    for patch_id in orphan_records:
        problems.append(f"catalog entry {patch_id!r} has no matching patch module")

    dangling_modules = sorted(modules - set(entries))
    for patch_id in dangling_modules:
        problems.append(f"patch module {patch_id!r} has no catalog entry")

    module_states = {module.patch_id: module.state for module in patchset.catalog(patches_dir)}
    for patch_id in sorted(set(entries) & modules):
        module_state = module_states[patch_id]
        if entries[patch_id].state != module_state:
            problems.append(
                f"{patch_id}: catalog state {entries[patch_id].state!r} does not match "
                f"module STATE {module_state!r}"
            )

    if verify_validation_evidence is None:
        verify_validation_evidence = catalog_path is None and patches_dir is None

    if verify_validation_evidence:
        statuses = validation_evidence_statuses(
            sorted(set(entries) & modules), catalog_path=catalog_path, patches_dir=patches_dir,
            pinned_ref=pinned_ref, evidence_root=evidence_root,
            allow_legacy_grandfather=allow_legacy_grandfather,
        )
        for patch_id, status in statuses.items():
            if status.ok:
                continue
            problems.append(
                f"{patch_id}: STATE='validated' has no current matching validation evidence: "
                + ("; ".join(status.problems) or status.status)
            )

    return problems


# ------------------------------------------------------------------ snapshot


@dataclass(frozen=True)
class CatalogSnapshot:
    """RE39 (external patch-management review, 2026-08-20): one immutable
    read of BOTH catalogs -- ``patchset.catalog()`` (GROUP/STATE/REQUIRES/
    CONFLICTS, authoritative for selection) and ``patch_catalog.load_catalog()``
    (kind/origin/backend/plan_ids/..., descriptive metadata) -- bundled
    together and keyed consistently by patch_id, so one command/campaign
    invocation reads the patches/ tree and patches/catalog.toml exactly
    once instead of each caller independently re-scanning and re-parsing.

    Not yet threaded through resolve_lane/materialize_source/build planning
    (that is a larger, separate plumbing change touching many call sites --
    left for a follow-up pass rather than risking those paths' existing
    identity/test guarantees in this one). This phase covers the CLI
    surface that visibly double-reads today (``cmd_patches``, which called
    ``patchset.describe()`` and ``patch_catalog.load_catalog()``
    separately) and gives the rest of the codebase a real, tested
    construction point to adopt incrementally.
    """

    root: Path
    modules: tuple["patchset.PatchModule", ...]
    metadata: dict[str, CatalogEntry]
    digest: str

    @property
    def by_id(self) -> dict[str, "patchset.PatchModule"]:
        return {module.patch_id: module for module in self.modules}

    def entry_for(self, patch_id: str) -> CatalogEntry | None:
        return self.metadata.get(patch_id)


def build_snapshot(
    *,
    patches_dir: Path | None = None,
    catalog_path: Path | None = None,
) -> CatalogSnapshot:
    """Build one ``CatalogSnapshot`` -- the single real filesystem read this
    phase covers. ``patches_dir``/``catalog_path`` default to the real
    project locations (``paths.PATCHES``/``paths.PATCH_CATALOG``)."""
    root = patches_dir or paths.PATCHES
    modules = tuple(patchset.catalog(root))
    metadata = load_catalog(catalog_path)
    payload = {
        "modules": [(m.patch_id, m.content_hash) for m in modules],
        "metadata": sorted(metadata.keys()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.blake2b(
        b"bigcherry/catalog-snapshot/v1\0" + encoded, digest_size=16
    ).hexdigest()
    return CatalogSnapshot(root=root, modules=modules, metadata=metadata, digest=digest)


# ------------------------------------------------------------------- explain


@dataclass(frozen=True)
class PatchExplanation:
    """RE43 (external patch-management review, Section 16): everything
    ``bigcherry patches explain <id>`` shows, assembled from data this
    project already has -- CatalogSnapshot (selection + descriptive
    metadata), each patch module's own PROVENANCE dict (source/plan-item),
    and config/recipes.toml's real patch-set/experiment membership. Read-
    only; does not duplicate any of those as a second authority."""

    patch_id: str
    content_hash: str
    group: str
    state: str
    kind: str | None
    origin: str | None
    backend: str | None
    plan_item: str | None
    plan_ids: tuple[str, ...]
    source_id: str | None
    upstream_ref: str | None
    requires: tuple[str, ...]
    conflicts: tuple[str, ...]
    selected_by_patch_sets: tuple[str, ...]
    selected_by_experiments: tuple[str, ...]
    files_touched: tuple[str, ...] = field(default_factory=tuple)


def explain(patch_id: str, snapshot: "CatalogSnapshot", cfg=None) -> PatchExplanation:
    """Assemble a :class:`PatchExplanation` for one patch. Raises
    ``KeyError`` if ``patch_id`` is not in ``snapshot`` -- callers get a
    real error, not a silently empty explanation for a typo'd id."""
    from . import sources as sources_module

    module = snapshot.by_id.get(patch_id)
    if module is None:
        raise KeyError(f"no such patch: {patch_id!r}")
    entry = snapshot.entry_for(patch_id)
    prov = sources_module._patch_provenance(module.path) or {}

    selected_patch_sets: list[str] = []
    selected_experiments: list[str] = []
    if cfg is not None:
        for name, patch_set in cfg.patch_sets.items():
            if patch_id in patch_set.patches:
                selected_patch_sets.append(name)
        for name, experiment in cfg.experiments.items():
            if patch_id in experiment.patches:
                selected_experiments.append(name)

    files_touched: tuple[str, ...] = ()
    try:
        loaded = patchset._load_module(module.path)
        files_touched = tuple(sorted({
            patch.path for patch in getattr(loaded, "PATCHES", ())
        }))
    except Exception:
        # Explain is read-only reporting, not a patch-application path --
        # a module that fails to import (syntax error mid-edit, missing
        # optional dependency) should still explain what the catalog/
        # PROVENANCE data alone can show, not crash the whole command.
        files_touched = ()

    return PatchExplanation(
        patch_id=patch_id,
        content_hash=module.content_hash,
        group=module.group,
        state=module.state,
        kind=entry.kind if entry else None,
        origin=entry.origin if entry else None,
        backend=entry.backend if entry else None,
        plan_item=prov.get("plan-item"),
        plan_ids=entry.plan_ids if entry else (),
        source_id=prov.get("source-id"),
        upstream_ref=module.upstream,
        requires=module.requires,
        conflicts=module.conflicts,
        selected_by_patch_sets=tuple(sorted(selected_patch_sets)),
        selected_by_experiments=tuple(sorted(selected_experiments)),
        files_touched=files_touched,
    )


def render_explanation(info: PatchExplanation) -> str:
    lines = [
        f"patch:          {info.patch_id}",
        f"content hash:   {info.content_hash}",
        f"group / state:  {info.group} / {info.state}",
        f"kind:           {info.kind or 'unknown (not in patches/catalog.toml)'}",
        f"origin:         {info.origin or 'unknown'}",
        f"backend:        {info.backend or 'unknown'}",
        f"source id:      {info.source_id or '-'}",
        f"plan item:      {info.plan_item or '-'}"
        + (f" (also: {', '.join(info.plan_ids)})" if info.plan_ids else ""),
        f"upstream ref:   {info.upstream_ref or '-'}",
        f"requires:       {', '.join(info.requires) or '(none)'}",
        f"conflicts:      {', '.join(info.conflicts) or '(none)'}",
        f"selected by patch-sets: {', '.join(info.selected_by_patch_sets) or '(none)'}",
        f"selected by experiments: {', '.join(info.selected_by_experiments) or '(none)'}",
    ]
    if info.files_touched:
        lines.append(f"files touched:  {', '.join(info.files_touched)}")
    return "\n".join(lines)


# --------------------------------------------------------------------- graph


def dependency_graph(snapshot: "CatalogSnapshot", *, roots: tuple[str, ...] = ()) -> str:
    """A textual REQUIRES/CONFLICTS topology (RE43 Section 16: "a textual
    graph is sufficient initially"). ``roots`` restricts output to just
    those patches and everything they transitively require (their real
    dependency closure, via ``patchset.expand_composition``); an empty
    ``roots`` shows every patch that has any requires/conflicts edge at
    all, skipping isolated nodes so the graph stays readable."""
    by_id = snapshot.by_id
    if roots:
        expansion = patchset.expand_composition(roots, directory=snapshot.root)
        patch_ids = expansion.expanded
    else:
        patch_ids = tuple(sorted(
            patch_id for patch_id, module in by_id.items()
            if module.requires or module.conflicts
            or any(patch_id in other.requires or patch_id in other.conflicts
                   for other in by_id.values())
        ))

    lines: list[str] = []
    for patch_id in patch_ids:
        module = by_id.get(patch_id)
        if module is None:
            continue
        lines.append(patch_id)
        for target in module.requires:
            lines.append(f"  requires -> {target}")
        for target in module.conflicts:
            lines.append(f"  conflicts x {target}")
    return "\n".join(lines) if lines else "(no REQUIRES/CONFLICTS edges in the catalog)"
