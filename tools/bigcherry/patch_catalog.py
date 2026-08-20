"""Declarative patch-stream metadata (RE30 phase 1).

``patches/catalog.toml`` is orthogonal metadata layered on top of each patch
module's existing ``GROUP``/``STATE`` Python constants (see ``patchset.py``).
It answers questions those two labels cannot: whether a patch is BigCherry's
own instrumentation, a backport of specific upstream work, or a port from an
external fork; which backend (hip/vulkan/agnostic) it targets; and, for
backports, where the fix came from and when it should be retired.

GROUP/STATE remain authoritative for patch *selection* (``patchset.py``'s
resolver, recipes.toml patch-set membership, every existing identity hash).
This module is read-only, additive metadata -- it does not change what
``bigcherry apply``/``bigcherry build`` actually select.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

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
        )
    return entries


def cross_check(
    catalog_path: Path | None = None,
    patches_dir: Path | None = None,
) -> list[str]:
    """Verify one-to-one coverage between the catalog and discovered patch
    modules, and that each entry's ``state`` matches its module's real STATE
    constant. Returns a list of problem descriptions (empty = clean)."""
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
