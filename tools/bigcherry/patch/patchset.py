"""Discovery and loading of the patch set in ``patches/``.

Patch modules live at the repository root rather than inside the tool because
they are the part a reviewer actually reads: what we change in upstream, and
why. They are Python because an anchored edit is a small piece of logic, not
data -- it has a guard, an expected match count, and sometimes a probe for
which upstream shape it handles.

Modules are loaded by explicit path so ``patches/`` needs no ``__init__.py``
and no ``sys.path`` manipulation, and so numeric ordering prefixes
(``0100_``, ``0200_``) are free to name the file without constraining the
Python identifier.

patch-system PA02/RS02 (runbook section 11): this module is now a
compatibility façade over ``patch_registry``. ``catalog()``/``describe()``
consume normalized :class:`patch_registry.PatchDescriptor` records (which
additionally accept packaged ``<id>/patch.toml`` patches); the legacy
metadata readers, the constant-reading helpers, and the public types
(``PatchModule``, ``PatchInfo``, ...) are unchanged in shape and behavior so
every existing caller keeps working until the loader cutover (RS04) and the
catalog integration (RS03) land.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from . import registry as patch_registry
from ..core import paths
from .apply import FilePatch

STATES: tuple[str, ...] = ("validated", "rejected", "untested")
DEFAULT_STATE = "untested"
DEFAULT_GROUP = "core"


def _load_module(path: Path) -> ModuleType:
    """COMPATIBILITY SHIM (RS04): explicit-path legacy module loader.

    All production implementation loading now goes through
    ``patch_registry.load_implementation`` (descriptor-driven, byte-compile,
    no importlib). This helper remains only for direct explicit-path callers
    that predate the registry (e.g. a test inspecting one specific module's
    constants by file path); it performs no discovery and no path guessing.

    The rationale for byte-compiling the SOURCE directly (deliberately not
    ``importlib``'s standard loader machinery, RV48/RE04 audit fix) is the
    registry's: a mtime-keyed ``__pycache__`` can serve STALE bytecode for a
    file whose on-disk bytes have already changed, and a patch module's
    actual applied effect must never diverge from what ``content_hash`` says
    it is.
    """
    name = f"bigcherry._patches.{path.stem}"
    source = path.read_text(encoding="utf-8")
    code = compile(source, str(path), "exec")
    module = ModuleType(name)
    module.__file__ = str(path)
    # Register before exec so a patch module may import its siblings.
    sys.modules[name] = module
    exec(code, module.__dict__)
    return module


def module_group(path: Path) -> str:
    """The group a patch module belongs to, without fully loading it."""
    return patch_registry.module_group(path)


def module_state(path: Path) -> str:
    """The state (validated/rejected/untested) of a patch module."""
    return patch_registry.module_state(path)


def module_upstream(path: Path) -> str | None:
    """The upstream commit SHA this patch backports from, if any."""
    return patch_registry.module_upstream(path)


@dataclass(frozen=True)
class PatchInfo:
    """Metadata about a single patch module."""
    name: str
    path: Path
    group: str
    state: str
    upstream: str | None

    @property
    def state_valid(self) -> bool:
        return self.state in STATES


@dataclass(frozen=True)
class PatchModule:
    """Canonical, content-identified patch module metadata."""

    patch_id: str
    path: Path
    order: int
    group: str
    state: str
    upstream: str | None
    content_hash: str
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    group_explicit: bool = True
    # RE30 phase 1: carried explicitly so callers resolving a nested (future
    # backend-scoped) catalog don't have to infer the catalog root from
    # ``catalog[0].path.parent`` -- that inference silently points at a
    # module's own subdirectory once discovery becomes recursive, not the
    # catalog root every other module in the same catalog shares.
    catalog_root: Path | None = None
    relative_path: Path | None = None


@dataclass(frozen=True)
class ResolvedPatchSet:
    """The exact named modules selected for an operation."""

    modules: tuple[PatchModule, ...]
    required_state: str | None = None


def discover_modules(root: Path) -> list[Path]:
    """Legacy recursive discovery, retained for RS02 compatibility.

    RS04 retires the remaining non-registry consumer
    (``patch_source_isolation.framework_baseline_digest``); the registry's own
    discovery (root-level simple + ``patch.toml`` packaged) is the normative
    one (runbook section 4).
    """
    """Recursively discover patch module ``.py`` files under ``root``.

    Excludes any path whose relative-to-root component starts with ``_``
    (helpers, ``__pycache__``) at any depth -- not just the filename, so a
    future nested catalog directory (e.g. a ``_shared/`` helper folder) is
    excluded the same way a leading-underscore file is today. For today's
    flat ``patches/`` layout this returns exactly what ``directory.glob(
    "*.py")`` did; it only starts differing once nested catalog directories
    (e.g. a future ``patches/vulkan/``) exist.
    """
    if not root.is_dir():
        return []
    found = []
    for path in root.rglob("*.py"):
        if any(part.startswith("_") for part in path.relative_to(root).parts):
            continue
        found.append(path)
    return sorted(found)


def describe(directory=None) -> list[PatchInfo]:
    """Describe every patch (name, group, state, upstream) without importing.

    RS02: backed by the registry, so packaged patches describe the same way
    flat ones do. For legacy modules this is exactly the old output.

    RV80 follow-up (GPT ruling b): this is a REPORTING/observational facade,
    so it uses the registry's report-only path (describe_all), which preserves
    a raw malformed lifecycle state (state_valid=False) instead of raising.
    Strict resolution/build paths still fail closed via load_registry.
    """
    directory = directory or paths.PATCHES
    root = directory.resolve()
    descriptors = patch_registry.describe_all(directory)
    return [
        PatchInfo(
            name=descriptor.patch_id,
            path=root / descriptor.implementation_path,
            group=descriptor.group,
            state=descriptor.state,
            upstream=descriptor.upstream,
        )
        for descriptor in descriptors
    ]


def _module_from_descriptor(descriptor, root: Path) -> PatchModule:
    """Compatibility mapping registry descriptor -> legacy PatchModule shape."""
    path = root / descriptor.implementation_path
    return PatchModule(
        patch_id=descriptor.patch_id,
        path=path,
        order=descriptor.order,
        group=descriptor.group,
        state=descriptor.state,
        upstream=descriptor.upstream,
        content_hash=descriptor.implementation_digest,
        requires=descriptor.requires,
        conflicts=descriptor.conflicts,
        group_explicit=(
            True if descriptor.representation == patch_registry.REPRESENTATION_PACKAGED
            else patch_registry.group_is_explicit(path)
        ),
        catalog_root=root,
        relative_path=descriptor.implementation_path,
    )


def catalog(directory=None) -> list[PatchModule]:
    """Return every patch (flat or packaged) as a canonical, hashed module
    descriptor.

    RS02: built from the registry's normalized descriptors; for today's flat
    tree this is field-for-field identical to the previous output (same IDs,
    order, groups, states, upstream refs, requires/conflicts, content hashes,
    and duplicate rejection). Packaged patches simply appear as extra entries
    once they exist.
    """
    directory = directory or paths.PATCHES
    registry = patch_registry.load_registry(directory)
    return [
        _module_from_descriptor(descriptor, registry.root)
        for descriptor in registry.descriptors
    ]


def topological_order(
    patch_ids: Sequence[str],
    *,
    modules: Mapping[str, PatchModule],
) -> tuple[str, ...]:
    """Deterministic TRUE topological order of an EXACT, dependency-complete
    patch id set: every REQUIRES dependency precedes its dependent, and the
    canonical ``(order, patch_id)`` key is used ONLY to pick among the
    currently READY nodes (minimum first) — never to re-sort the whole set.

    RV80/B4: a global ``sorted(ids, key=(order, id))`` silently destroys the
    dependency order whenever numbering and REQUIRES disagree (a ``0100``
    child that REQUIRES a ``0200`` parent was emitted before its parent),
    which changes the apply sequence and therefore which anchors exist when
    a patch is applied. Callers must have already validated that the set is
    dependency-complete (every ``requires`` target is in ``patch_ids``); an
    unsatisfiable remainder is reported as a cycle.
    """
    remaining = set(patch_ids)
    unmet = {
        pid: {dep for dep in modules[pid].requires if dep in remaining}
        for pid in remaining
    }
    ordered: list[str] = []
    while remaining:
        ready = [pid for pid in remaining if not unmet[pid]]
        if not ready:
            raise ValueError(
                "REQUIRES cycle detected among: " + ", ".join(sorted(remaining))
            )
        pick = min(ready, key=lambda pid: (modules[pid].order, pid))
        ordered.append(pick)
        remaining.discard(pick)
        for pid in remaining:
            unmet[pid].discard(pick)
    return tuple(ordered)


def resolve_exact(
    patch_ids: tuple[str, ...] | list[str],
    *,
    directory: Path | None = None,
    required_state: str | None = None,
    allow_rejected: bool = False,
    context_ids: frozenset[str] = frozenset(),
) -> ResolvedPatchSet:
    """Resolve a complete explicit module set without adding dependencies.

    The authoritative exact-composition validator (RV80): unknown IDs,
    invalid/rejected states, duplicate IDs, missing explicit requires, and
    internal conflicts ALL fail closed. The returned module order is a true
    topological order (``topological_order``), not a numeric re-sort.

    ``context_ids`` (HI134): patch IDs considered already-selected for the
    purpose of the REQUIRES/conflicts check ONLY -- e.g. an experiment
    overlay's own patch whose REQUIRES is already satisfied by the base
    patch set it builds on top of. They are never added to the returned
    module set or its ordering; a caller that also wants them present must
    still name them in ``patch_ids`` (or merge the two sets itself), exactly
    as before. This exists because the base set and an experiment overlay
    are resolved as two separate exact selections (campaign/resolution.py),
    so an experiment patch's REQUIRES on a base-set module previously had no
    way to be satisfied without also re-adding that module to the overlay
    itself -- which then collided with the overlay/base disjointness check.
    """
    modules = {module.patch_id: module for module in catalog(directory)}
    ids = tuple(patch_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("patch selection contains duplicate canonical IDs")
    unknown = sorted(set(ids) - set(modules))
    if unknown:
        raise ValueError(f"unknown patch module(s): {', '.join(unknown)}")
    selected = tuple(modules[item] for item in ids)
    for module in selected:
        if module.state not in STATES:
            raise ValueError(f"{module.patch_id}: invalid STATE={module.state!r}")
        if module.state == "rejected" and not allow_rejected:
            raise ValueError(f"{module.patch_id}: rejected patch requires --allow-rejected")
        if required_state is not None and module.state != required_state:
            raise ValueError(
                f"{module.patch_id}: state {module.state!r} does not satisfy "
                f"required state {required_state!r}"
            )
    selected_ids = {module.patch_id for module in selected} | context_ids
    for module in selected:
        missing = sorted(set(module.requires) - selected_ids)
        if missing:
            raise ValueError(
                f"{module.patch_id} requires explicitly selected module(s): {', '.join(missing)}"
            )
        conflicts = sorted(set(module.conflicts) & selected_ids)
        if conflicts:
            raise ValueError(
                f"{module.patch_id} conflicts with selected module(s): {', '.join(conflicts)}"
            )
    ordered_ids = topological_order(ids, modules=modules)
    return ResolvedPatchSet(
        tuple(modules[pid] for pid in ordered_ids), required_state
    )


@dataclass(frozen=True)
class CompositionExpansion:
    """RE42: what a caller asked for, versus what REQUIRES pulled in with
    it. Kept separate deliberately -- a caller/CLI can show "you asked for
    X, this also pulls in Y, Z because X requires them" rather than
    silently expanding, matching this project's standing rule that a
    dependency can never be silently added without being visible."""

    requested: tuple[str, ...]
    expanded: tuple[str, ...]

    @property
    def pulled_in(self) -> tuple[str, ...]:
        requested = set(self.requested)
        return tuple(patch_id for patch_id in self.expanded if patch_id not in requested)


def expand_composition(
    patch_ids: tuple[str, ...] | list[str],
    *,
    directory: Path | None = None,
) -> CompositionExpansion:
    """Compute the full REQUIRES closure of ``patch_ids``, in a stable
    topological order (dependencies before dependents, then canonical
    ``(order, patch_id)`` as a tie-breaker).

    This is a NEW layer ABOVE ``resolve_exact()`` (RE42, external
    patch-management review 2026-08-20) -- ``resolve_exact()`` stays the
    fail-closed exact layer that requires every dependency to already be in
    the explicitly-selected set; this function is what a caller uses to
    COMPUTE that complete set from a minimal request, so
    ``config/recipes.toml`` authors stop hand-listing full dependency
    chains. Nothing calls this automatically yet -- ``resolve_lane`` is
    unchanged and does not auto-expand, so no existing recipe/experiment's
    behavior or identity changes by this function merely existing.

    Raises ``ValueError`` on an unknown patch ID or a REQUIRES cycle
    (reported with the exact cycle path, not just "a cycle exists").
    """
    modules = {module.patch_id: module for module in catalog(directory)}
    requested = tuple(patch_ids)
    unknown = sorted(set(requested) - set(modules))
    if unknown:
        raise ValueError(f"unknown patch module(s): {', '.join(unknown)}")

    order: list[str] = []
    seen: set[str] = set()
    in_progress: list[str] = []

    def visit(patch_id: str) -> None:
        if patch_id in seen:
            return
        if patch_id in in_progress:
            cycle = " -> ".join(in_progress[in_progress.index(patch_id):] + [patch_id])
            raise ValueError(f"REQUIRES cycle detected: {cycle}")
        in_progress.append(patch_id)
        for dependency in modules[patch_id].requires:
            if dependency not in modules:
                raise ValueError(
                    f"{patch_id} REQUIRES unknown module {dependency!r}"
                )
            visit(dependency)
        in_progress.pop()
        seen.add(patch_id)
        order.append(patch_id)

    for patch_id in requested:
        visit(patch_id)

    # RV80/B4: the DFS order is already dependency-first; order it with the
    # true topological picker (minimum ready (order, patch_id)) instead of a
    # GLOBAL re-sort by (order, patch_id), which would put a numerically
    # earlier dependent ahead of its numerically later dependency.
    expanded = topological_order(order, modules=modules)
    return CompositionExpansion(requested=requested, expanded=expanded)


def load_resolved(selection: ResolvedPatchSet) -> list[FilePatch]:
    """Load only the already-resolved canonical modules, in catalog order.

    RS04: loading goes through ``patch_registry.load_implementation``
    (descriptor-driven, byte-compile, no importlib, no path guessing).
    """
    patches: list[FilePatch] = []
    registries: dict[Path, patch_registry.PatchRegistry] = {}
    for module in selection.modules:
        root = module.catalog_root or paths.PATCHES
        registry = registries.get(root)
        if registry is None:
            registry = patch_registry.load_registry(root)
            registries[root] = registry
        descriptor = registry.get(module.patch_id)
        patches.extend(patch_registry.load_implementation(descriptor, root=root))
    return patches


def upstream_landed(commit: str, root: Path) -> bool | None:
    """Check if an upstream commit is an ancestor of HEAD.

    Returns True/False if known, None if unknown (e.g., in a shallow clone
    that doesn't have the commit).
    """
    try:
        cmd = ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, "HEAD"]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0:
            return True
        elif result.returncode == 1:
            return False
        else:
            return None  # Unknown (e.g., commit not found)
    except Exception:
        return None


def parse_filter(arg: str | None) -> frozenset[str] | None:
    """Parse a comma-separated filter string.

    None (flag absent) = no filter (all).
    Empty string or "all" = empty set (select nothing).
    "core,upstream-fixes" = frozenset with those elements.
    """
    if arg is None:
        return None
    items = [v.strip() for v in arg.split(",") if v.strip()]
    return frozenset(items)


def load_patches(
    directory: Path | None = None,
    *,
    groups: frozenset[str] | None = None,
    states: frozenset[str] | None = None,
) -> list[FilePatch]:
    """Load every patch module matching the given criteria, in filename order.

    Order matters: edits within a file are applied in declaration order, and
    two modules touching the same file must run in a predictable sequence.
    Hence the numeric prefixes.

    Args:
        directory: path to patches directory (default: paths.PATCHES)
        groups: if given, only load patches in these groups (None = all groups)
        states: if given, only load patches in these states (None = all states)
    """
    directory = directory or paths.PATCHES
    # RS04: discovery AND loading go through the registry (root-level simple
    # + patch.toml packaged; byte-compile, no importlib). For today's flat
    # tree the filter and load behavior is exactly the old one.
    registry = patch_registry.load_registry(directory)
    patches: list[FilePatch] = []
    for descriptor in registry.descriptors:
        if groups is not None and descriptor.group not in groups:
            continue
        if states is not None and descriptor.state not in states:
            continue
        patches.extend(patch_registry.load_implementation(descriptor, root=registry.root))
    return patches
