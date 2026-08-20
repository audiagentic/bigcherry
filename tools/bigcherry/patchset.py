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
"""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from . import paths
from .patcher import FilePatch

STATES: tuple[str, ...] = ("validated", "rejected", "untested")
DEFAULT_STATE = "untested"
DEFAULT_GROUP = "core"


def _load_module(path: Path) -> ModuleType:
    """Compile and exec the module's SOURCE directly -- deliberately not
    ``importlib``'s standard loader machinery (RV48/RE04 audit fix).
    ``SourceFileLoader.exec_module`` writes/reads a mtime-keyed
    ``__pycache__/*.pyc`` bytecode cache; two writes to the same patch file
    within one mtime-resolution tick (routine in tests, and not impossible
    in fast development iteration) can serve STALE bytecode for a file
    whose on-disk bytes -- and ``catalog()``'s own sha256 content_hash --
    have already changed. A patch module's actual applied effect must never
    diverge from what ``content_hash`` says it is; compiling straight from
    freshly-read source bypasses that cache entirely, every call.
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


def _constant(path: Path, name: str, pattern: str) -> str | None:
    """Extract a module-level constant from a patch file without importing."""
    try:
        content = path.read_text(encoding='utf-8')
        match = re.search(rf'^{name}\s*=\s*["\']({pattern})["\']', content, re.MULTILINE)
        return match.group(1) if match else None
    except Exception:
        return None


def module_group(path: Path) -> str:
    """The group a patch module belongs to, without fully loading it."""
    return _constant(path, "GROUP", r"[\w-]+") or DEFAULT_GROUP


def module_state(path: Path) -> str:
    """The state (validated/rejected/untested) of a patch module."""
    return _constant(path, "STATE", r"[\w-]+") or DEFAULT_STATE


def module_upstream(path: Path) -> str | None:
    """The upstream commit SHA this patch backports from, if any."""
    return _constant(path, "UPSTREAM", r"[0-9a-fA-F]{7,40}")


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


def _literal_constant(path: Path, name: str) -> object | None:
    """Read a literal module constant without importing executable code."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError):
                return None
    return None


def _constant_strings(path: Path, name: str) -> tuple[str, ...]:
    value = _literal_constant(path, name)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list)) and all(
        isinstance(item, str) and item for item in value
    ):
        return tuple(value)
    raise ValueError(f"{path.name}: {name} must be a string or list/tuple of strings")


def _module_order(stem: str) -> int:
    match = re.match(r"^(\d+)_", stem)
    return int(match.group(1)) if match else 2**31


def discover_modules(root: Path) -> list[Path]:
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
    """Describe every patch module (name, group, state, upstream) without importing."""
    directory = directory or paths.PATCHES
    if not directory.is_dir():
        return []

    result = []
    for path in discover_modules(directory):
        info = PatchInfo(
            name=path.stem,
            path=path,
            group=module_group(path),
            state=module_state(path),
            upstream=module_upstream(path),
        )
        result.append(info)
    return result


def catalog(directory=None) -> list[PatchModule]:
    """Return every patch as a canonical, hashed module descriptor."""
    directory = directory or paths.PATCHES
    if not directory.is_dir():
        return []
    result: list[PatchModule] = []
    seen_ids: dict[str, Path] = {}
    for path in discover_modules(directory):
        patch_id = path.stem
        if patch_id in seen_ids:
            raise ValueError(
                f"duplicate patch ID {patch_id!r}: {seen_ids[patch_id]} and {path}"
            )
        seen_ids[patch_id] = path
        result.append(
            PatchModule(
                patch_id=patch_id,
                path=path,
                order=_module_order(patch_id),
                group=module_group(path),
                state=module_state(path),
                upstream=module_upstream(path),
                content_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
                requires=_constant_strings(path, "REQUIRES"),
                conflicts=_constant_strings(path, "CONFLICTS"),
                group_explicit=_literal_constant(path, "GROUP") is not None,
                catalog_root=directory,
                relative_path=path.relative_to(directory),
            )
        )
    return sorted(result, key=lambda module: (module.order, module.patch_id))


def resolve_exact(
    patch_ids: tuple[str, ...] | list[str],
    *,
    directory: Path | None = None,
    required_state: str | None = None,
    allow_rejected: bool = False,
) -> ResolvedPatchSet:
    """Resolve a complete explicit module set without adding dependencies."""
    modules = {module.patch_id: module for module in catalog(directory)}
    ids = tuple(patch_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("patch selection contains duplicate canonical IDs")
    unknown = sorted(set(ids) - set(modules))
    if unknown:
        raise ValueError(f"unknown patch module(s): {', '.join(unknown)}")
    selected = tuple(sorted((modules[item] for item in ids), key=lambda m: (m.order, m.patch_id)))
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
    selected_ids = {module.patch_id for module in selected}
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
    return ResolvedPatchSet(selected, required_state)


def load_resolved(selection: ResolvedPatchSet) -> list[FilePatch]:
    """Load only the already-resolved canonical modules, in catalog order."""
    patches: list[FilePatch] = []
    for descriptor in selection.modules:
        module = _load_module(descriptor.path)
        found = getattr(module, "PATCHES", None)
        if found is None:
            single = getattr(module, "PATCH", None)
            found = [single] if single is not None else []
        if not found:
            raise ImportError(
                f"{descriptor.patch_id} defines neither PATCH nor PATCHES"
            )
        patches.extend(found)
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
    if not directory.is_dir():
        return []

    patches: list[FilePatch] = []
    for path in discover_modules(directory):
        # Filter by group and state metadata (without loading the module)
        patch_group = module_group(path)
        patch_state = module_state(path)

        if groups is not None and patch_group not in groups:
            continue
        if states is not None and patch_state not in states:
            continue

        module = _load_module(path)
        found = getattr(module, "PATCHES", None)
        if found is None:
            single = getattr(module, "PATCH", None)
            found = [single] if single is not None else []
        if not found:
            raise ImportError(
                f"{path.name} defines neither PATCH nor PATCHES")
        patches.extend(found)
    return patches
