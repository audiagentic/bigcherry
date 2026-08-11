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

import importlib.util
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
    name = f"bigcherry._patches.{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load patch module {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so a patch module may import its siblings.
    sys.modules[name] = module
    spec.loader.exec_module(module)
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


def describe(directory=None) -> list[PatchInfo]:
    """Describe every patch module (name, group, state, upstream) without importing."""
    directory = directory or paths.PATCHES
    if not directory.is_dir():
        return []

    result = []
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        info = PatchInfo(
            name=path.stem,
            path=path,
            group=module_group(path),
            state=module_state(path),
            upstream=module_upstream(path),
        )
        result.append(info)
    return result


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
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue

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
