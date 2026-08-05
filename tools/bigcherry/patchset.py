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
import sys
from pathlib import Path
from types import ModuleType

from . import paths
from .patcher import FilePatch


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


def load_patches(directory: Path | None = None) -> list[FilePatch]:
    """Load every patch module, in filename order.

    Order matters: edits within a file are applied in declaration order, and
    two modules touching the same file must run in a predictable sequence.
    Hence the numeric prefixes.
    """
    directory = directory or paths.PATCHES
    if not directory.is_dir():
        return []

    patches: list[FilePatch] = []
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
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
