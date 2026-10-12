"""Pinned upstream sources for patch-mechanics tests.

A patch test must run against the file as the pinned revision has it. The vendor checkout's working tree is normally
in the patched state, so copying from it makes a test depend on which patches happen to be applied there (a
"missing anchor must fail" case then passes through the guard instead). copy_pinned reads the file from the vendor
checkout's HEAD commit, and pinned_checkout gives a whole pristine tree for tests that hand one to the composer.
"""
from __future__ import annotations

import atexit
import functools
import io
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from ..core import paths

# The vendor checkout belongs to the primary checkout; a slice worktree has none of its own.
_VENDOR = paths.llama_root()


def copy_pinned(src, dst):
    """Like shutil.copy2(src, dst); a src inside the vendor checkout is taken from its HEAD commit, not the working tree."""
    src, dst = Path(src).resolve(), Path(dst)
    try:
        rel = src.relative_to(_VENDOR).as_posix()
    except ValueError:
        shutil.copy2(src, dst)
        return
    blob = subprocess.run(["git", "-C", str(_VENDOR), "show", f"HEAD:{rel}"], capture_output=True, check=True).stdout
    dst.write_bytes(blob)


@functools.lru_cache(maxsize=1)
def pinned_checkout() -> Path:
    """A pristine copy of the vendor checkout's HEAD commit in a temporary directory, made once per process.

    The composer's probe path reads source files from the tree it is given. Handing it the vendor working tree makes
    every already-applied production patch report UPSTREAM_ABSORBED, and files a patch creates look like upstream
    files. The copy is removed at interpreter exit.
    """
    target = Path(tempfile.mkdtemp(prefix="bigcherry-pinned-"))
    atexit.register(shutil.rmtree, target, True)
    archive = subprocess.run(["git", "-C", str(_VENDOR), "archive", "--format=tar", "HEAD"], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target, filter="data")
    return target


def read_pinned(src) -> str:
    """Text of a vendor file as the pinned revision has it."""
    rel = Path(src).resolve().relative_to(_VENDOR).as_posix()
    blob = subprocess.run(["git", "-C", str(_VENDOR), "show", f"HEAD:{rel}"], capture_output=True, check=True).stdout
    return blob.decode("utf-8")


# Composed file texts by the tuple of production patch ids applied to reach them. Mechanics tests of several patches
# run in one process; a later request starts from the longest sequence already composed instead of from the pin.
_COMPOSED: dict[tuple, dict] = {}


def production_source(files, out, *, stop_before: str | None = None, without: str | None = None) -> Path:
    """Write `files` into `out` as the production composition (source `bigcherry`) leaves them.

    A production patch is written against the composed source (earlier patches create files and move the text its
    anchors sit in), so its mechanics test needs that source, not the bare pin. The selection is applied in the
    composer's order on a pristine pinned tree:
      stop_before=<id>  everything ordered before that patch (the source the patch itself is applied to);
      without=<id>      the whole set except that patch (for a patch whose test applies it last, in or out of the set).
    A patch of the set that does not apply raises RuntimeError naming it.
    """
    from . import rebase  # imported here: rebase imports this package's patch loader

    selected = rebase.resolve_selection(source_name="bigcherry", all_patches=False,
                                        focal_overlay_patch_id=stop_before or without)
    modules = []
    for module in selected.modules:
        if module.patch_id == stop_before:
            break
        if module.patch_id != without:
            modules.append(module)
    want = tuple(module.patch_id for module in modules)
    start = max((ids for ids in _COMPOSED if want[:len(ids)] == ids), key=len, default=())
    texts = dict(_COMPOSED[start]) if start else rebase._overlay_texts()
    overlay_paths = frozenset(rebase._overlay_texts())
    vendor = pinned_checkout()
    for module in modules[len(start):]:
        probe = rebase.probe_patch(module, vendor, texts, context_lines=3, previous_revision=None,
                                   revision="production-source", overlay_paths=overlay_paths)
        if probe.status not in (rebase.STATUS_CLEAN, rebase.STATUS_CLEAN_NOOP):
            raise RuntimeError(f"{module.patch_id} does not apply in the production composition: {probe.status}")
    _COMPOSED[want] = dict(texts)
    out = Path(out)
    for rel in files:
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel in texts:
            target.write_text(texts[rel], encoding="utf-8")
        else:
            copy_pinned(vendor / rel, target)
    return out


def production_order() -> list[str]:
    """Patch ids of the production selection in the composer's order."""
    from . import rebase

    return [module.patch_id for module in rebase.resolve_selection(source_name="bigcherry", all_patches=False).modules]
