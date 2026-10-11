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


def composed_before(patch_id: str, files, out) -> Path:
    """Write `files` into `out` as the production composition leaves them just before `patch_id` is applied.

    A production patch is written against the composed source (earlier patches create files and move the text its
    anchors sit in), so its mechanics test needs that source, not the bare pin. The production selection of the
    `bigcherry` source is applied in order on a pristine pinned tree, stopping at the patch's own place.
    """
    from . import rebase  # imported here: rebase imports this package's patch loader

    selected = rebase.resolve_selection(source_name="bigcherry", all_patches=False, focal_overlay_patch_id=patch_id)
    vendor = pinned_checkout()
    texts = rebase._overlay_texts()
    overlay_paths = frozenset(texts)
    for module in selected.modules:
        if module.patch_id == patch_id:
            break
        probe = rebase.probe_patch(module, vendor, texts, context_lines=3, previous_revision=None,
                                   revision="composed-before", overlay_paths=overlay_paths)
        if probe.status not in (rebase.STATUS_CLEAN, rebase.STATUS_CLEAN_NOOP):
            raise RuntimeError(f"{module.patch_id} does not apply before {patch_id}: {probe.status}")
    out = Path(out)
    for rel in files:
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel in texts:
            target.write_text(texts[rel], encoding="utf-8")
        else:
            copy_pinned(vendor / rel, target)
    return out
