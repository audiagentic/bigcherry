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

from ..core import engines, paths


def _vendor_root(engine: str, override: Path | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    return engines.load(paths.REPO_ROOT, engine).vendor_root(paths.primary_root()).resolve()


def copy_pinned(engine: str, src, dst, *, vendor_root: Path | None = None):
    """Like shutil.copy2(src, dst); a src inside the vendor checkout is taken from its HEAD commit, not the working tree."""
    vendor = _vendor_root(engine, vendor_root)
    src, dst = Path(src).resolve(), Path(dst)
    try:
        rel = src.relative_to(vendor).as_posix()
    except ValueError:
        shutil.copy2(src, dst)
        return
    blob = subprocess.run(["git", "-C", str(vendor), "show", f"HEAD:{rel}"], capture_output=True, check=True).stdout
    dst.write_bytes(blob)


@functools.lru_cache(maxsize=None)
def _pinned_checkout_cached(engine: str, vendor: str, revision: str) -> Path:
    """A pristine copy of the vendor checkout's HEAD commit in a temporary directory, made once per process.

    The composer's probe path reads source files from the tree it is given. Handing it the vendor working tree makes
    every already-applied production patch report UPSTREAM_ABSORBED, and files a patch creates look like upstream
    files. The copy is removed at interpreter exit.
    """
    target = Path(tempfile.mkdtemp(prefix=f"bigcherry-pinned-{engine}-"))
    atexit.register(shutil.rmtree, target, True)
    archive = subprocess.run(["git", "-C", vendor, "archive", "--format=tar", revision], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target, filter="data")
    return target


def pinned_checkout(engine: str, *, vendor_root: Path | None = None) -> Path:
    """Pristine checkout of the selected engine's resolved vendor HEAD, cached by engine + resolved pin."""
    vendor = _vendor_root(engine, vendor_root)
    revision = subprocess.run(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    return _pinned_checkout_cached(engine, str(vendor), revision)


def read_pinned(engine: str, src, *, vendor_root: Path | None = None) -> str:
    """Text of a vendor file as the pinned revision has it."""
    vendor = _vendor_root(engine, vendor_root)
    rel = Path(src).resolve().relative_to(vendor).as_posix()
    blob = subprocess.run(["git", "-C", str(vendor), "show", f"HEAD:{rel}"], capture_output=True, check=True).stdout
    return blob.decode("utf-8")


# Composed file texts by the tuple of production patch ids applied to reach them. Mechanics tests of several patches
# run in one process; a later request starts from the longest sequence already composed instead of from the pin.
_COMPOSED: dict[tuple[str, str, tuple[str, ...]], dict[str, str]] = {}


def production_source(engine: str, source_name: str, files, out, *, stop_before: str | None = None, without: str | None = None) -> Path:
    """Write `files` into `out` as the production composition (the selected source) leaves them.

    A production patch is written against the composed source (earlier patches create files and move the text its
    anchors sit in), so its mechanics test needs that source, not the bare pin. The selection is applied in the
    composer's order on a pristine pinned tree:
      stop_before=<id>  everything ordered before that patch (the source the patch itself is applied to);
      without=<id>      the whole set except that patch (for a patch whose test applies it last, in or out of the set).
    A patch of the set that does not apply raises RuntimeError naming it.
    """
    from . import rebase  # imported here: rebase imports this package's patch loader

    selected = rebase.resolve_selection(engine=engine, source_name=source_name, all_patches=False,
                                        focal_overlay_patch_id=stop_before or without)
    modules = []
    for module in selected.modules:
        if module.patch_id == stop_before:
            break
        if module.patch_id != without:
            modules.append(module)
    want = tuple(module.patch_id for module in modules)
    candidates = [
        ids for (cached_engine, cached_source, ids) in _COMPOSED
        if cached_engine == engine and cached_source == source_name and want[:len(ids)] == ids
    ]
    start = max(candidates, key=len, default=())
    cache_key = (engine, source_name, start)
    texts = dict(_COMPOSED[cache_key]) if start else rebase._overlay_texts(engine)
    overlay_paths = frozenset(rebase._overlay_texts(engine))
    vendor = pinned_checkout(engine)
    for module in modules[len(start):]:
        probe = rebase.probe_patch("llamacpp", module, vendor, texts, context_lines=3, previous_revision=None,
                                   revision="production-source", overlay_paths=overlay_paths)
        if probe.status not in (rebase.STATUS_CLEAN, rebase.STATUS_CLEAN_NOOP):
            raise RuntimeError(f"{module.patch_id} does not apply in the production composition: {probe.status}")
    _COMPOSED[(engine, source_name, want)] = dict(texts)
    out = Path(out)
    for rel in files:
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel in texts:
            target.write_text(texts[rel], encoding="utf-8")
        else:
            copy_pinned(engine, vendor / rel, target)
    return out


def production_order(engine: str, source_name: str) -> list[str]:
    """Patch ids of the production selection in the composer's order."""
    from . import rebase

    return [module.patch_id for module in rebase.resolve_selection(engine=engine, source_name=source_name, all_patches=False).modules]
