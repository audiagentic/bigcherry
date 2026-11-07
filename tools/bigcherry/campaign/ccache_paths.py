"""Stable compile paths for ccache (CI and local builds): a slot held exclusively for the whole build.

A build's sources and its build directory are content-addressed (``sources/<id>``, ``builds/<hash>/<plan>``), and both ids
are inside every compile command (``-I``, the source file, the generated include directory). Any change anywhere makes a new
id, so ccache saw a new command for every unchanged translation unit and almost never hit (1-3 % on ~720 compiles locally and
in CI; the engine took 24 minutes and libr3 48 of an 82 minute CI run).

``stable_paths`` gives the build a *slot*: two symlinks, ``<work>/.cc-stable/slotN/src`` and ``.../bld``, pointing at this
build's source and build directories, and sets the variables ``tools/bigcherry/bin/bc-ccache`` rewrites the compile command
with. The same unit in two builds then has the same command text and ccache hits on its content.

Correctness: a slot's links name one build's directories, so a slot is held *exclusively* (an OS file lock, released by the
OS if the holder dies, so there is no stale lock) from before configure to after the last compile. A build that cannot get a
slot (all taken) gets no variables and runs plain ccache: correct, only slower. Nothing but compile steps sees the links;
linking, RUNPATH and every output keep the real paths.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..core.host_lock import HostFileLock, HostLockError

SLOTS = 4
VARIABLES = ("BC_CCACHE_REAL_SRC", "BC_CCACHE_REAL_BLD", "BC_CCACHE_LINK_SRC", "BC_CCACHE_LINK_BLD")
# ccache's own settings while a slot is held, as CI sets them (radiance-build.yml): the slot is the base dir, so the rewritten
# paths become relative and equal whichever slot a build holds; the working directory (a real, per-build directory that the
# wrapper cannot rewrite) must not enter the hash, which ccache does by default whenever debug info is on (-g, RelWithDebInfo);
# the compiler is checked by content; and the cache lives in the work root, not in $HOME. An explicit CCACHE_DIR is kept.
CCACHE_VARIABLES = ("CCACHE_BASEDIR", "CCACHE_NOHASHDIR", "CCACHE_COMPILERCHECK", "CCACHE_DIR")


def _link(path: Path, target: Path) -> None:
    """Point `path` at `target` atomically (a reader never sees the link missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    os.close(fd)
    os.unlink(tmp)
    os.symlink(target, tmp)
    os.replace(tmp, path)


@contextmanager
def stable_paths(work_root: Path, source_root: Path, build_dir: Path, *, slots: int = SLOTS) -> Iterator[int | None]:
    """Hold a free slot for the body and export the variables bc-ccache reads; yield the slot number, or None when no slot is
    free (the body then runs with plain ccache). The variables are restored on exit."""
    base = work_root / ".cc-stable"
    held: HostFileLock | None = None
    number: int | None = None
    for n in range(slots):
        lock = HostFileLock(base / f"slot{n}.lock")
        try:
            lock.acquire(timeout_seconds=0)
        except HostLockError:
            continue
        held, number = lock, n
        break
    saved = {name: os.environ.get(name) for name in VARIABLES + CCACHE_VARIABLES}
    try:
        if held is not None and number is not None:
            slot = base / f"slot{number}"
            _link(slot / "src", source_root.resolve())
            _link(slot / "bld", build_dir.resolve())
            os.environ["BC_CCACHE_REAL_SRC"] = str(source_root.resolve())
            os.environ["BC_CCACHE_REAL_BLD"] = str(build_dir.resolve())
            os.environ["BC_CCACHE_LINK_SRC"] = str(slot / "src")
            os.environ["BC_CCACHE_LINK_BLD"] = str(slot / "bld")
            os.environ["CCACHE_BASEDIR"] = str(slot)
            os.environ["CCACHE_NOHASHDIR"] = "true"
            os.environ["CCACHE_COMPILERCHECK"] = "content"
            os.environ.setdefault("CCACHE_DIR", str(work_root / ".ccache"))
        else:
            for name in VARIABLES:
                os.environ.pop(name, None)
        yield number
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if held is not None:
            held.release()
