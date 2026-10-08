"""Pinned upstream sources for patch-mechanics tests.

A patch test must run against the file as the pinned revision has it. The vendor checkout's working tree is normally
in the patched state, so copying from it makes a test depend on which patches happen to be applied there (a
"missing anchor must fail" case then passes through the guard instead). copy_pinned reads the file from the vendor
checkout's HEAD commit.
"""
from __future__ import annotations

import shutil
import subprocess
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
