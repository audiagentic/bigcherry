"""Materialize and restore the BigCherry source overlay.

Overlay writes are a separate transaction from anchored patch application.  The
helpers in this module deliberately depend only on the patch domain and core
paths so CLI and release workflows can share the same byte-level behavior
without importing the compatibility entrypoint.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..core import paths


def copy_overlay(
    root: Path,
    *,
    dry_run: bool,
    backup: dict[str, str | None] | None = None,
    sim_texts: dict[str, str] | None = None,
) -> list[str]:
    """Mirror ``src/`` onto the checkout and return paths that would change.

    ``backup`` captures original target content before each write so a caller
    can roll back this transaction if anchored patch application fails.
    ``sim_texts`` captures post-write content even during a dry run so the
    patch engine sees the same overlay bytes that a real apply would see.
    """
    written: list[str] = []
    for source in sorted(paths.SRC_OVERLAY.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(paths.SRC_OVERLAY)
        target = root / relative
        text = source.read_text(encoding="utf-8")
        # Read raw bytes so a stale CRLF target is not mistaken for an
        # equivalent LF-only overlay by universal-newline translation.
        if target.is_file() and target.read_bytes().decode("utf-8") == text:
            continue
        relative_str = str(relative).replace("\\", "/")
        if backup is not None and relative_str not in backup:
            backup[relative_str] = (
                target.read_text(encoding="utf-8") if target.is_file() else None
            )
        if sim_texts is not None:
            sim_texts[relative_str] = text
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="")
        written.append(relative_str)
    return written


def restore_overlay(root: Path, backup: dict[str, str | None]) -> None:
    """Undo ``copy_overlay`` writes using the captured original contents."""
    for relative_str, original in backup.items():
        target = root / relative_str
        try:
            if original is None:
                if target.is_file():
                    target.unlink()
            else:
                nofollow = getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(
                    str(target),
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC | nofollow,
                    0o644,
                )
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                    handle.write(original)
        except OSError:
            # One failed restore must not prevent the remaining overlay files
            # from being returned to their captured state.
            pass
