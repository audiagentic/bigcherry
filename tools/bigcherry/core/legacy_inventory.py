"""WSL06 read-only inventory of legacy work-root objects.

No directory, symlink or manifest establishes ownership or rebuildability.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import run_manifest as rm


def _occupied_bytes(path: Path, device: int) -> tuple[int, bool]:
    """Measure payload without following links, crossing devices or double-counting hardlinks."""
    seen: set[tuple[int, int]] = set()
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            st = current.lstat()
        except OSError:
            return total, False
        if st.st_dev != device or current.is_symlink():
            continue
        identity = (st.st_dev, st.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        if not current.is_dir():
            total += st.st_size
            continue
        try:
            with os.scandir(current) as scan:
                children = [Path(entry.path) for entry in scan]
        except OSError:
            return total, False
        stack.extend(children)
    return total, True


def inventory_legacy(root: Path, *, measure: bool = False) -> dict:
    """Classify all immediate roots plus runs/worktrees/builds/sources; protect everything."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("work root must be an existing non-symlink directory")
    entries: list[dict] = []
    identities: dict[tuple[int, int], str] = {}

    def append(path: Path, category: str, evidence: str) -> None:
        try:
            st = path.lstat()
        except OSError as exc:
            entries.append({
                "path": str(path), "category": category, "evidence": "unreadable",
                "size_bytes": None, "size_complete": False, "alias_of": None,
                "action": "protected", "reasons": [f"lstat failed: {exc}"],
            })
            return
        symlink = path.is_symlink()
        identity = (st.st_dev, st.st_ino)
        alias = identities.get(identity)
        if alias is None and not symlink:
            identities[identity] = str(path)
        size = None
        complete = False
        if measure and not symlink and alias is None:
            size, complete = _occupied_bytes(path, st.st_dev)
        reasons = ["ownership, typed references, active consumers and rebuildability unverified"]
        if symlink:
            reasons.append("symlink: never traverse or authorize cleanup")
        if alias:
            reasons.append("duplicate device/inode: do not count or reclaim twice")
        if not complete and measure and not symlink and alias is None:
            reasons.append("size inspection incomplete: measured bytes are a lower bound")
        entries.append({
            "path": str(path), "category": category, "evidence": evidence,
            "device": st.st_dev, "inode": st.st_ino, "is_symlink": symlink,
            "size_bytes": size, "size_complete": complete,
            "alias_of": alias, "action": "protected", "reasons": reasons,
        })

    def children(path: Path) -> list[Path]:
        if path.is_symlink() or not path.is_dir():
            return []
        try:
            return sorted(path.iterdir())
        except OSError as exc:
            raise ValueError(f"unable to enumerate {path}: {exc}") from exc

    typed = {"runs", "builds", "sources", "worktrees"}
    for item in children(root):
        if item.name not in typed or item.is_symlink() or not item.is_dir():
            append(item, "unclassified-root", "unverified")
            continue
        if item.name == "runs":
            for kind in children(item):
                if kind.is_symlink() or not kind.is_dir() or kind.name not in rm.RUN_KINDS:
                    append(kind, "unclassified-run-kind", "unverified")
                    continue
                for run in children(kind):
                    evidence = "unverified"
                    if not run.is_symlink() and run.is_dir():
                        try:
                            m = rm.read_manifest(run)
                            if m["kind"] == kind.name and m["run_id"] == run.name:
                                evidence = "valid-manifest-not-ownership-proof"
                        except (rm.ManifestError, OSError, KeyError):
                            pass
                    append(run, "run", evidence)
        elif item.name == "builds":
            for group in children(item):
                if group.is_symlink() or not group.is_dir():
                    append(group, "unclassified-build", "unverified")
                    continue
                for build in children(group):
                    append(build, "build", "metadata-unverified")
        elif item.name == "sources":
            for source in children(item):
                append(source, "source", "metadata-unverified")
        elif item.name == "worktrees":
            for checkout in children(item):
                evidence = "git-pointer-not-ownership-proof" if (checkout / ".git").exists() else "unverified"
                append(checkout, "worktree", evidence)
    return {
        "schema_version": 1, "root": str(root), "read_only": True,
        "entries": entries,
        "totals": {
            "entries": len(entries),
            "protected": len(entries),
            "known_measured_bytes": sum(e["size_bytes"] or 0 for e in entries),
            "unmeasured_entries": sum(not e["size_complete"] for e in entries),
            "incomplete": any(not e["size_complete"] for e in entries),
        },
        "notes": [
            "Size is logical file payload, not allocated disk blocks or a filesystem-wide reconciliation.",
            "Inventory covers work-root top-level objects, run leaves, build leaves, source entries and worktree entries.",
            "No entry is an adoption or deletion candidate without independent provenance and owner authorization.",
        ],
    }
