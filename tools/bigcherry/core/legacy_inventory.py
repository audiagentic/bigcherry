"""WSL06 read-only inventory of legacy work-root objects.

Nothing here deletes, moves or writes anything under the inspected roots (git is run with --no-optional-locks so not
even an index refresh is written). No directory, symlink, manifest or reference establishes ownership or rebuildability:
every entry is protected, and its suggested disposition is advice for the owner's review, never an authorization.

The report reconciles bytes: each root is also measured as a whole, and bytes under it that no entry accounts for (loose
files between the levels the inventory enumerates) are reported as unattributed rather than silently dropped.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

from . import run_manifest as rm

SCHEMA_VERSION = 2
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{2,}")
_TEXT_LIMIT = 4 << 20   # tracked files larger than this are not searched for references


def _measure(path: Path, device: int, seen: set) -> tuple[int, bool, float]:
    """Payload bytes, completeness and newest mtime; no symlinks, no other filesystems, each inode counted once."""
    total, complete, newest = 0, True, 0.0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            st = current.lstat()
        except OSError:
            complete = False
            continue
        if st.st_dev != device or current.is_symlink():
            continue
        identity = (st.st_dev, st.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        newest = max(newest, st.st_mtime)
        if not current.is_dir():
            total += st.st_size
            continue
        try:
            with os.scandir(current) as scan:
                stack.extend(Path(entry.path) for entry in scan)
        except OSError:
            complete = False
    return total, complete, newest


def reference_index(repo_root: Path) -> dict[str, list[str]]:
    """Token -> tracked files that mention it (first three). Empty when the repository cannot be listed."""
    try:
        listing = subprocess.run(["git", "-C", str(repo_root), "ls-files", "-z"], capture_output=True, check=True,
                                 timeout=120).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return {}
    index: dict[str, list[str]] = {}
    for rel in filter(None, listing.split("\0")):
        path = Path(repo_root) / rel
        try:
            if path.stat().st_size > _TEXT_LIMIT:
                continue
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\0" in raw:
            continue
        for token in set(_TOKEN.findall(raw.decode("utf-8", "replace"))):
            files = index.setdefault(token, [])
            if len(files) < 3:
                files.append(rel)
    return index


def _active_cwds() -> list[str] | None:
    """Working directories of live processes (Linux /proc); None where that cannot be read."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    cwds = []
    for pid in proc.iterdir():
        if pid.name.isdigit():
            try:
                cwds.append(os.readlink(pid / "cwd"))
            except OSError:
                continue
    return cwds


def _git_state(path: Path) -> dict | None:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "--no-optional-locks", "-C", str(path), *args], capture_output=True,
                                  text=True, check=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            return None

    head = git("rev-parse", "--abbrev-ref", "HEAD")
    if head is None:
        return None
    status = git("status", "--porcelain")
    unpushed = git("rev-list", "--count", "HEAD", "--not", "--remotes")
    return {"head": head.strip(), "dirty_paths": None if status is None else len(status.splitlines()),
            "unpushed_commits": None if unpushed is None else int(unpushed.strip() or 0)}


def _rebuildability(category: str, evidence: str, git: dict | None) -> str:
    if category == "build" and evidence == "build-metadata":
        return "rebuildable-by-plan-unverified"
    if category == "worktree" and git and git["dirty_paths"] == 0 and git["unpushed_commits"] == 0:
        return "recreatable-from-remote"
    if category == "run":
        return "not-rebuildable-measurement"
    return "unknown"


def _disposition(entry: dict) -> str:
    if entry["active_process"]:
        return "in-use"
    if entry["references"]:
        return "keep-referenced"
    if entry["category"] == "run" and entry["evidence"] == "valid-manifest":
        return "review-adopt-or-harvest"
    if entry["rebuildability"] in ("recreatable-from-remote", "rebuildable-by-plan-unverified"):
        return "review-recreatable"
    return "owner-decision"


def inventory_legacy(root: Path, *, measure: bool = False, extra_roots: tuple[Path, ...] = (),
                     repo_root: Path | None = None) -> dict:
    """Classify the work root (and any extra legacy roots); protect everything; reconcile measured bytes."""
    roots = [Path(root), *map(Path, extra_roots)]
    for r in roots:
        if r.is_symlink() or not r.is_dir():
            raise ValueError(f"{r}: legacy roots must be existing non-symlink directories")
    refs = reference_index(repo_root) if repo_root else {}
    cwds = _active_cwds()
    entries: list[dict] = []
    identities: dict[tuple[int, int], str] = {}
    seen: set = set()

    def append(path: Path, category: str, evidence: str, owner_root: Path) -> None:
        try:
            st = path.lstat()
        except OSError as exc:
            entries.append({"root": str(owner_root), "path": str(path), "category": category, "evidence": "unreadable",
                            "size_bytes": None, "size_complete": False, "alias_of": None, "action": "protected",
                            "reasons": [f"lstat failed: {exc}"], "references": [], "active_process": None,
                            "newest_mtime": None, "git": None, "rebuildability": "unknown",
                            "disposition": "owner-decision"})
            return
        symlink = path.is_symlink()
        identity = (st.st_dev, st.st_ino)
        alias = identities.get(identity)
        if alias is None and not symlink:
            identities[identity] = str(path)
        size, complete, newest = None, False, None
        if measure and not symlink and alias is None:
            size, complete, mtime = _measure(path, st.st_dev, seen)
            newest = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mtime)) if mtime else None
        resolved = str(path.resolve()) if not symlink else str(path)
        active = None if cwds is None else any(c == resolved or c.startswith(resolved + os.sep) for c in cwds)
        git = _git_state(path) if category == "worktree" and not symlink else None
        reasons = ["ownership, typed references, active consumers and rebuildability unverified"]
        if symlink:
            reasons.append("symlink: never traverse or authorize cleanup")
        if alias:
            reasons.append("duplicate device/inode: do not count or reclaim twice")
        if measure and not symlink and alias is None and not complete:
            reasons.append("size inspection incomplete: measured bytes are a lower bound")
        entry = {"root": str(owner_root), "path": str(path), "category": category, "evidence": evidence,
                 "device": st.st_dev, "inode": st.st_ino, "is_symlink": symlink,
                 "size_bytes": size, "size_complete": complete, "alias_of": alias, "action": "protected",
                 "reasons": reasons, "references": refs.get(path.name, []), "active_process": active,
                 "newest_mtime": newest, "git": git}
        entry["rebuildability"] = _rebuildability(category, evidence, git)
        entry["disposition"] = _disposition(entry)
        entries.append(entry)

    def children(path: Path) -> list[Path]:
        if path.is_symlink() or not path.is_dir():
            return []
        try:
            return sorted(path.iterdir())
        except OSError as exc:
            raise ValueError(f"unable to enumerate {path}: {exc}") from exc

    typed = {"runs", "builds", "sources", "worktrees"}
    for base in roots:
        for item in children(base):
            if item.name not in typed or item.is_symlink() or not item.is_dir():
                append(item, "unclassified-root", "unverified", base)
                continue
            if item.name == "runs":
                for kind in children(item):
                    if kind.is_symlink() or not kind.is_dir() or kind.name not in rm.RUN_KINDS:
                        append(kind, "unclassified-run-kind", "unverified", base)
                        continue
                    for run in children(kind):
                        evidence = "unverified"
                        if not run.is_symlink() and run.is_dir():
                            try:
                                m = rm.read_manifest(run)
                                if m["kind"] == kind.name and m["run_id"] == run.name:
                                    evidence = "valid-manifest"
                            except (rm.ManifestError, OSError, KeyError):
                                pass
                        append(run, "run", evidence, base)
            elif item.name == "builds":
                for group in children(item):
                    if group.is_symlink() or not group.is_dir():
                        append(group, "unclassified-build", "unverified", base)
                        continue
                    for build in children(group):
                        meta = list(build.glob("bigcherry-build-metadata-*.json")) if build.is_dir() else []
                        append(build, "build", "build-metadata" if len(meta) == 1 else "metadata-missing", base)
            elif item.name == "sources":
                for source in children(item):
                    append(source, "source", "metadata-unverified", base)
            else:
                for checkout in children(item):
                    append(checkout, "worktree", "git-pointer" if (checkout / ".git").exists() else "unverified", base)

    reconciliation = []
    if measure:
        for base in roots:
            whole, complete, _ = _measure(base, base.lstat().st_dev, set())
            attributed = sum(e["size_bytes"] or 0 for e in entries if e["root"] == str(base))
            reconciliation.append({"root": str(base), "measured_bytes": whole, "complete": complete,
                                   "attributed_bytes": attributed, "unattributed_bytes": whole - attributed})
    by_category: dict[str, dict] = {}
    by_disposition: dict[str, int] = {}
    for e in entries:
        c = by_category.setdefault(e["category"], {"entries": 0, "bytes": 0})
        c["entries"] += 1
        c["bytes"] += e["size_bytes"] or 0
        by_disposition[e["disposition"]] = by_disposition.get(e["disposition"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION, "root": str(root), "roots": [str(r) for r in roots], "read_only": True,
        "entries": entries,
        "totals": {
            "entries": len(entries),
            "protected": len(entries),
            "known_measured_bytes": sum(e["size_bytes"] or 0 for e in entries),
            "unmeasured_entries": sum(not e["size_complete"] for e in entries),
            "incomplete": any(not e["size_complete"] for e in entries),
            "by_category": by_category,
            "by_disposition": by_disposition,
            "unknown_objects": sum(e["category"].startswith("unclassified") for e in entries),
        },
        "reconciliation": reconciliation,
        "notes": [
            "Size is logical file payload, not allocated disk blocks; hard links are counted once.",
            "Inventory covers top-level objects, run leaves, build leaves, source entries and worktree entries of every root.",
            "References are tracked repository files naming the entry; absence of a reference is not deletion safety.",
            "Every entry is protected; a disposition is advice for owner review, not an adoption or deletion candidate.",
        ],
    }


def render(report: dict) -> str:
    """Human report: one line an entry, then totals by category and disposition and the byte reconciliation."""
    def size(value: int | None) -> str:
        return "unmeasured" if value is None else f"{value / 1e9:.2f} GB"

    lines = [f"{e['category']:22} {size(e['size_bytes']):>12} {e['disposition']:26} {e['path']}"
             for e in report["entries"]]
    t = report["totals"]
    lines.append("")
    for name, c in sorted(t["by_category"].items()):
        lines.append(f"  {name:22} {c['entries']:6} entries {size(c['bytes']):>12}")
    for name, count in sorted(t["by_disposition"].items()):
        lines.append(f"  {name:26} {count:6}")
    for r in report["reconciliation"]:
        lines.append(f"  root {r['root']}: measured {size(r['measured_bytes'])}, attributed {size(r['attributed_bytes'])},"
                     f" unattributed {size(r['unattributed_bytes'])}{'' if r['complete'] else ' (lower bound)'}")
    lines.append(f"read-only: {t['entries']} protected entries, {t['unknown_objects']} unknown objects; "
                 f"{t['unmeasured_entries']} unmeasured or incomplete")
    return "\n".join(lines)
