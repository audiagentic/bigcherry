"""Read-only workspace inventory and the cleanup plan (WSL03 skeleton).

Nothing in this module deletes or moves anything. `inspect_workspace` reads the work root; `plan_cleanup` decides, for every
directory under runs/, whether it is a cleanup CANDIDATE or PROTECTED and says why. The rule that matters: bytes become a
candidate only when ownership, inactivity, reachability, harvest integrity and retention have all been established
independently. Anything unmanifested, unreadable, of an unknown kind or owner, or in a non-terminal state is protected and
reported. A plan id appearing in a manifest is metadata and pins nothing; only typed references (a run naming another run),
an administrative hold, an active run and the newest K runs of a kind are roots.
"""

from __future__ import annotations

import calendar
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import run_manifest as rm


@dataclass(frozen=True)
class Policy:
    min_age_days: float = 7.0       # a run younger than this is never a candidate
    newest_k: int = 3               # the newest K runs of each kind are kept regardless
    harvested_only: bool = True     # only a run whose harvest is verified can be a candidate


@dataclass
class Entry:
    path: Path
    kind: str
    run_id: str
    manifest: dict | None = None
    problem: str | None = None      # why it is unmanaged (no manifest, unreadable, kind or id mismatch)
    size_bytes: int = 0


@dataclass
class Inventory:
    root: Path
    runs: list = field(default_factory=list)
    unknown: list = field(default_factory=list)      # top-level paths that are not part of the layout
    builds: int = 0
    sources: int = 0


@dataclass
class Decision:
    path: Path
    kind: str
    run_id: str
    action: str                     # "candidate" or "protected"
    size_bytes: int
    reasons: list


@dataclass
class CleanupPlan:
    decisions: list
    unknown: list
    blocked: str | None = None      # a maintenance condition that stops any mutation (active measurement, lock)

    def totals(self) -> dict:
        out = {"candidate": [0, 0], "protected": [0, 0]}
        for d in self.decisions:
            out[d.action][0] += 1
            out[d.action][1] += d.size_bytes
        return {k: {"runs": v[0], "bytes": v[1]} for k, v in out.items()}


def dir_size(path: Path) -> int:
    """Bytes under a directory, never following a symlink and never crossing into another filesystem."""
    total = 0
    try:
        dev = os.lstat(path).st_dev
    except OSError:
        return 0
    stack = [path]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    try:
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    # st_dev is 0 in a DirEntry stat on Windows: unknown, not a different filesystem
                    if e.is_symlink() or (st.st_dev and dev and st.st_dev != dev):
                        continue
                    if e.is_dir(follow_symlinks=False):
                        stack.append(Path(e.path))
                    else:
                        total += st.st_size
        except OSError:
            continue
    return total


def inspect_workspace(root: Path, *, measure: bool = True) -> Inventory:
    root = Path(root)
    inv = Inventory(root)
    runs = root / "runs"
    if runs.is_dir() and not runs.is_symlink():
        for kind_dir in sorted(runs.iterdir()):
            if not kind_dir.is_dir() or kind_dir.is_symlink():
                inv.unknown.append(kind_dir)
                continue
            if kind_dir.name not in rm.RUN_KINDS:
                inv.unknown.append(kind_dir)          # a directory the layout does not define: reported, never touched
                continue
            for d in sorted(kind_dir.iterdir()):
                if not d.is_dir() or d.is_symlink():
                    inv.unknown.append(d)
                    continue
                e = Entry(d, kind_dir.name, d.name)
                try:
                    m = rm.read_manifest(d)
                    if m["kind"] != kind_dir.name or m["run_id"] != d.name:
                        e.problem = "manifest names a different kind or run id than its directory"
                    else:
                        e.manifest = m
                except rm.ManifestError as exc:
                    e.problem = str(exc)
                if measure:
                    e.size_bytes = dir_size(d)
                inv.runs.append(e)
    for name, attr in (("builds", "builds"), ("sources", "sources")):
        p = root / name
        if p.is_dir():
            setattr(inv, attr, sum(1 for _ in p.iterdir()))
    known = set(("runs", "builds", "sources", "cache", "objects", "evidence-hold", "trash", ".locks"))
    if root.is_dir():
        inv.unknown += [p for p in sorted(root.iterdir()) if p.name not in known and not p.name.startswith(".")]
    return inv


def inspect_cache_roots(root: Path, *, measure: bool = True) -> list[dict]:
    """WSL04 read-only build/source inventory; NEVER emits a deletion candidate.

    Layout recognition and a metadata sidecar only establish that the entry can
    be *described*. They do not prove cache ownership, provenance, references,
    rebuildability, or absence of a live consumer. A later GC design must
    establish all of those separately before considering a candidate.
    """
    root = Path(root)
    records: list[dict] = []

    def record(path: Path, kind: str, metadata: str, reason: str) -> None:
        records.append({
            "kind": kind, "path": str(path),
            "size_bytes": dir_size(path) if measure and path.is_dir() and not path.is_symlink() else None,
            "metadata": metadata,
            "action": "protected",
            "reasons": [reason, "no cache GC/rebuildability authorization is implemented"],
        })

    for kind in ("build", "source"):
        base = root / (kind + "s")
        if base.is_symlink():
            record(base, kind, "untrusted", "cache root is a symlink")
            continue
        if not base.exists():
            continue
        if not base.is_dir():
            record(base, kind, "untrusted", "cache root is not a directory")
            continue
        try:
            roots = sorted(base.iterdir())
            for entry in roots:
                if entry.is_symlink() or not entry.is_dir():
                    if kind == "source" and entry.is_file() and entry.name.endswith(".metadata.json") and not entry.is_symlink():
                        # Paired source metadata is attached to its directory;
                        # orphan sidecars are still reported as protected.
                        if (base / entry.name[:-len(".metadata.json")]).is_dir():
                            continue
                    record(entry, kind, "untrusted", "unclassified cache entry or symlink")
                    continue
                if kind == "source":
                    sidecar = base / (entry.name + ".metadata.json")
                    if sidecar.is_file() and not sidecar.is_symlink():
                        try:
                            import json
                            valid = isinstance(json.loads(sidecar.read_text(encoding="utf-8")), dict)
                        except (OSError, ValueError):
                            valid = False
                        metadata = "present-unverified" if valid else "invalid"
                    else:
                        metadata = "missing"
                    record(entry, kind, metadata, "source provenance and consumers not independently verified")
                    continue
                # Builds are grouped under a source/recipe key. Unknown children
                # (including reference-ladder caches) remain visible/protected.
                for child in sorted(entry.iterdir()):
                    if child.is_symlink() or not child.is_dir():
                        record(child, kind, "untrusted", "unclassified build entry or symlink")
                        continue
                    metadata_files = list(child.glob("bigcherry-build-metadata-*.json"))
                    valid = len(metadata_files) == 1 and metadata_files[0].is_file() and not metadata_files[0].is_symlink()
                    if valid:
                        try:
                            import json
                            valid = isinstance(json.loads(metadata_files[0].read_text(encoding="utf-8")), dict)
                        except (OSError, ValueError):
                            valid = False
                    record(child, kind, "present-unverified" if valid else "missing-or-invalid",
                           "build recipe, source pin, live claims and rebuildability not verified")
        except OSError as exc:
            record(base, kind, "untrusted", f"cache listing incomplete: {exc}")
    return records


def _epoch(stamp: str) -> float:
    return calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))


def plan_cleanup(inv: Inventory, *, now: float | None = None, policy: Policy | None = None,
                 blocked: str | None = None) -> CleanupPlan:
    policy = policy or Policy()
    now = time.time() if now is None else now
    managed = [e for e in inv.runs if e.manifest is not None]
    by_id = {e.run_id: e for e in managed}

    roots: dict[str, str] = {}
    for e in managed:
        m = e.manifest
        if m["state"] not in rm.TERMINAL_STATES:
            roots[e.run_id] = f"state is {m['state']}"
        if m["retention"].get("hold"):
            roots[e.run_id] = "administrative hold"
    for kind in rm.RUN_KINDS:
        same = sorted((e for e in managed if e.kind == kind), key=lambda e: e.manifest["created_at"], reverse=True)
        for e in same[:policy.newest_k]:
            roots.setdefault(e.run_id, f"one of the newest {policy.newest_k} {kind} runs")

    # reachability: a root run protects every run it names by a typed reference, and so on transitively
    protected: dict[str, str] = dict(roots)
    frontier = list(roots)
    while frontier:
        cur = frontier.pop()
        e = by_id.get(cur)
        if not e:
            continue
        for ref in e.manifest["references"]:
            if ref["type"] == "run" and ref["id"] in by_id and ref["id"] not in protected:
                protected[ref["id"]] = f"referenced by {cur}"
                frontier.append(ref["id"])

    # any run that still names another run keeps it, whether or not the naming run is itself a root
    for e in managed:
        if e.manifest["retention"]["state"] == "purged":
            continue
        for ref in e.manifest["references"]:
            if ref["type"] == "run" and ref["id"] in by_id and ref["id"] != e.run_id and ref["id"] not in protected:
                protected[ref["id"]] = f"referenced by {e.run_id}"

    decisions = []
    for e in inv.runs:
        reasons = []
        if e.manifest is None:
            reasons.append(f"unmanaged: {e.problem}")
        else:
            m = e.manifest
            if e.run_id in protected:
                reasons.append(protected[e.run_id])
            if m["provenance"] != "recorded":
                reasons.append(f"provenance is {m['provenance']}")
            if policy.harvested_only and m["harvest"]["state"] != "verified":
                reasons.append(f"harvest is {m['harvest']['state']}")
            age_days = (now - _epoch(m["created_at"])) / 86400.0
            if age_days < policy.min_age_days:
                reasons.append(f"younger than {policy.min_age_days:g} days ({age_days:.1f})")
            until = m["retention"].get("retain_until")
            if until and _epoch(until) > now:
                reasons.append(f"retained until {until}")
            if m["retention"]["state"] in ("quarantined", "purged"):
                reasons.append(f"retention state is {m['retention']['state']}")
        if blocked:
            reasons.append(f"cleanup blocked: {blocked}")
        decisions.append(Decision(e.path, e.kind, e.run_id, "protected" if reasons else "candidate", e.size_bytes,
                                  reasons or ["harvest verified, retention expired, unreferenced"]))
    return CleanupPlan(decisions, list(inv.unknown), blocked)


def render(plan: CleanupPlan, root: Path, *, verbose: bool = False) -> str:
    def gb(n: int) -> str:
        return f"{n / 1e9:.2f} GB"
    t = plan.totals()
    lines = [f"work root {root}",
             f"runs: {len(plan.decisions)}  candidates {t['candidate']['runs']} ({gb(t['candidate']['bytes'])})  "
             f"protected {t['protected']['runs']} ({gb(t['protected']['bytes'])})"]
    if plan.blocked:
        lines.append(f"BLOCKED: {plan.blocked}")
    reasons: dict[str, int] = {}
    for d in plan.decisions:
        if d.action == "protected":
            reasons[d.reasons[0].split(":")[0].split("(")[0].strip()] = reasons.get(d.reasons[0].split(":")[0].split("(")[0].strip(), 0) + 1
    for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"  protected, first reason: {r}: {n}")
    if plan.unknown:
        lines.append(f"not part of the layout, reported and never touched: {len(plan.unknown)}")
        for p in plan.unknown[:20]:
            lines.append(f"  {p}")
    if verbose:
        for d in plan.decisions:
            lines.append(f"{d.action:9} {d.kind:8} {d.size_bytes / 1e6:9.1f} MB {d.run_id}  {'; '.join(d.reasons)}")
    lines.append("dry run: nothing was changed (apply arrives with harvest and quarantine)")
    return "\n".join(lines)
