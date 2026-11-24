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
import stat
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
    build_logs: list = field(default_factory=list)
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
    content_decisions: list = field(default_factory=list)  # report-only: never sent to run quarantine

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
        def add_run(d: Path, kind: str) -> None:
            e = Entry(d, kind, d.name)
            try:
                if kind == "legacy":
                    # Never read a flat symlink or special file as an adoption.
                    from . import legacy_runs
                    try:
                        mode = os.lstat(d / rm.MANIFEST_NAME).st_mode
                    except FileNotFoundError as exc:
                        raise rm.ManifestError("flat run has no adopted manifest") from exc
                    if not stat.S_ISREG(mode):
                        raise rm.ManifestError("flat run manifest is a symlink or special file")
                m = rm.read_manifest(d)
                if kind == "legacy":
                    try:
                        legacy_runs.verify_adoption(root, d.name, m)
                    except (legacy_runs.LegacyRunError, OSError) as exc:
                        raise rm.ManifestError(f"untrusted flat run adoption: {exc}") from exc
                if m["kind"] != kind or m["run_id"] != d.name:
                    e.problem = "manifest names a different kind or run id than its directory"
                else:
                    e.manifest = m
            except rm.ManifestError as exc:
                e.problem = str(exc)
            if measure:
                e.size_bytes = dir_size(d)
            inv.runs.append(e)

        for kind_dir in sorted(runs.iterdir()):
            if not kind_dir.is_dir() or kind_dir.is_symlink():
                inv.unknown.append(kind_dir)
                continue
            if kind_dir.name not in rm.NESTED_RUN_KINDS:
                # A flat pre-manifest run; unadopted/corrupt ones remain GC blockers.
                add_run(kind_dir, "legacy")
                continue
            for d in sorted(kind_dir.iterdir()):
                if not d.is_dir() or d.is_symlink():
                    inv.unknown.append(d)
                    continue
                add_run(d, kind_dir.name)
    for name, attr in (("builds", "builds"), ("sources", "sources")):
        p = root / name
        if p.is_dir():
            setattr(inv, attr, sum(1 for _ in p.iterdir()))
    logs = root / "build-logs"
    if logs.is_dir() and not logs.is_symlink():
        for path in sorted(logs.iterdir()):
            inv.build_logs.append(Entry(path, "build-log", path.name,
                                        size_bytes=dir_size(path) if measure else 0))
    known = {"build-logs", "runs", "builds", "sources", "cache", "objects", "evidence-hold", "trash", "cleanup-records", ".locks", "run-records"}
    if root.is_dir():
        inv.unknown += [p for p in sorted(root.iterdir()) if p.name not in known and not p.name.startswith(".")]
    return inv


def inspect_cache_roots(root: Path, *, policy: Policy | None = None, now: float | None = None,
                        measure: bool = True, blocked: str | None = None) -> list[dict]:
    """WSL09 build candidate decisions and read-only source inventory."""
    from . import build_gc

    return build_gc.inspect(root, policy=policy or Policy(), now=now,
                            measure=measure, blocked=blocked)

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
    for entry in inv.build_logs:
        decisions.append(Decision(
            entry.path, entry.kind, entry.run_id, "protected", entry.size_bytes,
            ["build diagnostics retained indefinitely; explicit removal required"],
        ))
    # Run-record content lives outside runs/. Never pass these objects to the
    # run-quarantine path. Report orphan blobs only: deletion is not implemented.
    # Corrupt records prevent a safe reachability proof: protect everything and
    # block run cleanup until the store is repaired.
    from .run_records import RecordError, RunRecordStore

    content = []
    store = RunRecordStore(inv.root)
    objects = []
    failure = None
    try:
        objects = store.object_paths()
        rooted = store.protected_paths()
    except (RecordError, OSError) as exc:
        rooted = set()
        failure = f"run records unreadable: {exc}"
        blocked = f"{blocked}; {failure}" if blocked else failure
    for path in objects:
        if failure or path in rooted:
            action, reason = "protected", failure or "record root or referenced provenance blob"
        else:
            action, reason = "unreferenced", "unreferenced blob; inspection only (no content GC)"
        content.append(Decision(path, "record-object", path.name, action,
                                path.lstat().st_size, [reason]))
    if failure:
        for decision in decisions:
            if decision.action == "candidate":
                decision.action = "protected"
                decision.reasons = [f"cleanup blocked: {blocked}"]
    return CleanupPlan(decisions, list(inv.unknown), blocked, content)


def render(plan: CleanupPlan, root: Path, *, verbose: bool = False) -> str:
    def gb(n: int) -> str:
        return f"{n / 1e9:.2f} GB"
    t = plan.totals()
    lines = [f"work root {root}",
             (f"runs: {len(plan.decisions)}  candidates {t['candidate']['runs']} ({gb(t['candidate']['bytes'])})  "
              f"protected {t['protected']['runs']} ({gb(t['protected']['bytes'])})")]
    if plan.blocked:
        lines.append(f"BLOCKED: {plan.blocked}")
    reasons: dict[str, int] = {}
    for d in plan.decisions:
        if d.action == "protected":
            reasons[d.reasons[0].split(":")[0].split("(")[0].strip()] = reasons.get(d.reasons[0].split(":")[0].split("(")[0].strip(), 0) + 1
    for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"  protected, first reason: {r}: {n}")
    if plan.content_decisions:
        roots = sum(d.action == "protected" for d in plan.content_decisions)
        orphan = sum(d.action == "unreferenced" for d in plan.content_decisions)
        lines.append(f"run-record objects: {roots} protected, {orphan} unreferenced (no deletion)")
    if plan.unknown:
        lines.append(f"not part of the layout, reported and never touched: {len(plan.unknown)}")
        for p in plan.unknown[:20]:
            lines.append(f"  {p}")
    if verbose:
        for d in plan.decisions:
            lines.append(f"{d.action:9} {d.kind:8} {d.size_bytes / 1e6:9.1f} MB {d.run_id}  {'; '.join(d.reasons)}")
    lines.append("dry run: nothing was changed (apply arrives with harvest and quarantine)")
    return "\n".join(lines)


def native_installed_build_refs(root: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Read native installed-radiance receipts; malformed entries block all GC."""
    import json
    import re

    base = Path(root) / "installed-radiance"
    if not base.exists() and not base.is_symlink():
        return {}, []
    releases = base / "releases"
    if (base.is_symlink() or not base.is_dir()
            or releases.is_symlink() or not releases.is_dir()):
        return {}, [f"native installed releases root is missing or untrusted: {releases}"]
    refs: dict[str, list[str]] = {}
    errors: list[str] = []
    try:
        for release in sorted(releases.iterdir()):
            try:
                if release.is_symlink() or not release.is_dir():
                    raise ValueError("untrusted release entry")
                receipt = release / "install.json"
                if receipt.is_symlink() or not receipt.is_file():
                    raise ValueError("missing or untrusted install.json")
                doc = json.loads(receipt.read_text(encoding="utf-8"))
                if (not isinstance(doc, dict)
                        or doc.get("schema") != "bc-radiance-installed-v1"
                        or doc.get("tag") != release.name
                        or (doc.get("prefix") is not None
                            and doc["prefix"] != str(release))):
                    raise ValueError("invalid native install receipt identity")
                record = doc.get("record")
                if not isinstance(record, dict):
                    raise TypeError("missing native release record")
                slice_id, plan_id = record.get("source_slice_id"), record.get("plan_id")
                if (not isinstance(slice_id, str) or not re.fullmatch(r"[0-9a-f]{32}", slice_id)
                        or not isinstance(plan_id, str) or not re.fullmatch(r"[0-9a-f]{32}", plan_id)):
                    raise ValueError("invalid native release build identity")
                refs.setdefault(f"builds/{slice_id}/{plan_id}", []).append(
                    f"native installed release {release.name}")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"invalid native installed release {release.name}: {exc}")
    except OSError as exc:
        errors.append(f"cannot enumerate native installed releases: {exc}")
    return refs, errors


def measurement_record_build_refs(root: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Pin immutable MRC01 build IDs; fail closed on unverified record/blob."""
    from .run_records import RecordError, RunRecordStore

    location = Path(root) / "run-records"
    if not location.exists() and not location.is_symlink():
        return {}, []
    v1 = location / "v1"
    if (location.is_symlink() or not location.is_dir()
            or v1.is_symlink() or (v1.exists() and not v1.is_dir())):
        return {}, [f"untrusted run-record store directory: {location}"]
    try:
        store = RunRecordStore(root)
        records = store.query()
        refs: dict[str, list[str]] = {}
        for item in records:
            identity = item["record"]["identity"]
            refs.setdefault(identity["build_id"], []).append(
                "run record " + item["sha256"][:12])
        return refs, []
    except (RecordError, OSError, ValueError) as exc:
        return {}, [f"unreadable run-record store: {exc}"]


def stored_serving_manifest_build_refs(root: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Pin typed runtime paths in saved v1, v2 and v3 serving manifests.

    bc-serve writes manifest-*.json directly under its chosen run directory,
    normally a named site directory one level beneath the work root. Replay
    indexes and evidence sidecars are not serving manifests.
    """
    import json

    root = Path(root)
    refs: dict[str, list[str]] = {}
    errors: list[str] = []
    excluded = {"builds", "sources", "runs", "releases", "installed-radiance",
                "run-records", "objects", "trash", "cache", "artifacts-store",
                "test-tmp", "resource-locks", "cleanup-records", "build-logs"}
    try:
        sites = [root] + [site for site in sorted(root.iterdir())
                          if site.is_dir() and not site.is_symlink()
                          and site.name not in excluded]
        for site in sites:
            for saved in sorted(site.glob("manifest-*.json")):
                if ".json." in saved.name:
                    continue
                try:
                    if saved.is_symlink() or not saved.is_file():
                        raise ValueError("untrusted manifest file")
                    doc = json.loads(saved.read_text(encoding="utf-8"))
                    if (not isinstance(doc, dict)
                            or doc.get("kind") != "ServingManifest"
                            or doc.get("apiVersion") not in ("bigcherry/v1", "bigcherry/v2", "bigcherry/v3")):
                        raise ValueError("invalid stored serving manifest identity")
                    processes = doc.get("spec", {}).get("processes", {})
                    engine = processes.get("engine", {})
                    runtime = engine.get("runtime", {})
                    home = runtime.get("radiance_home")
                    if home is None:
                        continue  # older manifests have no typed runtime path
                    if not isinstance(home, str) or not home.startswith("/"):
                        raise ValueError("invalid stored runtime path")
                    refs.setdefault(home, []).append(
                        f"stored serving manifest {saved.name}")
                except (OSError, ValueError, TypeError, AttributeError) as exc:
                    errors.append(f"unreadable stored manifest {saved}: {exc}")
    except OSError as exc:
        errors.append(f"cannot enumerate stored serving manifests: {exc}")
    return refs, errors
