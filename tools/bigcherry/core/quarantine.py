"""Quarantine, restore and purge of a harvested run's bulk (WSL03).

Only the files a verified evidence bundle lists as bulk ever move: the bundle's inventory is the contract, never a glob. They go to
trash/<run-id>/files/<same relative path> by same-filesystem rename, driven by a durable journal (trash/<run-id>/journal.json) written
before the first move, so an interrupted operation is resumed from what is on disk rather than guessed. The run directory, its
manifest and its evidence bundle are never moved or deleted. Restore moves the same files back. Purge deletes only the journalled
files, after a grace period, with a fresh check that no other run still references this one. Every mutation needs the caller to say
the workspace is idle (`blocked` is None) and runs under the workspace GC lock.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath

from . import harvest, retention, run_manifest as rm

JOURNAL = "journal.json"


class QuarantineError(RuntimeError):
    pass


def trash_dir(root: Path, run_id: str) -> Path:
    return Path(root) / "trash" / run_id


@contextmanager
def gc_lock(root: Path):
    """One destructive operation at a time per work root. A leftover lock is reported, never silently broken."""
    lock = Path(root) / ".locks" / "gc.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise QuarantineError(f"another cleanup operation holds {lock} (remove it only if no bigcherry process is running)") from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _require_idle(blocked: str | None) -> None:
    if blocked:
        raise QuarantineError(f"refused: {blocked}")


def _rel_ok(rel: str) -> Path:
    p = Path(rel)
    if (p.is_absolute() or PurePosixPath(rel).is_absolute() or PureWindowsPath(rel).anchor or ".." in p.parts
            or not p.parts):
        raise QuarantineError(f"inventory path {rel!r} is not a relative path inside the run")
    return p


def _no_symlink_chain(base: Path, rel: Path) -> bool:
    cur = base
    for part in rel.parts:
        cur = cur / part
        if cur.is_symlink():
            return False
    return True


def inbound_references(root: Path, run_id: str) -> list:
    """Every other managed run that still names this run in a typed reference (terminal or not)."""
    inv = retention.inspect_workspace(root, measure=False)
    return sorted(e.run_id for e in inv.runs
                  if e.manifest is not None and e.run_id != run_id and e.manifest["retention"]["state"] != "purged"
                  and any(r.get("type") == "run" and r.get("id") == run_id for r in e.manifest["references"]))


def _write_journal(tdir: Path, doc: dict) -> None:
    rm.write_atomic(tdir / JOURNAL, json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _read_journal(tdir: Path) -> dict | None:
    try:
        return json.loads((tdir / JOURNAL).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise QuarantineError(f"journal {tdir / JOURNAL} is unreadable: {exc}") from exc


def _verified_inventory(root: Path, run_dir: Path, m: dict) -> list:
    if m["harvest"]["state"] != "verified":
        raise QuarantineError(f"harvest is {m['harvest']['state']}; only a verified bundle makes the bulk safe to move")
    bundle = harvest.bundle_dir(root, m["run_id"])
    problems = harvest.verify_bundle(bundle)
    if problems:
        raise QuarantineError("the evidence bundle does not verify: " + "; ".join(problems[:3]))
    doc = json.loads((bundle / harvest.BUNDLE_NAME).read_text(encoding="utf-8"))
    if doc["run_id"] != m["run_id"] or doc["bundle_id"] != m["harvest"]["bundle_id"]:
        raise QuarantineError("the evidence bundle is not the one the run manifest recorded")
    return [{"path": _rel_ok(b["path"]).as_posix(), "bytes": b["bytes"]} for b in doc["bulk"]]


def _check_source(run_dir: Path, entry: dict) -> None:
    rel = _rel_ok(entry["path"])
    src = run_dir / rel
    if not _no_symlink_chain(run_dir, rel):
        raise QuarantineError(f"{rel} is reached through a symlink")
    st = os.lstat(src)
    if not os.path.isfile(src) or st.st_nlink > 1:
        raise QuarantineError(f"{rel} is not a plain single-link file")
    if st.st_size != entry["bytes"]:
        raise QuarantineError(f"{rel} changed size since it was harvested ({st.st_size} != {entry['bytes']})")


def _same_fs(a: Path, b: Path) -> bool:
    da, db = os.lstat(a).st_dev, os.lstat(b).st_dev
    return not (da and db) or da == db


def quarantine(root: Path, run_dir: Path, *, policy: retention.Policy | None = None, blocked: str | None,
               apply: bool, now: float | None = None) -> dict:
    """Plan (apply=False) or perform the move of a candidate run's bulk into trash. Resumes an interrupted move."""
    root, run_dir = Path(root), Path(run_dir)
    _require_idle(blocked)
    with gc_lock(root):
        m = rm.read_manifest(run_dir)
        run_id = m["run_id"]
        tdir = trash_dir(root, run_id)
        journal = _read_journal(tdir)
        if m["retention"]["state"] == "quarantined":
            if journal and all(e["state"] == "moved" for e in journal["entries"]):
                return {"run_id": run_id, "state": "quarantined", "entries": len(journal["entries"]), "applied": False}
        elif m["retention"]["state"] != "active":
            raise QuarantineError(f"retention state is {m['retention']['state']}; only an active run can be quarantined")
        if journal is None:
            inv = retention.inspect_workspace(root, measure=False)
            plan = retention.plan_cleanup(inv, now=now, policy=policy, blocked=None)
            dec = next((d for d in plan.decisions if d.run_id == run_id), None)
            if dec is None or dec.action != "candidate":
                raise QuarantineError("not a cleanup candidate: " + "; ".join(dec.reasons if dec else ["unknown run"]))
            refs = inbound_references(root, run_id)
            if refs:
                raise QuarantineError("still referenced by " + ", ".join(refs))
            entries = _verified_inventory(root, run_dir, m)
            for e in entries:
                _check_source(run_dir, e)
            entries = [dict(e, state="pending") for e in entries]
        else:
            entries = journal["entries"]
        total = sum(e["bytes"] for e in entries)
        if not apply:
            return {"run_id": run_id, "state": "plan", "entries": len(entries), "bytes": total, "applied": False}
        tdir.mkdir(parents=True, exist_ok=True)
        if not _same_fs(run_dir, tdir):
            raise QuarantineError("the trash directory is on a different filesystem; a rename cannot be atomic")
        doc = journal or {"run_id": run_id, "operation": "quarantine", "bundle_id": m["harvest"]["bundle_id"],
                          "started_at": rm.utc_stamp(now), "entries": entries}
        _write_journal(tdir, doc)
        for e in doc["entries"]:
            _move(run_dir, tdir / "files", e, forward=True)
            _write_journal(tdir, doc)
        rm.update_manifest(run_dir, expected_generation=m["generation"],
                           patch={"retention": {"state": "quarantined", "quarantined_at": rm.utc_stamp(now)}}, now=now)
        return {"run_id": run_id, "state": "quarantined", "entries": len(entries), "bytes": total, "applied": True}


def _move(run_dir: Path, files_dir: Path, entry: dict, *, forward: bool) -> None:
    """Move one journalled file. Idempotent: the entry's state and what is on disk decide what is left to do."""
    rel = _rel_ok(entry["path"])
    src_live, src_trash = run_dir / rel, files_dir / rel
    here, there = (src_live, src_trash) if forward else (src_trash, src_live)
    want = "moved" if forward else "pending"
    if not os.path.lexists(here):
        if os.path.lexists(there):
            entry["state"] = want            # already moved by an interrupted earlier attempt
            return
        raise QuarantineError(f"{rel} is in neither place; the journal and the disk disagree")
    if os.path.lexists(there):
        raise QuarantineError(f"{rel} exists at both the source and the destination; refusing to overwrite")
    if forward:
        _check_source(run_dir, entry)
    elif not _no_symlink_chain(run_dir, Path(*rel.parts[:-1])):
        raise QuarantineError(f"{rel} would be restored through a symlink")
    there.parent.mkdir(parents=True, exist_ok=True)
    os.rename(here, there)
    entry["state"] = want


def restore(root: Path, run_dir: Path, *, blocked: str | None, now: float | None = None) -> dict:
    root, run_dir = Path(root), Path(run_dir)
    _require_idle(blocked)
    with gc_lock(root):
        m = rm.read_manifest(run_dir)
        if m["retention"]["state"] != "quarantined":
            raise QuarantineError(f"retention state is {m['retention']['state']}; only a quarantined run can be restored")
        tdir = trash_dir(root, m["run_id"])
        doc = _read_journal(tdir)
        if doc is None:
            raise QuarantineError("no journal for this quarantine")
        for e in doc["entries"]:
            _move(run_dir, tdir / "files", e, forward=False)
            _write_journal(tdir, doc)
        doc["restored_at"] = rm.utc_stamp(now)
        _write_journal(tdir, doc)
        (tdir / JOURNAL).rename(tdir / f"journal.restored-{time.time_ns()}.json")   # a later quarantine plans afresh
        rm.update_manifest(run_dir, expected_generation=m["generation"],
                           patch={"retention": {"state": "active", "quarantined_at": None}}, now=now)
        return {"run_id": m["run_id"], "state": "active", "entries": len(doc["entries"])}


def purge(root: Path, run_dir: Path, *, grace_days: float, blocked: str | None, now: float | None = None) -> dict:
    """Delete exactly the journalled trash files of a run quarantined for at least `grace_days` and still unreferenced."""
    root, run_dir = Path(root), Path(run_dir)
    _require_idle(blocked)
    now = time.time() if now is None else now
    with gc_lock(root):
        m = rm.read_manifest(run_dir)
        if m["retention"]["state"] != "quarantined":
            raise QuarantineError(f"retention state is {m['retention']['state']}; only a quarantined run can be purged")
        since = m["retention"].get("quarantined_at")
        if not since or (now - retention._epoch(since)) < grace_days * 86400.0:
            raise QuarantineError(f"quarantined {since}; the {grace_days:g}-day grace period has not passed")
        refs = inbound_references(root, m["run_id"])
        if refs:
            raise QuarantineError("referenced again since quarantine by " + ", ".join(refs))
        if m["retention"].get("hold"):
            raise QuarantineError("administrative hold")
        tdir = trash_dir(root, m["run_id"])
        doc = _read_journal(tdir)
        if doc is None or any(e["state"] != "moved" for e in doc["entries"]):
            raise QuarantineError("the journal does not show a complete quarantine")
        freed = 0
        for e in doc["entries"]:
            rel = _rel_ok(e["path"])
            p = tdir / "files" / rel
            if p.is_symlink() or (p.exists() and not p.is_file()):
                raise QuarantineError(f"{rel} in trash is not a plain file")
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
            e["state"] = "purged"
            _write_journal(tdir, doc)
        _prune_empty(tdir / "files")
        doc["purged_at"] = rm.utc_stamp(now)
        _write_journal(tdir, doc)
        rm.update_manifest(run_dir, expected_generation=m["generation"],
                           patch={"retention": {"state": "purged", "purged_at": rm.utc_stamp(now)}}, now=now)
        return {"run_id": m["run_id"], "state": "purged", "entries": len(doc["entries"]), "bytes": freed}


def _prune_empty(top: Path) -> None:
    """Remove empty directories under the trash files tree (never a file)."""
    if not top.is_dir():
        return
    for cur, _dirs, _files in os.walk(top, topdown=False):
        try:
            os.rmdir(cur)
        except OSError:
            pass
