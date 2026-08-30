"""Host-local tree-activity liveness protocol (HI151).

A maintenance operation (the pin-bump orchestrator, HI153) must refuse to
touch a tree that some other long-running process is actively using --
``git status`` alone does not catch this: a real b10502->b10680 bump this
project ran found a configured campaign tree with a clean-enough-looking
git state but two active experiment log tails from a concurrent session.

Two pieces, both under ``ProjectContext.work_root`` (host-local, never
committed):

- a **lease** (``leases/<uuid>.json``) that a long-running runner (build,
  campaign-build, tune-campaign, profile-campaign, an experiment/validation
  run) holds for its own duration -- new maintenance work refuses to start
  while any lease is live;
- a **maintenance lock** (``maintenance.lock/``) that a maintenance
  operation holds for ITS duration -- new long-running work should refuse
  to start while it is held (that refusal is the caller's job: this module
  only exposes ``list_active_leases`` for it to check).

Staleness is decided ONLY by real PID liveness on the lease's own host
(never on a different host -- this process cannot know if a remote PID is
alive, so a lease recorded from another hostname is always treated as
live). A lease is never silently broken just because it looks old.
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


class TreeActivityError(RuntimeError):
    pass


def _tree_activity_root(work_root: Path, project_root: Path) -> Path:
    import hashlib

    key = hashlib.sha256(str(project_root.resolve()).encode("utf-8")).hexdigest()[:16]
    return work_root / "tree-activity" / key


@dataclass(frozen=True)
class LeaseInfo:
    lease_id: str
    pid: int
    hostname: str
    command: str
    run_id: str
    project_root: str
    started_at: float
    path: Path

    def is_live(self) -> bool:
        if self.hostname != socket.gethostname():
            # Cannot check a remote PID's liveness -- fail closed (treat as
            # live) rather than guess. An operator can break_stale() this
            # explicitly once they've confirmed the remote host is idle.
            return True
        return _pid_alive(self.pid)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, just owned by someone else -- still alive.
        return True
    except OSError:
        return False
    return True


def list_active_leases(work_root: Path, project_root: Path) -> list[LeaseInfo]:
    """Every lease recorded for ``project_root``, live or not (caller filters)."""
    leases_dir = _tree_activity_root(work_root, project_root) / "leases"
    if not leases_dir.is_dir():
        return []
    out: list[LeaseInfo] = []
    for path in sorted(leases_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            out.append(LeaseInfo(
                lease_id=path.stem, pid=int(data["pid"]), hostname=str(data["hostname"]),
                command=str(data["command"]), run_id=str(data["run_id"]),
                project_root=str(data["project_root"]), started_at=float(data["started_at"]),
                path=path,
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def list_live_leases(work_root: Path, project_root: Path) -> list[LeaseInfo]:
    return [lease for lease in list_active_leases(work_root, project_root) if lease.is_live()]


def prune_stale_leases(work_root: Path, project_root: Path) -> list[str]:
    """Remove leases whose owning PID is confirmed dead on THIS host.

    Never touches a lease recorded from a different hostname -- those are
    always treated as live (see LeaseInfo.is_live)."""
    removed = []
    for lease in list_active_leases(work_root, project_root):
        if lease.hostname == socket.gethostname() and not _pid_alive(lease.pid):
            lease.path.unlink(missing_ok=True)
            removed.append(lease.lease_id)
    return removed


class Lease:
    """Held by a long-running runner for its own duration.

    Use as a context manager so the lease is removed even on an exception;
    a crash that kills the process outright leaves the file behind, which
    is exactly what PID-liveness pruning is for.
    """

    def __init__(self, work_root: Path, project_root: Path, *, command: str, run_id: str):
        self.work_root = work_root
        self.project_root = project_root
        self.command = command
        self.run_id = run_id
        self._lease_id = uuid.uuid4().hex
        self._path: Path | None = None

    def __enter__(self) -> "Lease":
        root = _tree_activity_root(self.work_root, self.project_root) / "leases"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self._lease_id}.json"
        payload = {
            "pid": os.getpid(), "hostname": socket.gethostname(), "command": self.command,
            "run_id": self.run_id, "project_root": str(self.project_root.resolve()),
            "started_at": time.time(),
        }
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        self._path = path
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._path is not None:
            self._path.unlink(missing_ok=True)
            self._path = None


class MaintenanceLock:
    """Held by a maintenance operation (e.g. pin-bump) for its duration.

    Directory-based mkdir() claim, same atomicity pattern as
    core.resources.ResourceLock. Refuses to acquire while any lease for
    this project_root is live.
    """

    def __init__(self, work_root: Path, project_root: Path):
        self.work_root = work_root
        self.project_root = project_root
        self.path = _tree_activity_root(work_root, project_root) / "maintenance.lock"
        self._acquired = False

    def acquire(self) -> None:
        live = list_live_leases(self.work_root, self.project_root)
        if live:
            names = ", ".join(f"{lease.command}({lease.run_id})" for lease in live)
            raise TreeActivityError(
                f"refusing to acquire maintenance lock: {len(live)} live lease(s) "
                f"still active for {self.project_root}: {names}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError as exc:
            raise TreeActivityError(
                f"maintenance lock already held for {self.project_root}: {self.path}"
            ) from exc
        owner = {
            "pid": os.getpid(), "hostname": socket.gethostname(), "started_at": time.time(),
        }
        (self.path / "owner.json").write_text(
            json.dumps(owner, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._acquired = True

    def release(self) -> None:
        if not self._acquired:
            return
        (self.path / "owner.json").unlink(missing_ok=True)
        try:
            self.path.rmdir()
        except FileNotFoundError:
            pass
        self._acquired = False

    def __enter__(self) -> "MaintenanceLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


def scan_proc_for_tree_usage(project_root: Path) -> list[str]:
    """Linux-only, DIAGNOSTIC ONLY -- never authoritative, never a gate.

    Scans /proc/*/cwd and /proc/*/cmdline for other processes whose
    working directory or command line references project_root, excluding
    this process's own ancestry. A transition-period aid for runners that
    are not yet Lease-aware; callers must surface this as a warning, not a
    hard stop, per HI150/HI151's design (git-status-only liveness checks
    already proved insufficient once, but an unverifiable heuristic must
    not become a NEW silent source of false confidence either way)."""
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    target = str(project_root.resolve())
    own_pid = os.getpid()
    own_ppid = os.getppid()
    hits: list[str] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in (own_pid, own_ppid):
            continue
        try:
            cwd = os.readlink(entry / "cwd")
        except OSError:
            cwd = ""
        cmdline = ""
        try:
            cmdline = (entry / "cmdline").read_bytes().decode("utf-8", "replace").replace("\0", " ")
        except OSError:
            pass
        if target in cwd or target in cmdline:
            hits.append(f"pid={pid} cwd={cwd!r} cmdline={cmdline.strip()!r}")
    return hits
