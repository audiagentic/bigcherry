"""Host-local tree-activity liveness protocol (HI151).

A maintenance operation (the pin-bump orchestrator, HI153) must refuse to
touch a tree that some other long-running process is actively using --
``git status`` alone does not catch this.

Both admission directions are enforced here:

- runners publish a lease and refuse/remove it if maintenance is active;
- maintenance publishes its lock before scanning leases and removes the lock
  if any live lease exists.

That two-phase handshake closes the TOCTOU where a runner and maintenance
operation could otherwise both observe an empty state and enter concurrently.
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


def maintenance_lock_path(work_root: Path, project_root: Path) -> Path:
    """Canonical maintenance marker for runner admission/status."""
    return _tree_activity_root(work_root, project_root) / "maintenance.lock"


def maintenance_is_held(work_root: Path, project_root: Path) -> bool:
    return maintenance_lock_path(work_root, project_root).is_dir()


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
    """Remove leases whose owning PID is confirmed dead on THIS host."""
    removed = []
    for lease in list_active_leases(work_root, project_root):
        if lease.hostname == socket.gethostname() and not _pid_alive(lease.pid):
            lease.path.unlink(missing_ok=True)
            removed.append(lease.lease_id)
    return removed


def cleanup_activity_blocker(work_root: Path) -> str | None:
    """Conservative cross-checkout lease admission for workspace cleanup (WSL03).

    Tree-activity leases are keyed by *project* root; cleanup mutates the shared
    physical work root. Until every producer uses a shared-root generation lease,
    do not mistake an absent lease in our checkout for an idle host. Any lease,
    including an apparently stale PID, needs explicit review: PID liveness alone
    cannot prove ownership (PID reuse). Malformed/foreign records fail closed.
    This is a read-only *additional* guard, not a substitute for exclusive
    maintenance admission across all producers.
    """
    activity = Path(work_root).resolve() / "tree-activity"
    try:
        if activity.is_symlink():
            return "tree-activity is a symlink; cleanup admission is unsafe"
        if not activity.exists():
            return None
        if not activity.is_dir():
            return "tree-activity is not a directory"
        for namespace in sorted(activity.iterdir()):
            if namespace.is_symlink() or not namespace.is_dir():
                return f"unrecognised tree-activity namespace: {namespace.name}"
            maintenance = namespace / "maintenance.lock"
            if maintenance.is_symlink() or maintenance.exists():
                return f"maintenance lock exists in namespace {namespace.name}"
            leases = namespace / "leases"
            if leases.is_symlink():
                return f"symlinked runner lease directory in namespace {namespace.name}"
            if not leases.exists():
                continue
            if not leases.is_dir():
                return f"invalid runner lease directory in namespace {namespace.name}"
            for entry in leases.iterdir():
                # Never silently drop a malformed lease, a foreign file or a
                # stale-PID record: all are protection until independently
                # verified. A later WSL03 slice will add an auditable retire path.
                return f"runner lease or unclassified entry exists: {namespace.name}/{entry.name}"
    except OSError as exc:
        return f"unable to inspect every work-root activity namespace: {exc}"
    return None


class Lease:
    """Held by a long-running runner for its own duration.

    Admission protocol:

    1. refuse if maintenance is already published;
    2. publish this lease;
    3. recheck maintenance; if it appeared between 1 and 2, withdraw the lease
       and refuse admission.

    Maintenance performs the complementary order (publish lock, then scan
    leases), so both sides cannot be admitted even under the race.
    """

    def __init__(self, work_root: Path, project_root: Path, *, command: str, run_id: str):
        self.work_root = work_root
        self.project_root = project_root
        self.command = command
        self.run_id = run_id
        self._lease_id = uuid.uuid4().hex
        self._path: Path | None = None

    def __enter__(self) -> "Lease":
        activity_root = _tree_activity_root(self.work_root, self.project_root)
        maintenance = activity_root / "maintenance.lock"
        if maintenance.is_dir():
            raise TreeActivityError(
                f"refusing to start {self.command}({self.run_id}): maintenance lock held "
                f"for {self.project_root}: {maintenance}"
            )

        leases = activity_root / "leases"
        leases.mkdir(parents=True, exist_ok=True)
        path = leases / f"{self._lease_id}.json"
        payload = {
            "pid": os.getpid(), "hostname": socket.gethostname(), "command": self.command,
            "run_id": self.run_id, "project_root": str(self.project_root.resolve()),
            "started_at": time.time(),
        }
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

        # Complementary second check closes check-then-create race against
        # MaintenanceLock.acquire() publishing its directory first.
        if maintenance.is_dir():
            path.unlink(missing_ok=True)
            raise TreeActivityError(
                f"refusing to start {self.command}({self.run_id}): maintenance began "
                f"during lease admission for {self.project_root}"
            )

        self._path = path
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._path is not None:
            self._path.unlink(missing_ok=True)
            self._path = None


class MaintenanceLock:
    """Held by maintenance (for example pin-bump) for its duration.

    Publish the maintenance directory BEFORE scanning leases. Any runner that
    races with this acquire either has already published a lease (so acquire
    fails), or sees/rechecks the maintenance directory and refuses itself.
    """

    def __init__(self, work_root: Path, project_root: Path):
        self.work_root = work_root
        self.project_root = project_root
        self.path = maintenance_lock_path(work_root, project_root)
        self._acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError as exc:
            raise TreeActivityError(
                f"maintenance lock already held for {self.project_root}: {self.path}"
            ) from exc

        try:
            live = list_live_leases(self.work_root, self.project_root)
            if live:
                names = ", ".join(f"{lease.command}({lease.run_id})" for lease in live)
                raise TreeActivityError(
                    f"refusing to acquire maintenance lock: {len(live)} live lease(s) "
                    f"still active for {self.project_root}: {names}"
                )
            owner = {
                "pid": os.getpid(), "hostname": socket.gethostname(), "started_at": time.time(),
            }
            (self.path / "owner.json").write_text(
                json.dumps(owner, sort_keys=True) + "\n", encoding="utf-8"
            )
            self._acquired = True
        except BaseException:
            (self.path / "owner.json").unlink(missing_ok=True)
            try:
                self.path.rmdir()
            except OSError:
                pass
            raise

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
    """Linux-only, DIAGNOSTIC ONLY -- never authoritative, never a gate."""
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
