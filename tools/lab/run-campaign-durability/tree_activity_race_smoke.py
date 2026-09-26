"""Stress the real HI151 Lease/MaintenanceLock two-phase admission protocol."""
from __future__ import annotations

import json
import multiprocessing as mp
import tempfile
import time
from pathlib import Path

from bigcherry.core.tree_activity import Lease, MaintenanceLock, TreeActivityError


def worker(role: str, work: str, project: str, start, own: str, peer: str, queue) -> None:
    work_path = Path(work)
    project_path = Path(project)
    own_path = Path(own)
    peer_path = Path(peer)
    start.wait(5)
    try:
        context = (
            Lease(work_path, project_path, command="race", run_id=role)
            if role == "lease"
            else MaintenanceLock(work_path, project_path)
        )
        with context:
            own_path.write_text(role)
            # Hold long enough that a concurrently-admitted peer would be seen.
            for _ in range(25):
                if peer_path.exists():
                    queue.put((role, "overlap"))
                    return
                time.sleep(0.002)
            queue.put((role, "entered"))
    except TreeActivityError:
        queue.put((role, "blocked"))
    finally:
        own_path.unlink(missing_ok=True)


def main() -> int:
    ctx = mp.get_context("fork")
    checks = 0
    entered_total = 0
    blocked_total = 0
    with tempfile.TemporaryDirectory(prefix="bigcherry-tree-race-") as temp:
        root = Path(temp)
        project = root / "project"
        work = root / "work"
        project.mkdir()

        for iteration in range(50):
            lease_inside = root / f"lease-{iteration}.inside"
            maintenance_inside = root / f"maintenance-{iteration}.inside"
            start = ctx.Event()
            queue = ctx.Queue()
            lease = ctx.Process(
                target=worker,
                args=("lease", str(work), str(project), start, str(lease_inside), str(maintenance_inside), queue),
            )
            maintenance = ctx.Process(
                target=worker,
                args=("maintenance", str(work), str(project), start, str(maintenance_inside), str(lease_inside), queue),
            )
            lease.start(); maintenance.start(); start.set()
            lease.join(5); maintenance.join(5)
            if lease.is_alive(): lease.kill(); lease.join()
            if maintenance.is_alive(): maintenance.kill(); maintenance.join()
            if lease.exitcode != 0 or maintenance.exitcode != 0:
                raise AssertionError(f"worker failure iteration={iteration}: lease={lease.exitcode} maintenance={maintenance.exitcode}")
            results = [queue.get(timeout=1), queue.get(timeout=1)]
            if any(state == "overlap" for _, state in results):
                raise AssertionError(f"lease and maintenance overlapped iteration={iteration}: {results}")
            entered_total += sum(state == "entered" for _, state in results)
            blocked_total += sum(state == "blocked" for _, state in results)
            # At least one contender must make progress; both may enter serially.
            if not any(state == "entered" for _, state in results):
                raise AssertionError(f"no contender admitted iteration={iteration}: {results}")
            checks += 2

    print(json.dumps({
        "checks": checks,
        "iterations": 50,
        "entered": entered_total,
        "blocked": blocked_total,
        "ok": True,
        "scope": "real-hi151-concurrent-admission-race",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
