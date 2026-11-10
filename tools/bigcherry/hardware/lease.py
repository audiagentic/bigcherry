"""Positive verification that THIS process holds a named agent-lease (default ``gpu``).

The lease is a JSON file in the lease directory (``/control/leases/<name>.json``: ``instance``, ``pids``, ``expires``).
Holding it means: the file exists, it has not expired, one of its recorded pids is a live ancestor of this process (the
command ``agent-lease acquire ... -- COMMAND`` started us, or a shell that took the lease did), and, when the
instance is known, the lease's instance is ours. Text from ``agent-lease status`` is never parsed.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

LEASE_DIR = "/control/leases"


def _alive(pid: int, proc: Path) -> bool:
    try:
        return ") Z " not in (proc / str(pid) / "stat").read_text()
    except OSError:
        return False


def _ancestors(pid: int, proc: Path) -> list[int]:
    out = []
    for _ in range(64):
        out.append(pid)
        try:
            stat = (proc / str(pid) / "stat").read_text()
            pid = int(stat.rsplit(") ", 1)[1].split(" ")[1])
        except (OSError, ValueError, IndexError):
            break
        if pid <= 1:
            break
    return out


def held_by_this_process(resource: str = "gpu", *, lease_dir: str | Path | None = None, proc: Path = Path("/proc"),
                         pid: int | None = None, instance: str | None = None, now: datetime | None = None) -> bool:
    directory = Path(lease_dir or os.environ.get("LEASE_DIR") or LEASE_DIR)
    try:
        lease = json.loads((directory / f"{resource}.json").read_text(encoding="utf-8"))
        expires = datetime.fromisoformat(str(lease["expires"]).replace("Z", "+00:00"))
        pids = [int(p) for p in lease["pids"]]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if expires <= (now or datetime.now(timezone.utc)):
        return False
    me = instance if instance is not None else (os.environ.get("AGENT_INSTANCE") or os.environ.get("CLAUDE_RC_NAME"))
    if me and lease.get("instance") != me:
        return False
    live = {p for p in pids if _alive(p, proc)}
    return bool(live & set(_ancestors(pid if pid is not None else os.getpid(), proc)))
