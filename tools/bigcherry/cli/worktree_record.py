"""Who made a worktree and why: the ``WORKTREE.md`` a slice carries and the registry that outlives its removal.

``git`` records the shared identity for every agent, so a stray worktree or branch cannot say who created it or what it was
for. ``slice start`` writes the record; ``slice status`` reads it back and flags worktrees that have none.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

RECORD_NAME = "WORKTREE.md"
REGISTRY_NAME = "bigcherry-worktrees.jsonl"
_FIELDS = ("creator", "created", "branch", "base", "why")


def creator_identity(env: Mapping[str, str] | None = None) -> str:
    """The session that is running this command: instance name, harness and session id, from the container's env."""
    env = os.environ if env is None else env
    instance = env.get("AGENT_INSTANCE", "").strip()
    harness = env.get("AGENT_HARNESS", "").strip()
    session = env.get("CLAUDE_CODE_SESSION_ID", "").strip()
    if not instance and not harness:
        raise RuntimeError("cannot tell which agent is creating this worktree: AGENT_INSTANCE and AGENT_HARNESS are unset")
    text = f"{instance or '?'} ({harness or '?'})"
    return f"{text} session {session}" if session else text


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def _common_dir(root: Path) -> Path:
    raw = Path(_git(root, "rev-parse", "--git-common-dir"))
    return (raw if raw.is_absolute() else root / raw).resolve()


def write_record(worktree: Path, *, root: Path, branch: str, base: str, why: str,
                 creator: str, now: datetime | None = None) -> Path:
    """Write ``WORKTREE.md`` into the worktree (kept out of git) and append the registry line."""
    why = why.strip()
    if not why:
        raise ValueError("a worktree needs a reason: pass --why with what it is for")
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {"creator": creator, "created": stamp, "branch": branch, "base": base, "why": why}
    lines = ["# Worktree record", "",
             "Written by `bigcherry slice start`; not committed. Update `why` if the purpose changes, and say here what",
             "evidence this branch holds before it is removed.", ""]
    lines += [f"- {name}: {record[name]}" for name in _FIELDS]
    path = worktree / RECORD_NAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    common = _common_dir(root)
    exclude = common / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    if f"/{RECORD_NAME}" not in existing.splitlines():
        exclude.write_text(existing + ("" if existing.endswith("\n") or not existing else "\n") + f"/{RECORD_NAME}\n",
                           encoding="utf-8")
    with (common / REGISTRY_NAME).open("a", encoding="utf-8") as registry:
        registry.write(json.dumps({**record, "path": str(worktree)}, sort_keys=True) + "\n")
    return path


def read_record(worktree: Path) -> dict[str, str] | None:
    """The fields of a worktree's ``WORKTREE.md``, or None when it has none."""
    path = worktree / RECORD_NAME
    if not path.is_file():
        return None
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("- ") and ": " in line:
            name, _, value = line[2:].partition(": ")
            if name in _FIELDS:
                fields[name] = value.strip()
    return fields
