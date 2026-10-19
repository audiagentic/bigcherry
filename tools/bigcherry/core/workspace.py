"""The managed work root (WSL01): one layout, one way to create a run.

    <work root>/
      builds/<lane>/<plan-id>/<build-id>/   build outputs (one canonical place; a build run only carries a manifest)
      sources/<slice-id>/                   materialised sources
      runs/<kind>/<run-id>/                 every experiment, benchmark, acceptance, sweep, capture
      cache/                                rebuildable and expendable (tune caches, ccache)
      objects/sha256/<xx>/<digest>/         valuable immutable blobs (captures, KL references), shared by typed references
      evidence-hold/<run-id>/               harvested bundles waiting for the repo
      trash/<deletion-id>/                  quarantine before a purge (same filesystem as runs/)
      .locks/

A run id is <UTC stamp>-<topic slug>-<12 hex of a uuid>: the uuid is the identity, the rest is for the reader, and an id is
never reused. A run directory is created with an exclusive mkdir, so two creators can never share one.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import run_manifest as rm

LAYOUT_DIRS = ("builds", "sources", "runs", "cache", "objects", "evidence-hold", "trash", ".locks")
_SLUG = re.compile(r"[^a-z0-9]+")


class WorkspaceError(RuntimeError):
    pass


def slug(topic: str, max_length: int = 32) -> str:
    s = _SLUG.sub("-", topic.lower()).strip("-")[:max_length].strip("-")
    if not s:
        raise WorkspaceError(f"topic {topic!r} has no usable characters")
    return s


def new_run_id(topic: str, now: float | None = None) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() if now is None else now))
    return f"{stamp}-{slug(topic)}-{uuid.uuid4().hex[:12]}"


def ensure_layout(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in LAYOUT_DIRS:
        (root / name).mkdir(exist_ok=True)
    return root


def run_dir(root: Path, kind: str, run_id: str) -> Path:
    if kind not in rm.RUN_KINDS:
        raise WorkspaceError(f"unknown run kind {kind!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise WorkspaceError(f"run id {run_id!r} is not path safe")
    return Path(root) / "runs" / kind / run_id


@dataclass(frozen=True)
class RunHandle:
    run_id: str
    kind: str
    path: Path

    def manifest(self) -> dict:
        return rm.read_manifest(self.path)


def _git(args: list, cwd: Path | None) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def source_identity(repo: Path | None) -> tuple[str | None, bool | None]:
    """(commit, dirty) of the repository the command ran from; (None, None) when it is not a git checkout."""
    if repo is None:
        return None, None
    commit = _git(["rev-parse", "HEAD"], repo)
    if commit is None:
        return None, None
    return commit, bool(_git(["status", "--porcelain"], repo))


def create_run(root: Path, kind: str, topic: str, *, plan_item: str | None, engine: str, argv: list, purpose: str = "",
               inputs: list | None = None, references: list | None = None, subkind: str | None = None,
               repo: Path | None = None, now: float | None = None) -> RunHandle:
    """Create <root>/runs/<kind>/<run-id>/ with its initial manifest. Never overwrites; retries on an id collision."""
    if kind not in rm.RUN_KINDS:
        raise WorkspaceError(f"unknown run kind {kind!r}; kinds are {sorted(rm.RUN_KINDS)}")
    ensure_layout(root)
    parent = Path(root) / "runs" / kind
    parent.mkdir(parents=True, exist_ok=True)
    commit, dirty = source_identity(repo)
    owner = {"uid": os.getuid() if hasattr(os, "getuid") else None, "pid": os.getpid()}
    for _ in range(8):
        run_id = new_run_id(topic, now)
        path = parent / run_id
        try:
            path.mkdir()
        except FileExistsError:
            continue
        manifest = rm.new_manifest(run_id=run_id, kind=kind, purpose=purpose, plan_item=plan_item, engine=engine, argv=argv,
                                   owner=owner, inputs=inputs, references=references, git_commit=commit, git_dirty=dirty,
                                   subkind=subkind, now=now)
        rm.write_manifest(path, manifest)
        return RunHandle(run_id, kind, path)
    raise WorkspaceError("could not allocate a unique run id")
