"""Run manifest v1 (WSL01): the one record that says what a run directory is, who made it, and whether it may be cleaned.

The command that creates a run writes the manifest first (state "created"), updates it as the run proceeds and finalises it
at the end. A crashed process leaves it at "running"; nothing here ever infers "deletable" from that. Writes are atomic
(temporary file, fsync, os.replace, fsync of the directory) and updates carry a generation number, so two writers cannot
silently overwrite each other. Run lifecycle, harvest lifecycle and retention are separate fields on purpose: a completed
run is not a harvested run, and a harvested run is not a deletable one.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MANIFEST_NAME = "run.manifest.json"

RUN_KINDS = frozenset({"ab", "accept", "stream", "replay", "capture", "kl", "profile", "build", "bump", "release"})

RUN_STATES = ("created", "running", "completed", "failed", "cancelled", "interrupted")
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
HARVEST_STATES = ("pending", "harvesting", "verified", "failed")
RETENTION_STATES = ("active", "eligible", "quarantined", "purged")

# run lifecycle: what may follow what. "interrupted" is set only by reconciliation after proving the process is gone.
_TRANSITIONS = {
    "created": {"running", "failed", "cancelled"},
    "running": {"completed", "failed", "cancelled", "interrupted"},
    "interrupted": {"failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}
_HARVEST_TRANSITIONS = {
    "pending": {"harvesting"},
    "harvesting": {"verified", "failed"},
    "failed": {"harvesting"},
    "verified": set(),
}

_REQUIRED = ("schema_version", "run_id", "kind", "purpose", "plan", "engine", "created_at", "execution", "owner", "inputs",
             "references", "state", "generation", "harvest", "retention")


class ManifestError(ValueError):
    """The manifest is missing, unreadable, of an unknown version, or inconsistent."""


def utc_stamp(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if now is None else now))


def new_manifest(*, run_id: str, kind: str, purpose: str, plan_item: str | None, engine: str, argv: list, owner: dict,
                 inputs: list | None = None, references: list | None = None, git_commit: str | None = None,
                 git_dirty: bool | None = None, host: dict | None = None, subkind: str | None = None,
                 now: float | None = None) -> dict[str, Any]:
    if kind not in RUN_KINDS:
        raise ManifestError(f"unknown run kind {kind!r}; kinds are {sorted(RUN_KINDS)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "kind": kind,
        "subkind": subkind,
        "purpose": purpose,
        "plan": {"item": plan_item},
        "engine": engine,
        "created_at": utc_stamp(now),
        "started_at": None,
        "ended_at": None,
        "execution": {"argv": list(argv), "git_commit": git_commit, "git_dirty": git_dirty, "exit_code": None},
        "host": host or {},
        "owner": dict(owner),
        "inputs": list(inputs or []),         # [{"role", "uri", "sha256"}]
        "references": list(references or []),  # [{"type": run|build|source|object, "id", "role"}] typed GC edges
        "state": "created",
        "generation": 1,
        "size_bytes": None,
        "harvest": {"state": "pending", "recipe": None, "bundle_id": None, "verified_at": None},
        "retention": {"state": "active", "class": "standard", "retain_until": None, "hold": False},
        "provenance": "recorded",             # "legacy_inferred" for adopted directories (WSL06)
    }


def validate_manifest(m: Any) -> dict[str, Any]:
    if not isinstance(m, dict):
        raise ManifestError("manifest is not an object")
    if m.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"manifest schema version {m.get('schema_version')!r}, this reader takes {SCHEMA_VERSION}")
    for key in _REQUIRED:
        if key not in m:
            raise ManifestError(f"manifest lacks {key}")
    if m["kind"] not in RUN_KINDS:
        raise ManifestError(f"unknown run kind {m['kind']!r}")
    if m["state"] not in RUN_STATES:
        raise ManifestError(f"unknown run state {m['state']!r}")
    if m["harvest"].get("state") not in HARVEST_STATES:
        raise ManifestError(f"unknown harvest state {m['harvest'].get('state')!r}")
    if m["retention"].get("state") not in RETENTION_STATES:
        raise ManifestError(f"unknown retention state {m['retention'].get('state')!r}")
    if not isinstance(m["generation"], int) or m["generation"] < 1:
        raise ManifestError("generation must be a positive integer")
    for ref in m["references"]:
        if not isinstance(ref, dict) or ref.get("type") not in {"run", "build", "source", "object"} or not ref.get("id"):
            raise ManifestError(f"malformed reference {ref!r}")
    return m


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return                     # not every platform can open a directory; the replace itself is still atomic
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def manifest_path(run_dir: Path) -> Path:
    return run_dir / MANIFEST_NAME


def write_manifest(run_dir: Path, m: dict[str, Any]) -> None:
    validate_manifest(m)
    write_atomic(manifest_path(run_dir), json.dumps(m, indent=1, sort_keys=True) + "\n")


def read_manifest(run_dir: Path) -> dict[str, Any]:
    path = manifest_path(run_dir)
    try:
        return validate_manifest(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise ManifestError(f"{path} does not exist") from exc
    except (OSError, ValueError) as exc:
        if isinstance(exc, ManifestError):
            raise
        raise ManifestError(f"{path} is unreadable: {exc}") from exc


class _Lock:
    """A short-lived exclusive lock file around one manifest update."""

    def __init__(self, run_dir: Path, timeout: float = 10.0):
        self.path = run_dir / (MANIFEST_NAME + ".lock")
        self.timeout = timeout

    def __enter__(self) -> "_Lock":
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise ManifestError(f"{self.path} is held; another writer is updating this run") from None
                time.sleep(0.02)

    def __exit__(self, *exc: object) -> None:
        try:
            os.unlink(self.path)
        except OSError:
            pass


def update_manifest(run_dir: Path, *, expected_generation: int, state: str | None = None, harvest_state: str | None = None,
                    patch: dict | None = None, now: float | None = None) -> dict[str, Any]:
    """Apply a change under the run's lock. Fails on a stale generation or an illegal state transition."""
    with _Lock(run_dir):
        m = read_manifest(run_dir)
        if m["generation"] != expected_generation:
            raise ManifestError(f"stale update: manifest is at generation {m['generation']}, caller read {expected_generation}")
        if state is not None and state != m["state"]:
            if state not in _TRANSITIONS.get(m["state"], set()):
                raise ManifestError(f"illegal run transition {m['state']} -> {state}")
            m["state"] = state
            if state == "running" and not m["started_at"]:
                m["started_at"] = utc_stamp(now)
            if state in TERMINAL_STATES or state == "interrupted":
                m["ended_at"] = utc_stamp(now)
        if harvest_state is not None and harvest_state != m["harvest"]["state"]:
            if harvest_state not in _HARVEST_TRANSITIONS.get(m["harvest"]["state"], set()):
                raise ManifestError(f"illegal harvest transition {m['harvest']['state']} -> {harvest_state}")
            m["harvest"]["state"] = harvest_state
            if harvest_state == "verified":
                m["harvest"]["verified_at"] = utc_stamp(now)
        for key, value in (patch or {}).items():
            if key in {"schema_version", "run_id", "kind", "generation", "state", "created_at"}:
                raise ManifestError(f"{key} cannot be patched")
            if isinstance(value, dict) and isinstance(m.get(key), dict):
                m[key].update(value)
            else:
                m[key] = value
        m["generation"] += 1
        write_manifest(run_dir, m)
        return m
