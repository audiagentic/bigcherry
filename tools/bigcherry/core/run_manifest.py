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
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from .run_records import RunRecordStore

SCHEMA_VERSION = 1
MANIFEST_NAME = "run.manifest.json"

# New producers still create only nested run kinds.
NESTED_RUN_KINDS = frozenset({"ab", "accept", "stream", "replay", "capture", "kl", "profile", "build", "bump", "release"})
RUN_KINDS = NESTED_RUN_KINDS

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
                 now: float | None = None, collect_host_facts: bool = False) -> dict[str, Any]:
    if kind not in RUN_KINDS:
        raise ManifestError(f"unknown run kind {kind!r}; kinds are {sorted(RUN_KINDS)}")
    if collect_host_facts:
        if host is not None:
            raise ManifestError("cannot supply host and request collected host facts")
        from .host_facts import collect_host_facts as collect_snapshot

        host = collect_snapshot()
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
        "host": dict(host or {}),
        "owner": dict(owner),
        "inputs": list(inputs or []),         # [{"role", "uri", "sha256"}]
        "references": list(references or []),  # typed GC edges, including measurement_record SHA
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
    if not isinstance(m["references"], list):
        raise ManifestError("references must be a list")
    for ref in m["references"]:
        if not isinstance(ref, dict) or ref.get("type") not in {
            "run", "build", "source", "object", "measurement_record"
        } or not isinstance(ref.get("id"), str) or not ref["id"]:
            raise ManifestError(f"malformed reference {ref!r}")
        if ref["type"] == "measurement_record" and (
            re.fullmatch(r"[0-9a-f]{64}", ref["id"]) is None
            or ref.get("role") != "measurement"
            or set(ref) != {"type", "id", "role"}
        ):
            raise ManifestError("measurement_record reference must contain a SHA256 id and measurement role")
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

    def __init__(self, run_dir: Path, timeout: float = 10.0, *, dir_fd: int | None = None):
        self.name = MANIFEST_NAME + ".lock"
        self.path = run_dir / self.name
        self.dir_fd = dir_fd
        self.timeout = timeout

    def _target(self) -> str:
        return self.name if self.dir_fd is not None else str(self.path)

    def __enter__(self) -> Self:
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self._target(), os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                             0o600, dir_fd=self.dir_fd)
                try:
                    os.write(fd, str(os.getpid()).encode())
                finally:
                    os.close(fd)
                return self
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise ManifestError(f"{self.path} is held; another writer is updating this run") from None
                time.sleep(0.02)

    def __exit__(self, *exc: object) -> None:
        try:
            os.unlink(self._target(), dir_fd=self.dir_fd)
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


def link_measurement_record(run_dir: Path, store: RunRecordStore, digest: str) -> dict[str, Any]:
    """Atomically link a validated immutable MRC01 record to its WSL01 lifecycle run.

    The store remains the record owner and the WSL01 manifest remains the
    lifecycle owner. Verify identity before publishing the typed edge.
    Repeated linking is idempotent and does not change manifest generation.
    """
    record = store.get_record(digest)  # checks SHA, shape, and every required blob
    with _Lock(run_dir):
        manifest = read_manifest(run_dir)
        if manifest["run_id"] != record["run_id"]:
            raise ManifestError("record run_id does not match lifecycle manifest")
        if manifest["engine"] != record["identity"]["engine"]:
            raise ManifestError("record engine does not match lifecycle manifest")
        edge = {"type": "measurement_record", "id": digest, "role": "measurement"}
        if edge in manifest["references"]:
            return manifest
        manifest["references"].append(edge)
        manifest["generation"] += 1
        write_manifest(run_dir, manifest)
        return manifest


def resolve_measurement_records(run_dir: Path, store: RunRecordStore) -> list[dict[str, Any]]:
    """Resolve only typed, content-verified record edges from a lifecycle run."""
    manifest = read_manifest(run_dir)
    rows = []
    for ref in manifest["references"]:
        if ref["type"] != "measurement_record":
            continue
        record = store.get_record(ref["id"])
        if record["run_id"] != manifest["run_id"] or record["identity"]["engine"] != manifest["engine"]:
            raise ManifestError("measurement_record edge points to another run or engine")
        rows.append({"sha256": ref["id"], "record": record})
    return rows


def find_run_for_record(runs_root: Path, store: RunRecordStore, digest: str) -> tuple[Path, dict[str, Any]]:
    """Resolve record -> manifest inside an explicitly supplied runs root.

    Never trust a path from record provenance as a directory locator; require
    a matching typed backlink and fail on ambiguous or unlinked run identities.
    """
    record = store.get_record(digest)
    root = Path(runs_root)
    if root.is_symlink() or not root.is_dir():
        raise ManifestError("unsafe or missing runs root")
    matches = []
    found_identity = False
    for path in root.rglob(MANIFEST_NAME):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ManifestError(f"unsafe run manifest: {path}")
        manifest = read_manifest(path.parent)
        if manifest["run_id"] != record["run_id"]:
            continue
        found_identity = True
        if manifest["engine"] != record["identity"]["engine"]:
            raise ManifestError("run_id matches manifest with a different engine")
        if {"type": "measurement_record", "id": digest, "role": "measurement"} in manifest["references"]:
            matches.append((path.parent, manifest))
    if len(matches) > 1:
        raise ManifestError("ambiguous manifests for measurement record")
    if not matches:
        raise ManifestError("unlinked measurement record" if found_identity else "no lifecycle run for record")
    return matches[0]
