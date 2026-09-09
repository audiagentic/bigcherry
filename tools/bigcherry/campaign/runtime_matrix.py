"""Reusable runtime-placement matrix orchestration (RHA06).

This module owns only runtime placement and progress. Build selection,
patch/admission policy, tuning, and server benchmarking remain delegated to
the existing campaign workers supplied by the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..core.environment import Host


class MatrixResolutionError(ValueError):
    """Raised when a complete matrix cannot be resolved before execution."""


@dataclass(frozen=True)
class ResolvedCell:
    """Immutable, identity-bound runtime cell descriptor."""

    cell_id: str
    model_id: str
    devices: tuple[int, ...]
    topology: str
    runtime_profile: str
    arm: str
    build_id: str
    binary: str
    cache_id: str | None
    visibility: tuple[tuple[str, str], ...]
    identity_digest: str
    workload: Mapping[str, Any]

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "cell_id": self.cell_id,
            "model_id": self.model_id,
            "devices": list(self.devices),
            "topology": self.topology,
            "runtime_profile": self.runtime_profile,
            "arm": self.arm,
            "build_id": self.build_id,
            "binary": self.binary,
            "cache_id": self.cache_id,
            "visibility": dict(self.visibility),
            "identity_digest": self.identity_digest,
            "workload": dict(self.workload),
        }


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(b"bigcherry/runtime-cell/v1\0" + encoded, digest_size=16).hexdigest()


def _required_text(raw: Mapping[str, Any], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MatrixResolutionError(f"{where}: {key!r} must be a non-empty string")
    return value.strip()


def resolve_matrix(
    cells: Sequence[Mapping[str, Any]],
    *,
    host: Host,
    known_models: frozenset[str] | None = None,
) -> tuple[ResolvedCell, ...]:
    """Resolve and validate every requested cell without touching a GPU.

    The input is deliberately a plain mapping so a future CLI/config adapter
    can select models and runtime profiles without changing this contract.
    """
    if not cells:
        raise MatrixResolutionError("matrix must contain at least one cell")
    resolved: list[ResolvedCell] = []
    seen_ids: set[str] = set()
    for position, raw in enumerate(cells):
        if not isinstance(raw, Mapping):
            raise MatrixResolutionError(f"cell[{position}] must be an object")
        where = f"cell[{position}]"
        cell_id = _required_text(raw, "cell_id", where)
        if cell_id in seen_ids:
            raise MatrixResolutionError(f"duplicate cell_id {cell_id!r}")
        seen_ids.add(cell_id)
        model_id = _required_text(raw, "model_id", where)
        if known_models is not None and model_id not in known_models:
            raise MatrixResolutionError(f"{where}: unknown model {model_id!r}")
        topology = _required_text(raw, "topology", where)
        runtime_profile = _required_text(raw, "runtime_profile", where)
        arm = _required_text(raw, "arm", where)
        build_id = _required_text(raw, "build_id", where)
        binary = _required_text(raw, "binary", where)
        devices_raw = raw.get("devices")
        if not isinstance(devices_raw, (list, tuple)) or not devices_raw:
            raise MatrixResolutionError(f"{where}: devices must be a non-empty list")
        try:
            devices = tuple(int(value) for value in devices_raw)
        except (TypeError, ValueError) as exc:
            raise MatrixResolutionError(f"{where}: devices must be integers") from exc
        if len(set(devices)) != len(devices):
            raise MatrixResolutionError(f"{where}: duplicate device index")
        visibility = host.gpu_visibility_env(*devices)
        cache_id = raw.get("cache_id")
        if cache_id is not None and (not isinstance(cache_id, str) or not cache_id.strip()):
            raise MatrixResolutionError(f"{where}: cache_id must be non-empty when supplied")
        if arm == "replay" and cache_id is None:
            raise MatrixResolutionError(f"{where}: replay arm requires cache_id")
        workload = raw.get("workload", {})
        if not isinstance(workload, Mapping):
            raise MatrixResolutionError(f"{where}: workload must be an object")
        identity = {
            "cell_id": cell_id,
            "model_id": model_id,
            "devices": list(devices),
            "topology": topology,
            "runtime_profile": runtime_profile,
            "arm": arm,
            "build_id": build_id,
            "binary": binary,
            "cache_id": cache_id,
            "visibility": visibility,
            "workload": dict(workload),
        }
        resolved.append(ResolvedCell(
            cell_id=cell_id, model_id=model_id, devices=devices,
            topology=topology, runtime_profile=runtime_profile, arm=arm,
            build_id=build_id, binary=binary, cache_id=cache_id,
            visibility=tuple(sorted(visibility.items())),
            identity_digest=_digest(identity), workload=dict(workload),
        ))
    return tuple(resolved)


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_matrix(
    cells: Sequence[ResolvedCell],
    *,
    output: str | Path,
    execute: Callable[[ResolvedCell], Mapping[str, Any]],
    quiescent: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run resolved cells serially and expose UI-safe status/event files.

    ``execute`` is the existing tune/benchmark delegate. Its returned mapping
    is copied verbatim under ``child_result``; this layer never upgrades a
    child's admission verdict.
    """
    if not cells:
        raise MatrixResolutionError("cannot run an empty matrix")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    status_path, events_path = root / "status.json", root / "events.jsonl"
    started = time.time()
    completed = 0

    def emit(state: str, *, cell: ResolvedCell | None = None, error: str | None = None) -> None:
        nonlocal completed
        event = {
            "schema_version": 1, "at": time.time(), "state": state,
            "active_cell": cell.cell_id if cell else None,
            "completed": completed, "total": len(cells),
        }
        if error:
            event["error"] = error[:500]
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        status = dict(event)
        status["run_started_at"] = started
        _atomic_json(status_path, status)

    emit("preflight_complete")
    results: list[dict[str, Any]] = []
    for cell in cells:
        if quiescent is not None and not quiescent():
            emit("failed", cell=cell, error="parent quiescence check failed")
            return {"state": "failed", "completed": completed, "total": len(cells), "results": results}
        emit("running", cell=cell)
        try:
            child = dict(execute(cell))
        except Exception as exc:  # noqa: BLE001 - preserve terminal evidence
            emit("failed", cell=cell, error=f"{type(exc).__name__}: {exc}")
            return {"state": "failed", "completed": completed, "total": len(cells), "results": results}
        result = {"cell": cell.document(), "matrix_status": "executed", "child_result": child}
        results.append(result)
        completed += 1
        if quiescent is not None and not quiescent():
            emit("failed", cell=cell, error="post-cell quiescence check failed")
            return {"state": "failed", "completed": completed, "total": len(cells), "results": results}
        emit("cell_complete", cell=cell)
    emit("completed")
    return {"state": "completed", "completed": completed, "total": len(cells), "results": results}


__all__ = ["MatrixResolutionError", "ResolvedCell", "resolve_matrix", "run_matrix"]
