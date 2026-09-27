"""Durable campaign-operation identity and verified result rehydration.

This is deliberately not a scheduler.  Executors/Slurm own execution and
resources; these records answer only whether one deterministic campaign stage
was already completed for the exact semantic spec and exact verified inputs.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from bigcherry.tuning.journal import atomic_write

from .executor import ResourceRequest
from .model import digest


class OperationError(RuntimeError):
    pass


_RESULT_STATES = frozenset({"succeeded", "failed", "interrupted"})


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise OperationError(f"operation semantic value is not canonical-JSON compatible: {type(value).__name__}")


def _token(value: str, field: str) -> str:
    text = value.strip()
    if not text or "/" in text or "\\" in text or text in {".", ".."}:
        raise ValueError(f"{field} must be a non-empty path-safe token")
    return text


@dataclass(frozen=True)
class OperationSpec:
    operation_id: str
    kind: str
    command_semantics: Mapping[str, object]
    environment_semantics: tuple[tuple[str, str], ...]
    resources: ResourceRequest
    dependencies: tuple[str, ...]
    declared_outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _token(self.operation_id, "operation_id"))
        object.__setattr__(self, "kind", _token(self.kind, "kind"))
        deps = tuple(self.dependencies)
        if len(deps) != len(set(deps)) or self.operation_id in deps:
            raise ValueError("operation dependencies must be unique and cannot include self")
        for dependency in deps:
            _token(dependency, "dependency")
        outputs = tuple(self.declared_outputs)
        if len(outputs) != len(set(outputs)):
            raise ValueError("declared_outputs must be unique")
        for output in outputs:
            _token(output, "declared output")
        env_keys = [key for key, _ in self.environment_semantics]
        if len(env_keys) != len(set(env_keys)):
            raise ValueError("environment semantic keys must be unique")
        _jsonable(self.command_semantics)

    def document(self) -> dict[str, object]:
        return {
            "schema": "bigcherry.operation-spec.v1",
            "operation_id": self.operation_id,
            "kind": self.kind,
            "command_semantics": _jsonable(self.command_semantics),
            "environment_semantics": [[key, value] for key, value in sorted(self.environment_semantics)],
            "resources": _jsonable(self.resources),
            "dependencies": list(self.dependencies),
            "declared_outputs": list(self.declared_outputs),
        }


@dataclass(frozen=True, order=True)
class ArtifactBinding:
    artifact_id: str
    relative_path: str
    descriptor_hash: str
    content_hash: str
    bytes: int

    def __post_init__(self) -> None:
        _token(self.artifact_id, "artifact_id")
        path = Path(self.relative_path)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("artifact relative_path must remain inside the stage root")
        for field, value in (("descriptor_hash", self.descriptor_hash), ("content_hash", self.content_hash)):
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
                raise ValueError(f"{field} must be a SHA-256 hex digest")
        if self.bytes < 0:
            raise ValueError("artifact byte count cannot be negative")

    def document(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class OperationResult:
    operation_spec_hash: str
    execution_hash: str
    outputs: tuple[ArtifactBinding, ...]
    state: str
    returncode: int | None
    finished_ns: int

    def __post_init__(self) -> None:
        if self.state not in _RESULT_STATES:
            raise ValueError(f"unknown operation result state: {self.state}")
        if self.state == "succeeded" and self.returncode != 0:
            raise ValueError("succeeded operation must have returncode 0")
        if self.state != "succeeded" and self.returncode == 0:
            raise ValueError("non-succeeded operation cannot have returncode 0")
        if len({item.artifact_id for item in self.outputs}) != len(self.outputs):
            raise ValueError("operation outputs must have unique artifact IDs")

    def document(self) -> dict[str, object]:
        return {
            "schema": "bigcherry.operation-result.v1",
            "operation_spec_hash": self.operation_spec_hash,
            "execution_hash": self.execution_hash,
            "outputs": [item.document() for item in self.outputs],
            "state": self.state,
            "returncode": self.returncode,
            "finished_ns": self.finished_ns,
        }


def operation_spec_hash(spec: OperationSpec) -> str:
    return digest(spec.document(), person=b"bc-operation-spec")


def execution_hash(spec_hash: str, inputs: Iterable[ArtifactBinding]) -> str:
    rows = sorted((binding.document() for binding in inputs), key=lambda row: (str(row["artifact_id"]), str(row["content_hash"])))
    return digest({"operation_spec_hash": spec_hash, "inputs": rows}, person=b"bc-operation-exec")


def _sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            state.update(chunk)
    return state.hexdigest()


def bind_artifact(stage_root: Path, artifact_id: str, relative_path: str, *, descriptor: Mapping[str, object] | None = None) -> ArtifactBinding:
    root = stage_root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        resolved_relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise OperationError(f"artifact escapes stage root: {relative_path}") from exc
    if not candidate.is_file():
        raise OperationError(f"declared artifact does not exist: {candidate}")
    content_hash = _sha256(candidate)
    descriptor_doc = {
        "artifact_id": artifact_id,
        "relative_path": resolved_relative,
        "descriptor": _jsonable(descriptor or {}),
    }
    descriptor_hash = hashlib.sha256(
        json.dumps(descriptor_doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    return ArtifactBinding(
        artifact_id=artifact_id,
        relative_path=resolved_relative,
        descriptor_hash=descriptor_hash,
        content_hash=content_hash,
        bytes=candidate.stat().st_size,
    )


def verify_artifact(stage_root: Path, binding: ArtifactBinding) -> bool:
    root = stage_root.resolve()
    candidate = (root / binding.relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return (
        candidate.is_file()
        and candidate.stat().st_size == binding.bytes
        and _sha256(candidate) == binding.content_hash
    )


def _write_immutable(path: Path, document: Mapping[str, object]) -> None:
    if path.exists():
        raise OperationError(f"immutable operation record already exists: {path}")
    atomic_write(
        path,
        json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True).encode("ascii") + b"\n",
    )


def persist_operation_spec(stage_root: Path, spec: OperationSpec) -> str:
    stage_root.mkdir(parents=True, exist_ok=True)
    value = spec.document()
    value["operation_spec_hash"] = operation_spec_hash(spec)
    _write_immutable(stage_root / "operation.json", value)
    return str(value["operation_spec_hash"])


def begin_operation(stage_root: Path, spec: OperationSpec, inputs: Iterable[ArtifactBinding]) -> tuple[str, str]:
    stage_root.mkdir(parents=True, exist_ok=True)
    spec_hash = operation_spec_hash(spec)
    operation_path = stage_root / "operation.json"
    if operation_path.exists():
        existing = json.loads(operation_path.read_text(encoding="ascii"))
        if existing.get("operation_spec_hash") != spec_hash:
            raise OperationError("existing operation.json does not match requested spec")
    else:
        persist_operation_spec(stage_root, spec)
    exec_hash = execution_hash(spec_hash, inputs)
    if (stage_root / "result.json").exists():
        raise OperationError("operation already has an immutable terminal result")
    _write_immutable(
        stage_root / "running.json",
        {
            "schema": "bigcherry.operation-running.v1",
            "operation_spec_hash": spec_hash,
            "execution_hash": exec_hash,
            "started_ns": time.time_ns(),
            "pid": os.getpid(),
        },
    )
    return spec_hash, exec_hash


def publish_result(
    stage_root: Path,
    *,
    spec: OperationSpec,
    inputs: Iterable[ArtifactBinding],
    state: str,
    returncode: int | None,
    outputs: Iterable[ArtifactBinding],
) -> OperationResult:
    spec_hash = operation_spec_hash(spec)
    exec_hash = execution_hash(spec_hash, inputs)
    running_path = stage_root / "running.json"
    if not running_path.is_file():
        raise OperationError("cannot publish operation result without running.json")
    running = json.loads(running_path.read_text(encoding="ascii"))
    if running.get("operation_spec_hash") != spec_hash or running.get("execution_hash") != exec_hash:
        raise OperationError("running marker identity does not match requested execution")
    output_rows = tuple(sorted(outputs, key=lambda item: item.artifact_id))
    declared = set(spec.declared_outputs)
    produced = {item.artifact_id for item in output_rows}
    if state == "succeeded" and produced != declared:
        raise OperationError(f"succeeded operation output set mismatch: {produced!r} != {declared!r}")
    for binding in output_rows:
        if not verify_artifact(stage_root, binding):
            raise OperationError(f"operation output failed content verification: {binding.artifact_id}")
    result = OperationResult(
        operation_spec_hash=spec_hash,
        execution_hash=exec_hash,
        outputs=output_rows,
        state=state,
        returncode=returncode,
        finished_ns=time.time_ns(),
    )
    _write_immutable(stage_root / "result.json", result.document())
    return result


def _parse_binding(value: object) -> ArtifactBinding:
    if not isinstance(value, Mapping):
        raise OperationError("operation artifact binding is malformed")
    return ArtifactBinding(
        artifact_id=str(value["artifact_id"]),
        relative_path=str(value["relative_path"]),
        descriptor_hash=str(value["descriptor_hash"]),
        content_hash=str(value["content_hash"]),
        bytes=int(value["bytes"]),
    )


def load_result(stage_root: Path) -> OperationResult | None:
    path = stage_root / "result.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="ascii"))
        if value.get("schema") != "bigcherry.operation-result.v1":
            raise OperationError("unknown operation result schema")
        return OperationResult(
            operation_spec_hash=str(value["operation_spec_hash"]),
            execution_hash=str(value["execution_hash"]),
            outputs=tuple(_parse_binding(item) for item in value.get("outputs", [])),
            state=str(value["state"]),
            returncode=None if value.get("returncode") is None else int(value["returncode"]),
            finished_ns=int(value["finished_ns"]),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise OperationError(f"invalid operation result at {path}: {exc}") from exc


def durable_state(stage_root: Path) -> str:
    result = load_result(stage_root)
    if result is not None:
        return result.state
    if (stage_root / "running.json").is_file():
        return "interrupted"
    if (stage_root / "operation.json").is_file():
        return "planned"
    return "absent"


def rehydrate_succeeded(stage_root: Path, spec: OperationSpec, inputs: Iterable[ArtifactBinding]) -> OperationResult | None:
    """Return a reusable success only after exact identity and bytes verify."""
    result = load_result(stage_root)
    if result is None or result.state != "succeeded":
        return None
    spec_hash = operation_spec_hash(spec)
    if result.operation_spec_hash != spec_hash:
        return None
    if result.execution_hash != execution_hash(spec_hash, inputs):
        return None
    if {item.artifact_id for item in result.outputs} != set(spec.declared_outputs):
        return None
    if not all(verify_artifact(stage_root, item) for item in result.outputs):
        return None
    return result
