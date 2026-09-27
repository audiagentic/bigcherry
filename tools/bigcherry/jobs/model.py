"""Versioned scheduler-neutral job-service domain records."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

SCHEMA_VERSION = "bigcherry.jobs.v1"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest(value: object, *, person: bytes = b"bc-jobs-v1") -> str:
    return hashlib.blake2b(
        canonical_bytes(value), digest_size=20, person=person[:16]
    ).hexdigest()


def _pairs(
    value: Mapping[str, str]
    | tuple[tuple[str, str], ...]
    | list[list[str]]
    | list[tuple[str, str]]
    | None,
) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    normalized = tuple(sorted((str(k), str(v)) for k, v in items))
    if len({k for k, _ in normalized}) != len(normalized):
        raise ValueError("duplicate mapping keys")
    return normalized


@dataclass(frozen=True)
class GpuRequirement:
    architecture: str
    count: int = 1
    min_vram_bytes: int = 0
    homogeneous_model: bool = True
    model: str | None = None
    require_peer_access: bool = False
    exact_device_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.architecture or any(ch.isspace() for ch in self.architecture):
            raise ValueError("GPU architecture must be a non-empty token")
        if self.count < 1:
            raise ValueError("GPU count must be >= 1")
        if self.min_vram_bytes < 0:
            raise ValueError("min_vram_bytes must be >= 0")
        if self.exact_device_ids and len(self.exact_device_ids) != self.count:
            raise ValueError("exact_device_ids length must equal count")


@dataclass(frozen=True)
class TargetPolicy:
    executor_id: str = "brutus"
    host_id: str | None = None
    platform_family: str | None = None

    def __post_init__(self) -> None:
        if not self.executor_id.strip():
            raise ValueError("executor_id is required")


@dataclass(frozen=True)
class JobSpec:
    """One planned scientific session before attempt-specific commit resolution."""

    patch: str
    architecture: str
    model: str
    session: int
    planned_sessions: int
    producer: str | None = None
    hip_path: str | None = None
    baseline_source: str = "bigcherry-tuning"
    common_patches: tuple[str, ...] = ()
    producer_inputs: tuple[tuple[str, str], ...] = ()
    producer_corpus: str | None = None
    production_lane: bool = False
    code_ref: str = "patch-refactor"
    gpu: GpuRequirement | None = None
    target: TargetPolicy = field(default_factory=TargetPolicy)
    timeout_seconds: int = 2700
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.patch.strip():
            raise ValueError("patch is required")
        if not self.architecture.strip():
            raise ValueError("architecture is required")
        if not self.model.strip():
            raise ValueError("model is required")
        if (
            self.session < 1
            or self.planned_sessions < 1
            or self.session > self.planned_sessions
        ):
            raise ValueError("session must be in 1..planned_sessions")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")
        if self.gpu is not None and self.gpu.architecture != self.architecture:
            raise ValueError("gpu architecture must match job architecture")
        object.__setattr__(
            self, "common_patches", tuple(sorted(set(self.common_patches)))
        )
        object.__setattr__(self, "producer_inputs", _pairs(self.producer_inputs))

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["schema"] = SCHEMA_VERSION
        return value

    @property
    def request_hash(self) -> str:
        return digest(self.to_dict(), person=b"bc-job-request")


@dataclass(frozen=True)
class BatchSpec:
    patch: str
    architectures: tuple[str, ...]
    model: str
    planned_sessions: int = 4
    producer: str | None = None
    hip_path: str | None = None
    baseline_source: str = "bigcherry-tuning"
    common_patches: tuple[str, ...] = ()
    producer_inputs: tuple[tuple[str, str], ...] = ()
    producer_corpus: str | None = None
    production_lane: bool = False
    code_ref: str = "patch-refactor"
    gpu_count: int = 1
    min_vram_bytes: int = 0
    require_peer_access: bool = False
    target: TargetPolicy = field(default_factory=TargetPolicy)
    timeout_seconds: int = 2700
    priority: int = 0

    def __post_init__(self) -> None:
        architectures = tuple(sorted(set(self.architectures)))
        if not architectures:
            raise ValueError("at least one architecture is required")
        if self.planned_sessions < 1:
            raise ValueError("planned_sessions must be >= 1")
        if self.gpu_count < 1:
            raise ValueError("gpu_count must be >= 1")
        object.__setattr__(self, "architectures", architectures)
        object.__setattr__(
            self, "common_patches", tuple(sorted(set(self.common_patches)))
        )
        object.__setattr__(self, "producer_inputs", _pairs(self.producer_inputs))

    def expand(self) -> tuple[JobSpec, ...]:
        jobs: list[JobSpec] = []
        for architecture in self.architectures:
            gpu = GpuRequirement(
                architecture=architecture,
                count=self.gpu_count,
                min_vram_bytes=self.min_vram_bytes,
                require_peer_access=self.require_peer_access,
            )
            for session in range(1, self.planned_sessions + 1):
                jobs.append(
                    JobSpec(
                        patch=self.patch,
                        architecture=architecture,
                        model=self.model,
                        session=session,
                        planned_sessions=self.planned_sessions,
                        producer=self.producer,
                        hip_path=self.hip_path,
                        baseline_source=self.baseline_source,
                        common_patches=self.common_patches,
                        producer_inputs=self.producer_inputs,
                        producer_corpus=self.producer_corpus,
                        production_lane=self.production_lane,
                        code_ref=self.code_ref,
                        gpu=gpu,
                        target=self.target,
                        timeout_seconds=self.timeout_seconds,
                        priority=self.priority,
                    )
                )
        return tuple(jobs)

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["schema"] = SCHEMA_VERSION
        return value

    @property
    def request_hash(self) -> str:
        return digest(self.to_dict(), person=b"bc-batch-request")


def target_from_mapping(value: Mapping[str, Any] | None) -> TargetPolicy:
    value = value or {}
    return TargetPolicy(
        executor_id=str(value.get("executor_id", "brutus")),
        host_id=None if value.get("host_id") is None else str(value["host_id"]),
        platform_family=(
            None
            if value.get("platform_family") is None
            else str(value["platform_family"])
        ),
    )


def gpu_from_mapping(value: Mapping[str, Any] | None) -> GpuRequirement | None:
    if value is None:
        return None
    return GpuRequirement(
        architecture=str(value["architecture"]),
        count=int(value.get("count", 1)),
        min_vram_bytes=int(value.get("min_vram_bytes", 0)),
        homogeneous_model=bool(value.get("homogeneous_model", True)),
        model=None if value.get("model") is None else str(value["model"]),
        require_peer_access=bool(value.get("require_peer_access", False)),
        exact_device_ids=tuple(str(item) for item in value.get("exact_device_ids", [])),
    )


def job_from_mapping(value: Mapping[str, Any]) -> JobSpec:
    schema = value.get("schema", SCHEMA_VERSION)
    if schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported jobs schema: {schema!r}")
    return JobSpec(
        patch=str(value["patch"]),
        architecture=str(value["architecture"]),
        model=str(value["model"]),
        session=int(value["session"]),
        planned_sessions=int(value["planned_sessions"]),
        producer=None if value.get("producer") is None else str(value["producer"]),
        hip_path=None if value.get("hip_path") is None else str(value["hip_path"]),
        baseline_source=str(value.get("baseline_source", "bigcherry-tuning")),
        common_patches=tuple(str(item) for item in value.get("common_patches", [])),
        producer_inputs=_pairs(value.get("producer_inputs")),
        producer_corpus=(
            None
            if value.get("producer_corpus") is None
            else str(value["producer_corpus"])
        ),
        production_lane=bool(value.get("production_lane", False)),
        code_ref=str(value.get("code_ref", "patch-refactor")),
        gpu=gpu_from_mapping(value.get("gpu")),
        target=target_from_mapping(value.get("target")),
        timeout_seconds=int(value.get("timeout_seconds", 2700)),
        priority=int(value.get("priority", 0)),
    )


def batch_from_mapping(value: Mapping[str, Any]) -> BatchSpec:
    schema = value.get("schema", SCHEMA_VERSION)
    if schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported jobs schema: {schema!r}")
    architectures = value.get("architectures")
    if isinstance(architectures, str):
        architectures = [architectures]
    if not isinstance(architectures, list):
        raise ValueError("architectures must be an array")
    common = value.get("common_patches", [])
    inputs = value.get("producer_inputs", {})
    return BatchSpec(
        patch=str(value["patch"]),
        architectures=tuple(str(item) for item in architectures),
        model=str(value["model"]),
        planned_sessions=int(value.get("planned_sessions", 4)),
        producer=None if value.get("producer") is None else str(value["producer"]),
        hip_path=None if value.get("hip_path") is None else str(value["hip_path"]),
        baseline_source=str(value.get("baseline_source", "bigcherry-tuning")),
        common_patches=tuple(str(item) for item in common),
        producer_inputs=_pairs(inputs),
        producer_corpus=(
            None
            if value.get("producer_corpus") is None
            else str(value["producer_corpus"])
        ),
        production_lane=bool(value.get("production_lane", False)),
        code_ref=str(value.get("code_ref", "patch-refactor")),
        gpu_count=int(value.get("gpu_count", 1)),
        min_vram_bytes=int(value.get("min_vram_bytes", 0)),
        require_peer_access=bool(value.get("require_peer_access", False)),
        target=target_from_mapping(value.get("target")),
        timeout_seconds=int(value.get("timeout_seconds", 2700)),
        priority=int(value.get("priority", 0)),
    )
