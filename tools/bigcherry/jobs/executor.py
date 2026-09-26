"""Platform-neutral execution contract for the BigCherry job service (RCD04).

Scientific capability resolution happens before this layer. The executor sees
only resolved scheduler resources; it never decides model/VRAM/peer eligibility
or statistical-series hardware identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Protocol


class ExecutorError(RuntimeError):
    pass


class ExecutorCapabilityError(ExecutorError):
    pass


class ExecutorControl(str, Enum):
    HOLD = "hold"
    RELEASE = "release"


class ExecutionState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    HELD = "held"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SchedulerGpuRequest:
    architecture: str
    reserved_count: int

    def __post_init__(self) -> None:
        if not self.architecture.strip() or any(ch.isspace() for ch in self.architecture):
            raise ValueError("GPU architecture must be a non-empty token")
        if self.reserved_count < 1:
            raise ValueError("GPU reserved_count must be >= 1")


@dataclass(frozen=True)
class ResourceRequest:
    cpu_slots: int
    gpu: SchedulerGpuRequest | None
    activity_class: str
    memory_bytes: int | None
    timeout_seconds: int

    def __post_init__(self) -> None:
        if self.cpu_slots < 1:
            raise ValueError("cpu_slots must be >= 1")
        if self.memory_bytes is not None and self.memory_bytes < 1:
            raise ValueError("memory_bytes must be positive when supplied")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")
        if not self.activity_class.strip():
            raise ValueError("activity_class is required")


@dataclass(frozen=True)
class ExecutionRequest:
    execution_id: str
    command: tuple[str, ...]
    cwd: str
    env: tuple[tuple[str, str], ...]
    stdout_path: str
    stderr_path: str
    resources: ResourceRequest
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.execution_id.strip():
            raise ValueError("execution_id is required")
        if not self.command or not self.command[0]:
            raise ValueError("command is required")
        keys = [key for key, _ in self.env]
        if len(keys) != len(set(keys)):
            raise ValueError("execution environment keys must be unique")

    @property
    def environment(self) -> Mapping[str, str]:
        return dict(self.env)


@dataclass(frozen=True)
class ExecutionHandle:
    executor: str
    native_id: str
    execution_id: str


@dataclass(frozen=True)
class ExecutionStatus:
    state: ExecutionState
    reason: str | None = None
    native_state: str | None = None


@dataclass(frozen=True)
class Allocation:
    """Executor-native allocation identity before RCD12 stable-ID resolution."""

    native_gpu_ids: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ExecutorEvent:
    sequence: int
    kind: str
    payload: Mapping[str, object]


class Executor(Protocol):
    def submit(self, request: ExecutionRequest) -> ExecutionHandle: ...
    def status(self, handle: ExecutionHandle) -> ExecutionStatus: ...
    def cancel(self, handle: ExecutionHandle) -> None: ...
    def control(self, handle: ExecutionHandle, action: ExecutorControl) -> None: ...
    def allocation(self, handle: ExecutionHandle) -> Allocation | None: ...
    def events(
        self, handle: ExecutionHandle, *, after: int | None = None
    ) -> Iterable[ExecutorEvent]: ...
