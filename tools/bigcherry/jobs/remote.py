"""JSON remote-agent Executor for Windows and non-Slurm hosts.

The transport is intentionally narrow: a remote BigCherry worker owns local
process lifetime and resource locks.  The controller never sends arbitrary
interactive shell fragments as job state.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import dataclass
from typing import Iterable, Protocol

from .executor import (
    Allocation,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionState,
    ExecutionStatus,
    ExecutorControl,
    ExecutorError,
    ExecutorEvent,
    ResourceRequest,
    SchedulerGpuRequest,
)


class RemoteTransport(Protocol):
    def call(self, action: str, payload: dict[str, object]) -> dict[str, object]: ...


@dataclass(frozen=True)
class SshWorkerConfig:
    host: str
    worker_command: str = "python -m bigcherry.jobs.worker"
    ssh_binary: str = "ssh"


class SshJsonTransport:
    def __init__(self, config: SshWorkerConfig) -> None:
        self.config = config

    def call(self, action: str, payload: dict[str, object]) -> dict[str, object]:
        if action not in {"submit", "correlate", "status", "cancel", "control", "allocation", "events", "capabilities"}:
            raise ValueError(action)
        completed = subprocess.run(
            (self.config.ssh_binary, self.config.host, f"{self.config.worker_command} {action}"),
            input=json.dumps(payload, sort_keys=True), text=True, capture_output=True,
        )
        if completed.returncode != 0:
            raise ExecutorError(f"remote worker {action} failed: {completed.stderr.strip()}")
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ExecutorError(f"remote worker returned invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ExecutorError("remote worker response must be an object")
        if value.get("error"):
            raise ExecutorError(str(value["error"]))
        return value


def _request_dict(request: ExecutionRequest) -> dict[str, object]:
    return dataclasses.asdict(request)


def request_from_dict(value: dict[str, object]) -> ExecutionRequest:
    resources = value["resources"]
    assert isinstance(resources, dict)
    gpu_value = resources.get("gpu")
    gpu = None
    if isinstance(gpu_value, dict):
        gpu = SchedulerGpuRequest(str(gpu_value["architecture"]), int(gpu_value["reserved_count"]))
    return ExecutionRequest(
        execution_id=str(value["execution_id"]),
        command=tuple(str(item) for item in value["command"]),
        cwd=str(value["cwd"]),
        env=tuple((str(k), str(v)) for k, v in value.get("env", [])),
        stdout_path=str(value["stdout_path"]),
        stderr_path=str(value["stderr_path"]),
        resources=ResourceRequest(
            cpu_slots=int(resources["cpu_slots"]), gpu=gpu,
            activity_class=str(resources["activity_class"]),
            memory_bytes=None if resources.get("memory_bytes") is None else int(resources["memory_bytes"]),
            timeout_seconds=int(resources["timeout_seconds"]),
        ),
        dependencies=tuple(str(item) for item in value.get("dependencies", [])),
    )


class RemoteExecutor:
    def __init__(self, name: str, transport: RemoteTransport) -> None:
        if not name.strip():
            raise ValueError("remote executor name is required")
        self.name = name
        self.transport = transport

    def submit(self, request: ExecutionRequest) -> ExecutionHandle:
        value = self.transport.call("submit", {"request": _request_dict(request)})
        return ExecutionHandle(self.name, str(value["native_id"]), request.execution_id)

    def correlate(self, execution_id: str) -> tuple[ExecutionHandle, ...]:
        value = self.transport.call("correlate", {"execution_id": execution_id})
        ids = value.get("native_ids", [])
        return tuple(ExecutionHandle(self.name, str(native), execution_id) for native in ids)

    def status(self, handle: ExecutionHandle) -> ExecutionStatus:
        value = self.transport.call("status", _handle_payload(handle))
        try:
            state = ExecutionState(str(value["state"]))
        except ValueError:
            state = ExecutionState.UNKNOWN
        return ExecutionStatus(state, None if value.get("reason") is None else str(value["reason"]), None if value.get("native_state") is None else str(value["native_state"]))

    def cancel(self, handle: ExecutionHandle) -> None:
        self.transport.call("cancel", _handle_payload(handle))

    def control(self, handle: ExecutionHandle, action: ExecutorControl) -> None:
        payload = _handle_payload(handle)
        payload["action"] = action.value
        self.transport.call("control", payload)

    def allocation(self, handle: ExecutionHandle) -> Allocation | None:
        value = self.transport.call("allocation", _handle_payload(handle))
        if value.get("allocation") is None:
            return None
        allocation = value["allocation"]
        assert isinstance(allocation, dict)
        return Allocation(
            native_gpu_ids=tuple(str(item) for item in allocation.get("native_gpu_ids", [])),
            environment=tuple((str(k), str(v)) for k, v in allocation.get("environment", [])),
        )

    def events(self, handle: ExecutionHandle, *, after: int | None = None) -> Iterable[ExecutorEvent]:
        payload = _handle_payload(handle)
        payload["after"] = after
        value = self.transport.call("events", payload)
        return tuple(ExecutorEvent(int(item["sequence"]), str(item["kind"]), dict(item.get("payload") or {})) for item in value.get("events", []))


def _handle_payload(handle: ExecutionHandle) -> dict[str, object]:
    return {"native_id": handle.native_id, "execution_id": handle.execution_id}


class FakeRemoteTransport:
    """In-process remote protocol mock used to prove adapter semantics."""

    def __init__(self, executor) -> None:
        self.executor = executor
        self.requests: list[tuple[str, dict[str, object]]] = []

    def call(self, action: str, payload: dict[str, object]) -> dict[str, object]:
        self.requests.append((action, payload))
        if action == "submit":
            request = request_from_dict(dict(payload["request"]))
            handle = self.executor.submit(request)
            return {"native_id": handle.native_id}
        if action == "correlate":
            return {"native_ids": [handle.native_id for handle in self.executor.correlate(str(payload["execution_id"]))]}
        handle = ExecutionHandle(self.executor.name, str(payload["native_id"]), str(payload["execution_id"]))
        if action == "status":
            status = self.executor.status(handle)
            return {"state": status.state.value, "reason": status.reason, "native_state": status.native_state}
        if action == "cancel":
            self.executor.cancel(handle); return {"ok": True}
        if action == "control":
            self.executor.control(handle, ExecutorControl(str(payload["action"]))); return {"ok": True}
        if action == "allocation":
            allocation = self.executor.allocation(handle)
            return {"allocation": None if allocation is None else dataclasses.asdict(allocation)}
        if action == "events":
            return {"events": [dataclasses.asdict(event) for event in self.executor.events(handle, after=payload.get("after"))]}
        if action == "capabilities":
            return {"executor": self.executor.name}
        raise ValueError(action)
