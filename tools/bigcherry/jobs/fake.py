"""Deterministic in-memory Executor for RCD04 service tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .executor import (
    Allocation,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionState,
    ExecutionStatus,
    ExecutorControl,
    ExecutorEvent,
)


@dataclass
class _FakeRun:
    request: ExecutionRequest
    handle: ExecutionHandle
    state: ExecutionState = ExecutionState.QUEUED
    reason: str | None = None
    allocation: Allocation | None = None
    events: list[ExecutorEvent] = field(default_factory=list)


class FakeExecutor:
    name = "fake"

    def __init__(self) -> None:
        self._counter = 0
        self._by_native: dict[str, _FakeRun] = {}
        self._native_by_execution: dict[str, str] = {}

    def submit(self, request: ExecutionRequest) -> ExecutionHandle:
        if request.execution_id in self._native_by_execution:
            native = self._native_by_execution[request.execution_id]
            return self._by_native[native].handle
        self._counter += 1
        native = f"fake-{self._counter:06d}"
        handle = ExecutionHandle(self.name, native, request.execution_id)
        run = _FakeRun(request, handle)
        self._by_native[native] = run
        self._native_by_execution[request.execution_id] = native
        self._emit(run, "submitted", {})
        return handle

    def status(self, handle: ExecutionHandle) -> ExecutionStatus:
        run = self._run(handle)
        return ExecutionStatus(run.state, run.reason, run.state.value)

    def cancel(self, handle: ExecutionHandle) -> None:
        run = self._run(handle)
        if run.state not in {ExecutionState.COMPLETED, ExecutionState.CANCELLED, ExecutionState.FAILED}:
            run.state = ExecutionState.CANCELLED
            self._emit(run, "cancelled", {})

    def control(self, handle: ExecutionHandle, action: ExecutorControl) -> None:
        run = self._run(handle)
        if action == ExecutorControl.HOLD:
            if run.state == ExecutionState.QUEUED:
                run.state = ExecutionState.HELD
                self._emit(run, "held", {})
            return
        if action == ExecutorControl.RELEASE:
            if run.state == ExecutionState.HELD:
                run.state = ExecutionState.QUEUED
                self._emit(run, "released", {})
            return
        raise ValueError(action)

    def allocation(self, handle: ExecutionHandle) -> Allocation | None:
        return self._run(handle).allocation

    def events(
        self, handle: ExecutionHandle, *, after: int | None = None
    ) -> Iterable[ExecutorEvent]:
        events = self._run(handle).events
        minimum = 0 if after is None else after
        return tuple(event for event in events if event.sequence > minimum)

    def set_allocation(self, handle: ExecutionHandle, allocation: Allocation) -> None:
        self._run(handle).allocation = allocation

    def start_ready(self) -> tuple[ExecutionHandle, ...]:
        complete = {
            run.request.execution_id
            for run in self._by_native.values()
            if run.state == ExecutionState.COMPLETED
        }
        started: list[ExecutionHandle] = []
        for run in self._by_native.values():
            if run.state != ExecutionState.QUEUED:
                continue
            if not set(run.request.dependencies) <= complete:
                continue
            run.state = ExecutionState.RUNNING
            self._emit(run, "started", {})
            started.append(run.handle)
        return tuple(started)

    def complete(self, handle: ExecutionHandle, *, failed: bool = False) -> None:
        run = self._run(handle)
        if run.state != ExecutionState.RUNNING:
            raise RuntimeError(f"cannot complete {run.state.value} fake execution")
        run.state = ExecutionState.FAILED if failed else ExecutionState.COMPLETED
        self._emit(run, "failed" if failed else "completed", {})

    def _run(self, handle: ExecutionHandle) -> _FakeRun:
        if handle.executor != self.name or handle.native_id not in self._by_native:
            raise KeyError(handle)
        return self._by_native[handle.native_id]

    @staticmethod
    def _emit(run: _FakeRun, kind: str, payload: dict[str, object]) -> None:
        run.events.append(ExecutorEvent(len(run.events) + 1, kind, payload))
