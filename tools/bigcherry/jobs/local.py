"""Durable direct-process executor for Windows, tests and emergency use."""
from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Iterable

from ..tuning.journal import atomic_write
from .executor import (
    Allocation,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionState,
    ExecutionStatus,
    ExecutorCapabilityError,
    ExecutorControl,
    ExecutorError,
    ExecutorEvent,
)


class LocalExecutor:
    """Launch a detached local process and persist enough metadata to reconnect.

    The managed runner writes ``executor-result.json`` in BIGCHERRY_ATTEMPT_ROOT;
    status uses that terminal sentinel first, then a live PID probe. Hold/release
    is deliberately unsupported because portable process suspension is not a
    safe job-service contract.
    """

    name = "local"

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        (self.root / "executions").mkdir(parents=True, exist_ok=True)

    def _path(self, execution_id: str) -> Path:
        safe = execution_id.replace("/", "_").replace("\\", "_")
        return self.root / "executions" / f"{safe}.json"

    def _read(self, execution_id: str) -> dict[str, object] | None:
        path = self._path(execution_id)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None

    def _write(self, execution_id: str, value: dict[str, object]) -> None:
        path = self._path(execution_id)
        atomic_write(
            path,
            json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n",
        )

    def submit(self, request: ExecutionRequest) -> ExecutionHandle:
        matches = self.correlate(request.execution_id)
        if matches:
            return matches[0]
        env = os.environ.copy()
        env.update(request.environment)
        stdout_path = Path(request.stdout_path)
        stderr_path = Path(request.stderr_path)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        creationflags = 0
        kwargs: dict[str, object] = {}
        if os.name == "nt":
            creationflags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
            )
        else:
            kwargs["start_new_session"] = True
        try:
            with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open(
                "ab", buffering=0
            ) as stderr:
                process = subprocess.Popen(
                    list(request.command),
                    cwd=request.cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=creationflags,
                    **kwargs,
                )
        except OSError as exc:
            raise ExecutorError(f"local process launch failed: {exc}") from exc
        selected = tuple(
            item
            for item in env.get("BIGCHERRY_SELECTED_DEVICE_IDS", "").split(",")
            if item
        )
        self._write(
            request.execution_id,
            {
                "execution_id": request.execution_id,
                "pid": process.pid,
                "attempt_root": env.get("BIGCHERRY_ATTEMPT_ROOT"),
                "stdout_path": request.stdout_path,
                "stderr_path": request.stderr_path,
                "selected_device_ids": list(selected),
            },
        )
        return ExecutionHandle(self.name, str(process.pid), request.execution_id)

    def correlate(self, execution_id: str) -> tuple[ExecutionHandle, ...]:
        record = self._read(execution_id)
        if not record:
            return ()
        pid = record.get("pid")
        if not isinstance(pid, int) or pid < 1:
            return ()
        return (ExecutionHandle(self.name, str(pid), execution_id),)

    def status(self, handle: ExecutionHandle) -> ExecutionStatus:
        self._check(handle)
        record = self._read(handle.execution_id)
        if not record:
            return ExecutionStatus(ExecutionState.UNKNOWN, "local execution metadata missing")
        attempt_root = record.get("attempt_root")
        if isinstance(attempt_root, str) and attempt_root:
            result = Path(attempt_root) / "executor-result.json"
            if result.is_file():
                try:
                    payload = json.loads(result.read_text(encoding="utf-8"))
                    code = int(payload.get("returncode", 1))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    return ExecutionStatus(
                        ExecutionState.UNKNOWN, "invalid executor-result.json"
                    )
                return ExecutionStatus(
                    ExecutionState.COMPLETED if code == 0 else ExecutionState.FAILED,
                    f"exit={code}",
                    "terminal",
                )
        alive = _pid_alive(int(handle.native_id))
        return ExecutionStatus(
            ExecutionState.RUNNING if alive else ExecutionState.UNKNOWN,
            None if alive else "process exited without terminal sentinel",
            "running" if alive else None,
        )

    def cancel(self, handle: ExecutionHandle) -> None:
        self._check(handle)
        pid = int(handle.native_id)
        try:
            if os.name == "nt":
                completed = subprocess.run(
                    ("taskkill", "/PID", str(pid), "/T", "/F"),
                    capture_output=True,
                    text=True,
                )
                if completed.returncode not in (0, 128):
                    raise ExecutorError(completed.stderr.strip() or "taskkill failed")
            else:
                os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    def control(self, handle: ExecutionHandle, action: ExecutorControl) -> None:
        self._check(handle)
        raise ExecutorCapabilityError(f"LocalExecutor does not support {action.value}")

    def allocation(self, handle: ExecutionHandle) -> Allocation | None:
        self._check(handle)
        record = self._read(handle.execution_id) or {}
        selected = record.get("selected_device_ids")
        if isinstance(selected, list):
            return Allocation(tuple(str(item) for item in selected))
        return None

    def events(
        self, handle: ExecutionHandle, *, after: int | None = None
    ) -> Iterable[ExecutorEvent]:
        self._check(handle)
        return ()

    def _check(self, handle: ExecutionHandle) -> None:
        if handle.executor != self.name or not handle.native_id.isdigit():
            raise ExecutorError(f"invalid local handle: {handle}")


def _pid_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            completed = subprocess.run(
                ("tasklist", "/FI", f"PID eq {pid}", "/NH"),
                capture_output=True,
                text=True,
            )
            return completed.returncode == 0 and str(pid) in completed.stdout
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False
