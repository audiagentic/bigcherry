"""Slurm execution adapter for BigCherry RCD03.

This module is the only job-service layer that encodes Slurm CLI syntax. It
receives already-resolved scheduler resources; scientific GPU capability and
stable-device selection remain domain/RCD12 responsibilities.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from .executor import (
    Allocation,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionState,
    ExecutionStatus,
    ExecutorControl,
    ExecutorError,
    ExecutorEvent,
)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
    ) -> CommandResult: ...


class SubprocessRunner:
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
    ) -> CommandResult:
        completed = subprocess.run(
            list(argv), cwd=cwd, env=None if env is None else dict(env),
            text=True, capture_output=True, input=input_text,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True)
class SlurmPolicy:
    """Single-host v1 scheduler mapping proven by Noble CI.

    account=None is intentional: v1 uses AccountingStorageType=none and must
    not manufacture an association requirement.
    """

    account: str | None = None
    build_partition: str = "bc-build"
    measure_partition: str = "bc-measure"
    build_licenses: tuple[str, ...] = ("build_slot:1", "host_activity:1")
    correctness_licenses: tuple[str, ...] = ("host_activity:1",)
    measure_licenses: tuple[str, ...] = ("host_activity:2",)

    def placement(self, activity_class: str) -> tuple[str, tuple[str, ...]]:
        if activity_class == "build":
            return self.build_partition, self.build_licenses
        if activity_class == "correctness":
            return self.measure_partition, self.correctness_licenses
        if activity_class == "timed-measure":
            return self.measure_partition, self.measure_licenses
        if activity_class == "harvest":
            return self.build_partition, ()
        raise ValueError(f"unknown Slurm activity_class: {activity_class!r}")


def _slurm_time(seconds: int) -> str:
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    body = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}-{body}" if days else body


def _job_name(execution_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "-", execution_id).strip("-")
    if not safe:
        raise ValueError("execution_id cannot produce an empty Slurm job name")
    return f"bc:{safe}"[:128]


def build_sbatch_argv(
    request: ExecutionRequest,
    policy: SlurmPolicy,
    *,
    native_dependency_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    partition, licenses = policy.placement(request.resources.activity_class)
    argv: list[str] = ["sbatch", "--parsable"]
    if policy.account:
        argv.extend(("--account", policy.account))
    argv.extend((
        "--job-name", _job_name(request.execution_id),
        "--comment", f"bigcherry:{request.execution_id}",
        "--partition", partition,
        "--cpus-per-task", str(request.resources.cpu_slots),
        "--time", _slurm_time(request.resources.timeout_seconds),
        "--chdir", request.cwd,
        "--output", request.stdout_path,
        "--error", request.stderr_path,
        "--export", "ALL",
    ))
    if licenses:
        argv.extend(("--licenses", ",".join(licenses)))
    if request.resources.memory_bytes is not None:
        mib = max(1, math.ceil(request.resources.memory_bytes / (1024 * 1024)))
        argv.extend(("--mem", str(mib)))
    if request.resources.gpu is not None:
        gpu = request.resources.gpu
        argv.extend(("--gres", f"gpu:{gpu.architecture}:{gpu.reserved_count}"))
    if native_dependency_ids:
        if any(not str(item).isdigit() for item in native_dependency_ids):
            raise ValueError("Slurm dependency IDs must be numeric native job IDs")
        argv.extend(("--dependency", "afterok:" + ":".join(native_dependency_ids)))
    argv.extend(request.command)
    return tuple(argv)


_FAILURE_STATES = {
    "BOOT_FAIL", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY",
    "PREEMPTED", "TIMEOUT",
}
_CANCEL_STATES = {"CANCELLED", "CANCELLED+"}
_QUEUE_STATES = {"CONFIGURING", "PENDING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD"}
_RUNNING_STATES = {"COMPLETING", "RUNNING", "STAGE_OUT"}


def normalize_slurm_state(native_state: str, reason: str | None = None) -> ExecutionStatus:
    state = native_state.upper().strip()
    if state == "PENDING" and reason in {"JobHeldUser", "JobHeldAdmin"}:
        normalized = ExecutionState.HELD
    elif state in _QUEUE_STATES:
        normalized = ExecutionState.QUEUED
    elif state in _RUNNING_STATES:
        normalized = ExecutionState.RUNNING
    elif state == "COMPLETED":
        normalized = ExecutionState.COMPLETED
    elif state in _CANCEL_STATES or state.startswith("CANCELLED"):
        normalized = ExecutionState.CANCELLED
    elif state in _FAILURE_STATES:
        normalized = ExecutionState.FAILED
    else:
        normalized = ExecutionState.UNKNOWN
    return ExecutionStatus(normalized, reason, state)


def _state_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, str):
            return first
    if isinstance(value, dict):
        for key in ("current", "state", "name"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return "UNKNOWN"


def _squeue_jobs(payload: str) -> list[dict[str, object]]:
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ExecutorError(f"invalid squeue JSON: {exc}") from exc
    jobs = document.get("jobs")
    if not isinstance(jobs, list):
        raise ExecutorError("squeue JSON missing jobs array")
    return [job for job in jobs if isinstance(job, dict)]


def parse_squeue_json(payload: str, native_id: str) -> ExecutionStatus | None:
    for job in _squeue_jobs(payload):
        job_id = job.get("job_id", job.get("job_id_raw"))
        if str(job_id) != str(native_id):
            continue
        native = _state_text(job.get("job_state", job.get("state", "UNKNOWN")))
        reason_value = job.get("state_reason", job.get("reason"))
        reason = _state_text(reason_value) if reason_value not in (None, "") else None
        if reason == "UNKNOWN":
            reason = None
        return normalize_slurm_state(native, reason)
    return None


def parse_sacct_pipe(payload: str, native_id: str) -> ExecutionStatus | None:
    """Compatibility parser retained for old fixtures; v1 status does not call sacct."""
    for line in payload.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) < 2 or fields[0].strip() != str(native_id):
            continue
        native = fields[1].strip().split()[0]
        return normalize_slurm_state(native)
    return None


def parse_correlation(payload: str, execution_id: str) -> tuple[str, ...]:
    marker = f"bigcherry:{execution_id}"
    matches: list[str] = []
    for job in _squeue_jobs(payload):
        comment = job.get("comment", job.get("admin_comment"))
        if comment != marker:
            continue
        native = job.get("job_id", job.get("job_id_raw"))
        if str(native).isdigit():
            matches.append(str(native))
    return tuple(sorted(set(matches), key=int))


class SlurmExecutor:
    name = "slurm"

    def __init__(
        self,
        *,
        runner: CommandRunner | None = None,
        policy: SlurmPolicy | None = None,
        dependency_resolver: Callable[[str], str] | None = None,
    ) -> None:
        self.runner = runner or SubprocessRunner()
        self.policy = policy or SlurmPolicy()
        self.dependency_resolver = dependency_resolver

    def submit(self, request: ExecutionRequest) -> ExecutionHandle:
        native_dependencies: tuple[str, ...] = ()
        if request.dependencies:
            if self.dependency_resolver is None:
                raise ExecutorError("Slurm dependencies require an execution->native resolver")
            native_dependencies = tuple(self.dependency_resolver(item) for item in request.dependencies)
        argv = build_sbatch_argv(request, self.policy, native_dependency_ids=native_dependencies)
        env = os.environ.copy()
        env.update(request.environment)
        result = self.runner.run(argv, cwd=request.cwd, env=env)
        if result.returncode != 0:
            raise ExecutorError(f"sbatch failed rc={result.returncode}: {result.stderr.strip()}")
        token = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        native_id = token.split(";", 1)[0]
        if not native_id.isdigit():
            raise ExecutorError(f"sbatch --parsable returned invalid job id: {token!r}")
        return ExecutionHandle(self.name, native_id, request.execution_id)

    def correlate(self, execution_id: str) -> tuple[ExecutionHandle, ...]:
        result = self.runner.run(("squeue", "--json"))
        if result.returncode != 0:
            raise ExecutorError(f"squeue correlation failed: {result.stderr.strip()}")
        return tuple(ExecutionHandle(self.name, native, execution_id) for native in parse_correlation(result.stdout, execution_id))

    def status(self, handle: ExecutionHandle) -> ExecutionStatus:
        self._check(handle)
        queued = self.runner.run(("squeue", "--json", "--jobs", handle.native_id))
        if queued.returncode == 0:
            status = parse_squeue_json(queued.stdout, handle.native_id)
            if status is not None:
                return status
        # With AccountingStorageType=none there is deliberately no sacct
        # dependency.  BigCherry's attempt-local result/start sentinels and
        # jobcomp/filetxt reconcile terminal history in the service layer.
        return ExecutionStatus(ExecutionState.UNKNOWN, "not present in active Slurm queue", None)

    def cancel(self, handle: ExecutionHandle) -> None:
        self._check(handle)
        result = self.runner.run(("scancel", handle.native_id))
        if result.returncode != 0:
            raise ExecutorError(f"scancel failed: {result.stderr.strip()}")

    def control(self, handle: ExecutionHandle, action: ExecutorControl) -> None:
        self._check(handle)
        verb = "hold" if action == ExecutorControl.HOLD else "release"
        result = self.runner.run(("scontrol", verb, handle.native_id))
        if result.returncode != 0:
            raise ExecutorError(f"scontrol {verb} failed: {result.stderr.strip()}")

    def allocation(self, handle: ExecutionHandle) -> Allocation | None:
        self._check(handle)
        return None

    def events(
        self, handle: ExecutionHandle, *, after: int | None = None
    ) -> Iterable[ExecutorEvent]:
        self._check(handle)
        return ()

    def _check(self, handle: ExecutionHandle) -> None:
        if handle.executor != self.name or not handle.native_id.isdigit():
            raise ExecutorError(f"invalid Slurm handle: {handle}")
