"""Attempt-local process health monitoring.

Monitoring is infrastructure policy only. It never inspects benchmark results
or decides whether a scientific effect is surprising. Disk pressure or a
predeclared all-channel stall terminates the process tree and classifies the
attempt as retryable environment/harness work, never scientific FAIL.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


class MonitorError(RuntimeError):
    pass


_GIB = 1024**3


@dataclass(frozen=True)
class DiskGuard:
    name: str
    path: Path
    min_free_bytes: int
    min_free_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.min_free_bytes < 0:
            raise ValueError("min_free_bytes cannot be negative")
        if not 0.0 <= self.min_free_fraction < 1.0:
            raise ValueError("min_free_fraction must be in [0, 1)")


@dataclass(frozen=True)
class MonitorPolicy:
    poll_seconds: float = 15.0
    stall_seconds: float = 0.0
    terminate_grace_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if self.stall_seconds < 0:
            raise ValueError("stall_seconds cannot be negative")
        if self.terminate_grace_seconds < 0:
            raise ValueError("terminate_grace_seconds cannot be negative")

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> "MonitorPolicy":
        return cls(
            poll_seconds=float(environment.get("BIGCHERRY_MONITOR_POLL_SECONDS", "15")),
            stall_seconds=float(environment.get("BIGCHERRY_STALL_SECONDS", "0")),
            terminate_grace_seconds=float(environment.get("BIGCHERRY_TERM_GRACE_SECONDS", "5")),
        )


@dataclass(frozen=True)
class ProgressSample:
    monotonic: float
    cpu_ticks: int | None
    output_bytes: int


@dataclass(frozen=True)
class MonitorResult:
    returncode: int
    incident: str | None = None
    detail: str | None = None


class StallDetector:
    """Direction/result-blind liveness detector over process/output progress."""

    def __init__(self, stall_seconds: float) -> None:
        if stall_seconds < 0:
            raise ValueError("stall_seconds cannot be negative")
        self.stall_seconds = stall_seconds
        self._previous: ProgressSample | None = None
        self._last_progress: float | None = None

    def observe(self, sample: ProgressSample) -> bool:
        """Return True only when no observed progress exceeded the fixed age."""
        if self.stall_seconds <= 0:
            self._previous = sample
            self._last_progress = sample.monotonic
            return False
        if self._previous is None:
            self._previous = sample
            self._last_progress = sample.monotonic
            return False
        cpu_progress = (
            sample.cpu_ticks is not None
            and self._previous.cpu_ticks is not None
            and sample.cpu_ticks > self._previous.cpu_ticks
        )
        output_progress = sample.output_bytes > self._previous.output_bytes
        if cpu_progress or output_progress:
            self._last_progress = sample.monotonic
        self._previous = sample
        assert self._last_progress is not None
        return sample.monotonic - self._last_progress >= self.stall_seconds


def _float_env(environment: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(environment.get(key, str(default)))
    except ValueError as exc:
        raise MonitorError(f"{key} must be numeric") from exc


def disk_guards_from_environment(
    *,
    project_root: Path,
    work_root: Path,
    environment: Mapping[str, str],
) -> tuple[DiskGuard, ...]:
    """Build the two host guards used by managed campaigns.

    Defaults encode the real lab failure mode (compiler instability on an
    almost-full root filesystem) while remaining overridable per host. Setting
    either GiB/fraction value to zero disables that component of the floor.
    """
    root_gib = _float_env(environment, "BIGCHERRY_ROOT_MIN_FREE_GIB", 50.0)
    work_gib = _float_env(environment, "BIGCHERRY_WORK_MIN_FREE_GIB", 200.0)
    root_fraction = _float_env(
        environment, "BIGCHERRY_ROOT_MIN_FREE_FRACTION", 0.05
    )
    work_fraction = _float_env(
        environment, "BIGCHERRY_WORK_MIN_FREE_FRACTION", 0.02
    )
    if min(root_gib, work_gib, root_fraction, work_fraction) < 0:
        raise MonitorError("disk guard floors cannot be negative")
    return (
        DiskGuard(
            "project-root",
            project_root,
            int(root_gib * _GIB),
            root_fraction,
        ),
        DiskGuard(
            "work-root",
            work_root,
            int(work_gib * _GIB),
            work_fraction,
        ),
    )


def check_disk_guards(guards: Iterable[DiskGuard]) -> None:
    for guard in guards:
        if guard.min_free_bytes <= 0 and guard.min_free_fraction <= 0:
            continue
        try:
            usage = shutil.disk_usage(guard.path)
        except OSError as exc:
            raise MonitorError(
                f"disk guard {guard.name} cannot inspect {guard.path}: {exc}"
            ) from exc
        failures: list[str] = []
        if usage.free < guard.min_free_bytes:
            failures.append(f"free_bytes={usage.free} < {guard.min_free_bytes}")
        free_fraction = usage.free / usage.total if usage.total > 0 else 0.0
        if free_fraction < guard.min_free_fraction:
            failures.append(
                f"free_fraction={free_fraction:.4f} < {guard.min_free_fraction:.4f}"
            )
        if failures:
            raise MonitorError(
                f"disk guard {guard.name} below hard free-space floor at "
                f"{guard.path}: {'; '.join(failures)}"
            )


def _tree_cpu_ticks_linux(root_pid: int, *, proc_root: Path = Path("/proc")) -> int | None:
    """Return utime+stime for a Linux process tree; None when unavailable."""
    if os.name == "nt" or not proc_root.is_dir():
        return None
    rows: dict[int, tuple[int, int]] = {}
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="ascii")
            # comm is parenthesized and may contain spaces/parentheses. Fields
            # after the final ')' start with state, ppid ... utime, stime.
            tail = raw[raw.rfind(")") + 2 :].split()
            pid = int(entry.name)
            ppid = int(tail[1])
            utime = int(tail[11])
            stime = int(tail[12])
            rows[pid] = (ppid, utime + stime)
        except (OSError, ValueError, IndexError):
            continue
    if root_pid not in rows:
        return None
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in rows.items():
            if pid not in descendants and ppid in descendants:
                descendants.add(pid)
                changed = True
    return sum(rows[pid][1] for pid in descendants if pid in rows)


def _path_bytes(paths: Iterable[Path]) -> int:
    total = 0
    seen: set[Path] = set()
    for raw in paths:
        path = raw.resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            if path.is_file():
                total += path.stat().st_size
            elif path.is_dir():
                for child in path.rglob("*"):
                    try:
                        if child.is_file():
                            total += child.stat().st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def progress_sample(pid: int, paths: Iterable[Path]) -> ProgressSample:
    return ProgressSample(
        monotonic=time.monotonic(),
        cpu_ticks=_tree_cpu_ticks_linux(pid),
        output_bytes=_path_bytes(paths),
    )


def terminate_process_tree(process: subprocess.Popen[object], *, grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ("taskkill", "/PID", str(process.pid), "/T"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ("taskkill", "/PID", str(process.pid), "/T", "/F"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


def run_monitored(
    command: tuple[str, ...],
    *,
    environment: Mapping[str, str],
    guards: tuple[DiskGuard, ...],
    progress_paths: tuple[Path, ...],
    policy: MonitorPolicy,
) -> MonitorResult:
    """Run a child with bounded disk/stall monitoring and process-tree kill."""
    try:
        check_disk_guards(guards)
    except MonitorError as exc:
        return MonitorResult(75, "disk_pressure", str(exc))

    creationflags = 0
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(
            command,
            env=dict(environment),
            creationflags=creationflags,
            **kwargs,
        )
    except OSError as exc:
        return MonitorResult(76, "launch_error", f"process launch failed: {exc}")

    detector = StallDetector(policy.stall_seconds)
    try:
        while True:
            code = process.poll()
            if code is not None:
                return MonitorResult(int(code))
            try:
                check_disk_guards(guards)
            except MonitorError as exc:
                terminate_process_tree(
                    process, grace_seconds=policy.terminate_grace_seconds
                )
                return MonitorResult(75, "disk_pressure", str(exc))
            sample = progress_sample(process.pid, progress_paths)
            # No trustworthy CPU sampler is insufficient evidence of stall.
            if sample.cpu_ticks is not None and detector.observe(sample):
                terminate_process_tree(
                    process, grace_seconds=policy.terminate_grace_seconds
                )
                return MonitorResult(
                    75,
                    "stalled",
                    f"no process-tree CPU or output progress for {policy.stall_seconds:.1f}s",
                )
            time.sleep(policy.poll_seconds)
    except KeyboardInterrupt:
        terminate_process_tree(process, grace_seconds=policy.terminate_grace_seconds)
        return MonitorResult(130, "interrupted", "interrupted by host")
