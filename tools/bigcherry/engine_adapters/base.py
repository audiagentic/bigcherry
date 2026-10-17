"""The contract between generic campaign tooling and one engine's specifics (MEN11).

Generic code (``campaign/benchmark.py`` and friends) never branches on an engine name; it asks the engine's adapter.
An adapter answers: which dispatch modes an arm may run in, what a build's compiled configuration shows, whether the
engine's server log carries a physical-device execution attestation the harness can parse, and which helper
processes (sidecars) must run next to the server for an arm, and how it measures quality against a reference.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.context import ProjectContext


class SidecarError(RuntimeError):
    """A sidecar failed to start, exited early or never reported ready."""


@dataclass(frozen=True)
class SidecarSpec:
    """One helper process that runs beside the server for an arm cell."""

    name: str
    argv: tuple[str, ...]
    env: dict[str, str]                 # the complete environment of the process (nothing is inherited)
    ready_pattern: str                  # regular expression matched (multi-line) against the sidecar log
    ready_timeout_s: float
    log_path: Path
    stop_signal: int = signal.SIGTERM
    stop_timeout_s: float = 60.0
    pre_start_cleanup: tuple[Path, ...] = ()   # removed before any sidecar of the cell starts


@dataclass(frozen=True)
class QualityCommand:
    """One offline quality-measurement process (a KL run) for an arm cell."""

    argv: tuple[str, ...]
    env: dict[str, str]                 # the complete environment of the process (nothing is inherited)
    log_path: Path
    result_path: Path | None            # the JSON document the run writes; None for a reference recording
    timeout_s: float


@dataclass(frozen=True)
class SourceCheck:
    """One engine-specific invariant ``bigcherry check`` runs for that engine; ``run`` returns None when it holds."""

    id: str
    tier: str                           # "quick", "default" or "full"
    run: Callable[[], str | None]


class EngineAdapter(ABC):
    """What a campaign needs to know about one engine."""

    #: whether the engine server log carries a device-execution attestation the harness parses
    parses_execution_attestation: bool

    @abstractmethod
    def dispatch_modes(self) -> tuple[str, ...]:
        """The dispatch modes an arm may declare."""

    @abstractmethod
    def inspect_build(self, build_dir: Path) -> dict[str, Any] | None:
        """Observation of the build compiled configuration (``issues`` and ``instrumented`` keys), or None."""

    @abstractmethod
    def check_arm_environment(self, controls: dict[str, str]) -> None:
        """Raise ValueError when an arm own environment variables are not allowed for this engine."""

    @abstractmethod
    def sidecars(self, arm: dict[str, Any], env: dict[str, str], run_dir: Path, binary: Path) -> list[SidecarSpec]:
        """Helper processes for one arm cell, in start order. ``arm`` carries ``name``, ``model`` and ``settings``;
        ``env`` is the arm complete server environment."""

    @abstractmethod
    def quality_command(
        self, kind: str, arm: dict[str, Any], env: dict[str, str], run_dir: Path, binary: Path, model: Path,
        quality: dict[str, Any],
    ) -> QualityCommand | None:
        """How this engine measures output quality against a recorded reference, or None when it has no such mode.
        ``kind`` is "record" (write the reference named by ``quality["reference_dir"]`` from ``quality["corpus"]``) or
        "score" (score the arm against it); ``env`` is the arm complete environment, so its sidecars apply."""

    @abstractmethod
    def source_checks(self, context: ProjectContext) -> tuple[SourceCheck, ...]:
        """Invariants that hold only for this engine's source tree (``bigcherry check`` runs them under the engine's
        label); empty when the engine has none."""


def _tail(path: Path, lines: int = 40) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "<no log>"
    return "\n".join(text.splitlines()[-lines:])


@dataclass
class _Running:
    spec: SidecarSpec
    process: subprocess.Popen
    log: Any
    started: float
    ready_s: float | None = None


@dataclass
class SidecarSet:
    """Starts a cell sidecars in order and stops them in reverse order. Always call :meth:`stop` in a finally."""

    specs: list[SidecarSpec]
    poll_s: float = 0.1
    _running: list[_Running] = field(default_factory=list)

    def start(self) -> None:
        for spec in self.specs:
            for path in spec.pre_start_cleanup:
                path.unlink(missing_ok=True)
        for spec in self.specs:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = spec.log_path.open("wb")
            try:
                process = subprocess.Popen(
                    list(spec.argv), env=spec.env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                )
            except OSError as exc:
                log.close()
                raise SidecarError(f"sidecar {spec.name}: cannot start {spec.argv[0]}: {exc}") from exc
            running = _Running(spec, process, log, time.monotonic())
            self._running.append(running)
            self._wait_ready(running)

    def _wait_ready(self, running: _Running) -> None:
        spec = running.spec
        pattern = re.compile(spec.ready_pattern, re.MULTILINE)
        deadline = running.started + spec.ready_timeout_s
        while True:
            if pattern.search(spec.log_path.read_text(encoding="utf-8", errors="replace")):
                running.ready_s = time.monotonic() - running.started
                return
            code = running.process.poll()
            if code is not None:
                # One last read: the ready line may have landed just before the exit.
                if pattern.search(spec.log_path.read_text(encoding="utf-8", errors="replace")):
                    running.ready_s = time.monotonic() - running.started
                    return
                raise SidecarError(f"sidecar {spec.name} exited with {code} before ready:\n{_tail(spec.log_path)}")
            if time.monotonic() >= deadline:
                raise SidecarError(
                    f"sidecar {spec.name} not ready within {spec.ready_timeout_s:g}s "
                    f"(pattern {spec.ready_pattern!r}):\n{_tail(spec.log_path)}"
                )
            time.sleep(self.poll_s)

    def stop(self) -> list[dict[str, Any]]:
        """Stop every started sidecar (reverse order): its stop signal, then a kill only after ``stop_timeout_s``."""
        records: list[dict[str, Any]] = []
        for running in reversed(self._running):
            spec, process = running.spec, running.process
            exited_before_stop = process.poll() is not None
            killed = False
            if not exited_before_stop:
                if os.name == "nt":
                    process.terminate()
                else:
                    process.send_signal(spec.stop_signal)
                try:
                    process.wait(timeout=spec.stop_timeout_s)
                except subprocess.TimeoutExpired:
                    killed = True
                    process.kill()
                    process.wait()
            running.log.close()
            records.append({
                "name": spec.name, "argv": list(spec.argv), "log": str(spec.log_path),
                "ready_s": running.ready_s, "returncode": process.returncode,
                "exited_before_stop": exited_before_stop, "killed": killed,
            })
        self._running = []
        records.reverse()
        return records
