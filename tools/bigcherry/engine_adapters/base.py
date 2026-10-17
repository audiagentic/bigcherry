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

from ..core.engines import ServeSpec

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


class BumpCheckError(RuntimeError):
    """A bump check could not run at all (as opposed to running and finding failures)."""


@dataclass(frozen=True)
class BumpCheck:
    """One engine-specific gate of ``bigcherry pin-bump``'s compile phase (MEN10 step f).

    ``run`` returns the failures it found (empty when the check holds) and raises :class:`BumpCheckError` when it could not
    run. ``tools`` are executables that must be on PATH; ``failure_code`` is the structured code a failure STOPs the bump
    with. ``patch_attributed`` asks the orchestrator to map ``file:line`` failures onto the patch edits that own those lines.
    """

    id: str
    failure_code: str
    tools: tuple[str, ...]
    install_hint: str
    run: Callable[[], list[str]]
    patch_attributed: bool = False


@dataclass(frozen=True)
class DriftCheck:
    """One engine-specific upstream-drift check of ``bigcherry pin-bump``'s drift phase (MEN19), run on the freshly pulled,
    still unpatched upstream tree.

    ``run`` returns the failures by file (empty when nothing drifted) and raises :class:`BumpCheckError` when it could not
    run. ``failure_code`` is the structured code a drift STOPs the bump with; ``fix_hint`` says what to do about it.
    """

    id: str
    failure_code: str
    fix_hint: str
    run: Callable[[], dict[str, list[str]]]


#: Variables that choose which GPUs a process sees. A CPU smoke sets every one of them to the empty string.
GPU_HIDING_ENV = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")

#: Three short fixed prompts; each is run twice at temperature 0 and the two passes must match.
SMOKE_PROMPTS = (
    "The capital of France is",
    "1, 2, 3, 4, 5,",
    "Once upon a time there was a",
)
SMOKE_TOKENS = 16
SMOKE_SEED = 1234


@dataclass(frozen=True)
class CompletionApi:
    """How an engine's server is asked for a greedy completion and where the text is in the reply."""

    path: str                              # POST route, e.g. /completion
    prompt_field: str
    token_field: str                       # request field holding the number of tokens to generate
    fixed: tuple[tuple[str, Any], ...]     # remaining request fields (temperature 0, seed, ...)
    text_path: tuple[str | int, ...]       # keys/indices from the reply document to the generated text
    models_path: str = ""                  # when set, GET it and send the first served id as "model"


@dataclass(frozen=True)
class DeviceAssertion:
    """Server-log evidence that a GPU was opened. Any line matching one of these regular expressions fails the smoke."""

    forbidden_log_patterns: tuple[str, ...]


@dataclass(frozen=True)
class CpuSmokeSpec:
    """Everything the generic runner needs to smoke one engine on the CPU with no GPU visible (MEN15)."""

    engine: str
    binary: Path
    serve: ServeSpec                       # flags, health route, shutdown method, fixed environment
    args: tuple[str, ...]                  # server arguments after the model/host/port flags
    env_overrides: dict[str, str]          # applied over the inherited environment; every GPU_HIDING_ENV name is ""
    model: Path
    model_sha256: str
    completion: CompletionApi
    device: DeviceAssertion
    prompts: tuple[str, ...] = SMOKE_PROMPTS
    n_tokens: int = SMOKE_TOKENS
    port: int | None = None                # None: the server runner picks a free port
    health_timeout_s: int = 240
    request_timeout_s: int = 300
    command_prefix: tuple[str, ...] = ()
    #: inherited variables whose name starts with one of these are removed before the server starts, so an ambient
    #: engine setting (device list, layer count, dispatch controls) cannot reach the smoke
    env_unset_prefixes: tuple[str, ...] = ()


def build_cpu_smoke(
    context: ProjectContext, build_dir: Path, model_root: Path, *, engine_args: tuple[str, ...],
    extra_env: dict[str, str], env_unset_prefixes: tuple[str, ...], completion: CompletionApi,
    device: DeviceAssertion,
) -> CpuSmokeSpec | None:
    """The spec common to every engine: the declared tiny model and extra arguments, the engine's CPU-only arguments
    and environment, and every GPU-selection variable emptied. None when the engine declares no [smoke.cpu]."""
    layout = context.layout
    declared = layout.smoke_cpu
    if declared is None:
        return None
    env = {name: "" for name in GPU_HIDING_ENV}
    env.update(extra_env)
    return CpuSmokeSpec(
        engine=layout.name, binary=Path(build_dir) / layout.serve.binary, serve=layout.serve,
        args=(*engine_args, *declared.args), env_overrides=env,
        model=Path(model_root) / declared.model, model_sha256=declared.sha256,
        completion=completion, device=device, env_unset_prefixes=env_unset_prefixes,
    )


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
    def cpu_smoke(self, context: ProjectContext, build_dir: Path, model_root: Path) -> CpuSmokeSpec | None:
        """The RAM-only smoke of this engine's build (tiny model, CPU kernels, no GPU visible), or None when the engine
        declares no ``[smoke.cpu]``. ``model_root`` is the directory the declared model path is relative to."""

    @abstractmethod
    def compile_checks(
        self, context: ProjectContext, source_dir: Path, workdir: Path, *, platform_name: str | None,
    ) -> tuple[BumpCheck, ...]:
        """The checks pin-bump's compile phase runs on the patched tree ``source_dir`` (``workdir`` is scratch space for
        them), in order; the first failing check STOPs the bump. ``platform_name`` is the recipe ``[platform.*]`` named by
        ``--platform``; a gate that builds configures exactly as ``bigcherry build`` does for that platform and raises
        ValueError when it is missing or unknown, a gate that does not build ignores it."""

    @abstractmethod
    def source_drift_checks(self, context: ProjectContext, source_dir: Path) -> tuple[DriftCheck, ...]:
        """Checks of BigCherry's own source against the freshly pulled upstream tree ``source_dir`` (MEN19), run by
        pin-bump's drift phase before any patch work; empty when the engine has nothing derived from upstream text. Each
        reports what no longer matches, per file, so one bump shows the whole port."""

    @abstractmethod
    def audit_checks(self, context: ProjectContext) -> tuple[Callable[[Any], None], ...]:
        """Source-audit checks (``bigcherry audit``, pin-bump's audit and reaudit phases) for this engine's tree. Each takes
        a ``source.audit.AuditContext`` and records results on it."""

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
