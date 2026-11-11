"""Inactive S13 P3 production process/stop adapters for the S11/S12 supervisor.

The live CLI still refuses start/stop until the one v2 S13 activation.
This adapter never invokes the v1 shell launcher or restores GPU power itself.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..hardware.power_receipt import read_json, write_json_atomic
from .activation import require_power_authorization
from .manifest_power import read_manifest_power
from .measured_plan import MeasuredPlanValidator
from .supervisor import ProcessSpec, ServeRefused, ServingPlan, Supervisor, valid_run_id

_HELPER_READY = re.compile(r"^r3_moe_worker: (ready |waiting for engine runtime expert maps)")
_TERMINAL = frozenset(("stopped", "failed", "exit_unverified"))


@dataclass(frozen=True)
class LifecycleLimits:
    """Explicit operator policy, not guessed hidden timeouts."""

    readiness_s: float
    terminate_s: float
    poll_s: float
    stop_wait_s: float

    def __post_init__(self):
        if any(type(v) not in (float, int) or not (v > 0 and v < float("inf"))
               for v in (self.readiness_s, self.terminate_s,
                         self.poll_s, self.stop_wait_s)):
            raise ServeRefused("all lifecycle intervals must be positive finite durations")


@dataclass(frozen=True)
class StopLimits:
    """Operator-selected cross-process stop timeout, no hidden defaults."""

    stop_wait_s: float
    poll_s: float

    def __post_init__(self):
        if any(type(v) not in (float, int) or not (v > 0 and v < float("inf"))
               for v in (self.stop_wait_s, self.poll_s)):
            raise ServeRefused("stop polling and wait intervals must be positive finite durations")


@dataclass
class Child:
    role: str
    handle: object
    log_path: Path
    log_stream: object

    @property
    def pid(self):
        return getattr(self.handle, "pid", None)


class SubprocessLauncher:
    """Own real child handles; verify readiness, stop, and exit via those handles."""

    def __init__(self, *, log_dir: Path, health_url: Callable[[], str],
                 limits: LifecycleLimits, popen: Callable = subprocess.Popen,
                 health: Callable | None = None, clock: Callable = time.monotonic,
                 sleep: Callable = time.sleep):
        self.log_dir = Path(log_dir)
        self.health_url = health_url
        self.limits = limits
        self.popen = popen
        self.health = health or self._http_health
        self.clock = clock
        self.sleep = sleep

    @staticmethod
    def _http_health(url: str, timeout_s: float) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError, ValueError):
            return False

    def start(self, spec: ProcessSpec) -> Child:
        if spec.role != "engine" and not re.fullmatch(r"helper[0-9]+", spec.role):
            raise ServeRefused("unknown serving process role")
        if not spec.argv or not all(isinstance(arg, str) and arg for arg in spec.argv):
            raise ServeRefused("empty or malformed process argv")
        if any(name.startswith("R3_") for name in spec.env):
            raise ServeRefused("legacy R3_* environment is not a v2 process input")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{spec.role}.log"
        if log_path.exists():
            raise ServeRefused(f"refusing to overwrite existing child log: {log_path}")
        out = log_path.open("x", encoding="utf-8")
        # A hostile inherited R3_* value must be inert. Only typed v2
        # manifest configuration is forwarded to the child.
        environment = {k: v for k, v in os.environ.items() if not k.startswith("R3_")}
        environment.update(spec.env)
        try:
            handle = self.popen(spec.argv, env=environment, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
        except Exception:
            out.close()
            raise
        return Child(spec.role, handle, log_path, out)

    def ready(self, child: Child, spec: ProcessSpec) -> None:
        if child.role != spec.role:
            raise ServeRefused("child/process role mismatch")
        deadline = self.clock() + self.limits.readiness_s
        while True:
            if child.handle.poll() is not None:
                raise ServeRefused(f"{spec.role} exited before becoming ready")
            if spec.role == "engine":
                url = self.health_url()
                if not url.startswith("http://127.0.0.1:") or not url.endswith("/health"):
                    raise ServeRefused("engine readiness must probe local v2 HTTP endpoint")
                if self.health(url, self.limits.poll_s):
                    return
            elif child.log_path.exists():
                # The helper creates the shared segment before printing this
                # line. Match whole log lines; never accept a stale prior log.
                lines = child.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if any(_HELPER_READY.match(line) for line in lines):
                    return
            if self.clock() >= deadline:
                raise ServeRefused(f"{spec.role} readiness deadline exceeded")
            self.sleep(min(self.limits.poll_s, deadline - self.clock()))

    def stop(self, child: Child) -> None:
        if child.handle.poll() is None:
            child.handle.terminate()
            try:
                child.handle.wait(timeout=self.limits.terminate_s)
            except subprocess.TimeoutExpired:
                # Never declare shutdown on SIGTERM alone. Escalate and
                # still demand a real wait()/poll() exit witness.
                child.handle.kill()
                child.handle.wait(timeout=self.limits.terminate_s)
        if child.handle.poll() is None:
            raise ServeRefused(f"{child.role}: child exit could not be verified")
        child.log_stream.close()

    def exited(self, child: Child) -> bool:
        exited = child.handle.poll() is not None
        if exited and not child.log_stream.closed:
            child.log_stream.close()
        return exited


class StopRequestWait:
    """Cross-process stop request received by the lease-owning foreground start."""

    def __init__(self, record_dir: Path, run_id: str, *, poll_s: float,
                 sleep: Callable = time.sleep):
        if not valid_run_id(run_id) or poll_s <= 0:
            raise ServeRefused("invalid stop-wait identity or polling period")
        self.path = Path(record_dir) / f"{run_id}.stop.json"
        self.run_id = run_id
        self.poll_s = poll_s
        self.sleep = sleep

    def __call__(self, check_lease: Callable[[], None]) -> None:
        while True:
            check_lease()
            if self.path.exists():
                request = read_json(self.path)
                if request != {"schema_version": 1, "run_id": self.run_id,
                               "action": "stop"}:
                    raise ServeRefused("invalid or tampered stop request")
                return
            self.sleep(self.poll_s)


def request_stop(run_id: str, record_dir: Path, *, limits: LifecycleLimits | StopLimits,
                 clock: Callable = time.monotonic, sleep: Callable = time.sleep) -> dict:
    """Ask the owning supervisor to stop; return only after verified final state.

    Never signal another process from a persisted PID and never restore power
    from a separate CLI process.
    """
    if not valid_run_id(run_id):
        raise ServeRefused("invalid run ID")
    root = Path(record_dir)
    record_path = root / f"{run_id}.json"
    if not record_path.is_file():
        raise ServeRefused(f"no supervisor record for {run_id}")
    record = read_json(record_path)
    if record.get("run_id") != run_id:
        raise ServeRefused("supervisor run record identity mismatch")
    if record.get("state") in _TERMINAL:
        if record["state"] != "stopped" or record.get("errors"):
            raise ServeRefused(f"supervisor is not cleanly stopped: {record['state']}")
        return record
    request_path = root / f"{run_id}.stop.json"
    requested = {"schema_version": 1, "run_id": run_id, "action": "stop"}
    if request_path.exists():
        if read_json(request_path) != requested:
            raise ServeRefused("conflicting stop request")
    else:
        write_json_atomic(request_path, requested)
    deadline = clock() + limits.stop_wait_s
    while True:
        if record_path.is_file():
            record = read_json(record_path)
            if record.get("run_id") != run_id:
                raise ServeRefused("supervisor run record changed identity")
            if record.get("state") in _TERMINAL:
                if record["state"] != "stopped" or record.get("errors"):
                    raise ServeRefused(
                        f"supervisor stop is unverified: {record['state']} "
                        f"{record.get('errors', [])}")
                return record
        if clock() >= deadline:
            raise ServeRefused("stop request remains pending; no verified child exit or power restore")
        sleep(min(limits.poll_s, deadline - clock()))


def start_with_adapters(*, name: str, run_id: str, resolve: Callable,
                        source, writer, validator, record_dir: Path,
                        journal_dir: Path, lease_ok: Callable,
                        limits: LifecycleLimits, power_factory=None,
                        gfxoff_capture=None, observe=None,
                        popen: Callable = subprocess.Popen,
                        health: Callable | None = None,
                        explicit_power_profile: str | None = None,
                        qualification_gate: Callable | None = None) -> dict:
    """S13 hook assembling the actual S11 planner with the S12 supervisor.

    No second C++ resolver: S11's injected writer/engine validator own the
    strictly validated v2 manifest. Power remains observe-only unless the
    future cutover explicitly opts in after qualified S10 plan resolution.
    """
    if not valid_run_id(run_id):
        raise ServeRefused("invalid run ID")
    record_dir = Path(record_dir)
    health_target = {"url": None}
    measured = MeasuredPlanValidator(source=source, writer=writer,
                                     validator=validator, record_dir=record_dir)

    def plan(layout, requested_run_id):
        serving: ServingPlan = measured(layout, requested_run_id)
        manifest = read_json(serving.manifest)
        if manifest.get("apiVersion") != "bigcherry/v2":
            raise ServeRefused("serving start requires v2; v1 fallback refused")
        port = manifest["spec"]["processes"]["engine"]["server"]["port"]
        if type(port) is not int or not 1 <= port <= 65535:
            raise ServeRefused("v2 engine server has no valid health port")
        policy = read_manifest_power(serving.manifest,
                                     engine_bdfs=layout.engine_bdfs,
                                     helper_bdfs=layout.helper_bdfs)
        # The only power writes are those already authenticated by S10 and
        # present in the one strict v2 manifest. A selected profile is not,
        # by itself, power qualification. Observe remains write-free.
        require_power_authorization(
            policy, layout,
            explicit_profile=explicit_power_profile,
            qualification_gate=qualification_gate,
        )
        health_target["url"] = f"http://127.0.0.1:{port}/health"
        return serving

    def url():
        value = health_target["url"]
        if not isinstance(value, str):
            raise ServeRefused("engine manifest health target unavailable")
        return value

    launcher = SubprocessLauncher(log_dir=record_dir / run_id, health_url=url,
                                  limits=limits, popen=popen, health=health)
    kwargs = {}
    if power_factory is not None:
        kwargs["power_factory"] = power_factory
    if gfxoff_capture is not None:
        kwargs["gfxoff_capture"] = gfxoff_capture
    if observe is not None:
        kwargs["observe"] = observe
    supervisor = Supervisor(
        resolve=resolve, plan=plan, launcher=launcher,
        record_dir=record_dir, journal_dir=journal_dir,
        wait=StopRequestWait(record_dir, run_id, poll_s=limits.poll_s),
        lease_ok=lease_ok, **kwargs,
    )
    return supervisor.execute(name=name, run_id=run_id)
