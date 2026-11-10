"""RSC11 S6: fail-closed, fakeable owner of one serving session.

The serving *plan* is provided by a separate step: future S11 supplies a
measured, validated manifest and command specs. This module does not resolve a
second layout, execute the old shell launcher, or infer a power setting.
S12 will wire the live plan, readiness/canary and power policy into this owner.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..engine_adapters.radiance_inventory import Layout
from ..hardware import lease
from ..hardware.gpu_power import Setting
from ..hardware.power_receipt import read_json, write_json_atomic
from ..hardware.power_transaction import PowerTransaction

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
_UNRESOLVED = {"applying", "applied", "rollback_failed", "restore_failed"}


def valid_run_id(value: str) -> bool:
    return bool(_RUN_ID.fullmatch(value))


class ServeRefused(RuntimeError):
    """The session cannot safely proceed or complete."""


@dataclass(frozen=True)
class ProcessSpec:
    role: str
    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ServingPlan:
    """Output of the separately injected measured-plan/validator step.

    Specs are ordered helper0..helperN, then engine; no shell invocation.
    Manifest must already be produced and validated by the planning step.
    The power map is BDF-keyed; an empty map means observe/no writes.
    """

    manifest: Path
    processes: tuple[ProcessSpec, ...]
    power: dict[str, Setting] = field(default_factory=dict)


class ProcessLauncher(Protocol):
    def start(self, spec: ProcessSpec) -> object: ...
    def ready(self, process: object, spec: ProcessSpec) -> None: ...
    def stop(self, process: object) -> None: ...
    def exited(self, process: object) -> bool: ...


class PowerOps(Protocol):
    @property
    def journal_path(self) -> Path: ...
    def stale_journals(self) -> list[Path]: ...
    def prepare(
        self, requested: dict[str, Setting], selected_bdfs: list[str]
    ) -> None: ...
    def apply(self) -> None: ...
    def restore(self, *, processes_exited: bool) -> bool: ...


class Supervisor:
    """Own process ordering, lease verification, power and a durable run record.

    The CLI must stay under a gpu lease for the entire synchronous execute()
    call. An injected wait(check_lease) keeps the session alive; it must call
    check_lease periodically and return only on a stop request. Once S11/S12
    enable live starts, shutdown failure must keep the supervisor/lease alive
    until child exit can be verified; execute() currently raises a hard error
    and leaves the journal unresolved rather than making unsafe power writes.
    """

    def __init__(
        self,
        *,
        resolve: Callable[[str], Layout],
        plan: Callable[[Layout, str], ServingPlan],
        launcher: ProcessLauncher,
        record_dir: Path,
        journal_dir: Path,
        wait: Callable[[Callable[[], None]], None],
        lease_ok: Callable[[], bool] = lease.held_by_this_process,
        power_factory: Callable[[str, Callable[[], bool]], PowerOps] | None = None,
        power_recover: Callable[..., bool] = PowerTransaction.recover,
    ) -> None:
        self.resolve = resolve
        self.plan = plan
        self.launcher = launcher
        self.record_dir = Path(record_dir)
        self.journal_dir = Path(journal_dir)
        self.wait = wait
        self.lease_ok = lease_ok
        self.power_factory = power_factory or (
            lambda rid, ok: PowerTransaction(rid, self.journal_dir, lease_ok=ok)
        )
        self.power_recover = power_recover

    def _lease(self) -> None:
        if not self.lease_ok():
            raise ServeRefused("gpu lease not verified for the serving supervisor")

    def _clean_journals(self, power: PowerOps) -> None:
        stale = list(power.stale_journals())
        # PowerTransaction.stale_journals deliberately skips its own run ID.
        # A reused run ID must never make an unresolved journal invisible.
        own = power.journal_path
        if own.exists():
            try:
                if read_json(own).get("state") in _UNRESOLVED:
                    stale.append(own)
            except (OSError, ValueError, TypeError):
                stale.append(own)
        if stale:
            raise ServeRefused(
                "unresolved power journal: "
                + ", ".join(sorted({p.name for p in stale}))
                + "; recover with verified process exit before starting"
            )

    @staticmethod
    def _validate(layout: Layout, plan: ServingPlan) -> list[str]:
        selected = [*layout.engine_bdfs, *layout.helper_bdfs]
        if len(set(selected)) != len(selected) or not layout.engine_bdfs:
            raise ServeRefused("resolver returned duplicate or empty serving BDFs")
        if not plan.manifest.is_file():
            raise ServeRefused("planner did not produce a serving manifest")
        want = tuple(f"helper{i}" for i in range(len(layout.helper_bdfs))) + ("engine",)
        if tuple(s.role for s in plan.processes) != want:
            raise ServeRefused(f"planner process order must be {want}")
        if any(not s.argv or not s.argv[0] for s in plan.processes):
            raise ServeRefused("planner supplied an empty process command")
        if set(plan.power) - set(selected):
            raise ServeRefused(
                "planner power settings include a card outside the resolved layout"
            )
        return selected

    def execute(self, *, name: str, run_id: str) -> dict:
        if not valid_run_id(run_id):
            raise ServeRefused(
                "invalid run ID (letters, digits, underscore, dash only)"
            )
        path = self.record_dir / f"{run_id}.json"
        if path.exists():
            raise ServeRefused(f"run record already exists: {run_id}")
        started: list[tuple[ProcessSpec, object]] = []
        power: PowerOps | None = None
        applied = False
        failures: list[str] = []
        record: dict = {
            "schema_version": 1,
            "run_id": run_id,
            "layout": name,
            "state": "created",
            "events": [],
            "manifest": None,
            "engine_bdfs": [],
            "helper_bdfs": [],
            "processes": [],
            "power": "not_requested",
        }

        def event(kind: str, **detail: object) -> None:
            record["events"].append({"kind": kind, **detail})
            record["state"] = kind
            write_json_atomic(path, record)

        try:
            self._lease()
            power = self.power_factory(run_id, self.lease_ok)
            self._clean_journals(power)
            layout = self.resolve(name)
            self._lease()
            serving = self.plan(layout, run_id)
            selected = self._validate(layout, serving)
            record["manifest"] = str(serving.manifest)
            record["engine_bdfs"] = list(layout.engine_bdfs)
            record["helper_bdfs"] = list(layout.helper_bdfs)
            event("planned")
            if serving.power:
                self._lease()
                power.prepare(serving.power, selected)
                self._lease()
                power.apply()
                applied = True
                record["power"] = "applied"
                event("power_applied")
            for spec in serving.processes:
                self._lease()
                process = self.launcher.start(spec)
                started.append((spec, process))
                record["processes"].append(
                    {"role": spec.role, "pid": getattr(process, "pid", None)}
                )
                event("started", role=spec.role)
                self._lease()
                self.launcher.ready(process, spec)
                event("ready", role=spec.role)
            event("running")
            self.wait(self._lease)
        except KeyboardInterrupt:
            event("stop_requested")
        except Exception as exc:  # noqa: BLE001 - fault-injection boundary must unwind any failure  # unwind even if plan, readiness, or record persistence failed
            failures.append(f"{type(exc).__name__}: {exc}")
        finally:
            # Best-effort stop of EVERY child; failure in one must not hide others.
            if started:
                try:
                    event("stopping")
                except Exception as exc:  # noqa: BLE001 - fault-injection boundary must unwind any failure
                    failures.append(f"record stopping: {exc}")
                for spec, process in reversed(started):
                    try:
                        self.launcher.stop(process)
                    except Exception as exc:  # noqa: BLE001 - fault-injection boundary must unwind any failure
                        failures.append(f"stop {spec.role}: {exc}")
                exited = True
                for spec, process in started:
                    try:
                        if not self.launcher.exited(process):
                            exited = False
                            failures.append(f"{spec.role} has not exited")
                    except Exception as exc:  # noqa: BLE001 - fault-injection boundary must unwind any failure
                        exited = False
                        failures.append(f"cannot verify {spec.role} exited: {exc}")
            else:
                exited = True
            if applied:
                if not exited:
                    failures.append(
                        "power restoration REFUSED while a serving child may be alive"
                    )
                    record["power"] = "restore_blocked_live_process"
                elif not self.lease_ok():
                    failures.append(
                        "power restoration REFUSED without a verified gpu lease"
                    )
                    record["power"] = "restore_blocked_lease"
                else:
                    try:
                        if power is None or not power.restore(processes_exited=True):
                            failures.append(
                                "power restoration could not be verified; journal retained"
                            )
                            record["power"] = "restore_failed"
                        else:
                            record["power"] = "restored"
                    except Exception as exc:  # noqa: BLE001 - fault-injection boundary must unwind any failure
                        failures.append(f"power restoration failed: {exc}")
                        record["power"] = "restore_failed"
            record["errors"] = failures
            try:
                event(
                    "exit_unverified"
                    if not exited
                    else ("failed" if failures else "stopped")
                )
            except Exception as exc:  # noqa: BLE001 - fault-injection boundary must unwind any failure
                failures.append(f"final record: {exc}")
        if failures:
            raise ServeRefused("; ".join(failures))
        return record

    def recover(
        self, journal_path: Path, *, verify_processes_exited: Callable[[], bool]
    ) -> bool:
        """Explicit crash recovery only after a trusted child-exit witness.

        A bare PID or operator assertion is not accepted here. The real S12
        supervisor must obtain the witness from its process owner.
        """
        self._lease()
        journal_path = Path(journal_path)
        if journal_path.parent.resolve() != self.journal_dir.resolve():
            raise ServeRefused(
                "recovery journal is outside the serving journal directory"
            )
        if not verify_processes_exited():
            raise ServeRefused(
                "power recovery refused: serving processes have not exited"
            )
        self._lease()
        return self.power_recover(
            journal_path, lease_ok=self.lease_ok, processes_exited=True
        )
