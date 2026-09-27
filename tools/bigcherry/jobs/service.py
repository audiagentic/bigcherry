"""Scheduler-neutral durable job control plane."""
from __future__ import annotations

import dataclasses
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Callable, Mapping

from bigcherry.hardware.inventory import HardwareBindingError, bind_gpu_requirement
from bigcherry.hardware.model import HardwareInventory, SeriesGpuBinding

from .executor import (
    ExecutionHandle,
    ExecutionRequest,
    ExecutionState,
    Executor,
    ExecutorControl,
    ResourceRequest,
    SchedulerGpuRequest,
)
from .model import BatchSpec, JobSpec, digest
from .store import RunStore, StoreError
from .workspace import WorkspaceManager


class JobServiceError(RuntimeError):
    pass


class AmbiguousSubmission(JobServiceError):
    pass


def _binding_dict(binding: SeriesGpuBinding) -> dict[str, object]:
    return dataclasses.asdict(binding)


def _job_dict(job: JobSpec) -> dict[str, object]:
    return job.to_dict()


class JobService:
    """Application API shared by CLI, agents and future HTTP/UI adapters."""

    def __init__(
        self,
        *,
        store: RunStore,
        executors: Mapping[str, Executor],
        workspace_manager: WorkspaceManager,
        inventory_loader: Callable[[str], HardwareInventory],
        shared_root: Path,
    ) -> None:
        self.store = store
        self.executors = dict(executors)
        self.workspace_manager = workspace_manager
        self.inventory_loader = inventory_loader
        self.shared_root = shared_root.resolve()
        self.shared_root.mkdir(parents=True, exist_ok=True)

    def plan_batch(self, batch: BatchSpec) -> dict[str, object]:
        executor_id = batch.target.executor_id
        if executor_id not in self.executors:
            raise JobServiceError(f"unknown executor: {executor_id}")
        inventory = self.inventory_loader(executor_id)
        if batch.target.host_id and batch.target.host_id != inventory.host_id:
            raise JobServiceError(f"target host mismatch: requested {batch.target.host_id}, inventory is {inventory.host_id}")
        if batch.target.platform_family and batch.target.platform_family != inventory.platform_family:
            raise JobServiceError("target platform family does not match accepted inventory")
        by_arch: dict[str, dict[str, object]] = {}
        for job in batch.expand():
            if job.architecture in by_arch:
                continue
            if job.gpu is None:
                raise JobServiceError("validation batch requires an explicit GPU requirement")
            try:
                binding = bind_gpu_requirement(job.gpu, inventory)
            except HardwareBindingError as exc:
                raise JobServiceError(str(exc)) from exc
            series_material = {
                "patch": job.patch,
                "architecture": job.architecture,
                "model": job.model,
                "producer": job.producer,
                "hip_path": job.hip_path,
                "baseline_source": job.baseline_source,
                "common_patches": job.common_patches,
                "producer_inputs": job.producer_inputs,
                "producer_corpus": job.producer_corpus,
                "production_lane": job.production_lane,
                "planned_sessions": job.planned_sessions,
                "code_ref": job.code_ref,
                "executor_id": executor_id,
                "host_id": inventory.host_id,
                "platform_environment_hash": inventory.platform_environment_hash,
                "hardware_cohort_hash": binding.hardware_cohort_hash,
            }
            series_id = "s-" + digest(series_material, person=b"bc-series-v1")
            runs = [
                {
                    "run_id": "r-" + digest({"series_id": series_id, "session": session}, person=b"bc-run-v1"),
                    "session": session,
                }
                for session in range(1, job.planned_sessions + 1)
            ]
            by_arch[job.architecture] = {
                "series_id": series_id,
                "series_material": series_material,
                "binding": _binding_dict(binding),
                "accepted_inventory_hash": inventory.material_hash,
                "runs": runs,
            }
        return {
            "schema": "bigcherry.jobs.plan.v1",
            "request_hash": batch.request_hash,
            "executor_id": executor_id,
            "host_id": inventory.host_id,
            "platform_family": inventory.platform_family,
            "series": [by_arch[key] for key in sorted(by_arch)],
        }

    def submit_batch(self, batch: BatchSpec, *, idempotency_key: str, actor: str = "unknown") -> dict[str, object]:
        plan = self.plan_batch(batch)
        batch_id = "b-" + digest({"key": idempotency_key, "request_hash": batch.request_hash}, person=b"bc-batch-id")
        accepted = self.store.accept_idempotency(idempotency_key, batch.request_hash, batch_id)
        if accepted["batch_id"] != batch_id:
            return self.store.batch(str(accepted["batch_id"]))
        record = {
            "schema": "bigcherry.jobs.batch-record.v1",
            "batch_id": batch_id,
            "request": batch.to_dict(),
            "request_hash": batch.request_hash,
            "idempotency_key": idempotency_key,
            "actor": actor,
            "plan": plan,
            "created_ns": time.time_ns(),
        }
        self.store.create_batch(batch_id, record)
        jobs = {(job.architecture, job.session): job for job in batch.expand()}
        for series in plan["series"]:
            assert isinstance(series, dict)
            series_id = str(series["series_id"])
            series_record = {
                "schema": "bigcherry.jobs.series.v1",
                "series_id": series_id,
                "batch_id": batch_id,
                **series,
            }
            self.store.create_series(series_id, series_record)
            architecture = str(series["series_material"]["architecture"])
            for run in series["runs"]:
                session = int(run["session"])
                run_id = str(run["run_id"])
                job = jobs[(architecture, session)]
                run_record = {
                    "schema": "bigcherry.jobs.run.v1",
                    "run_id": run_id,
                    "series_id": series_id,
                    "batch_id": batch_id,
                    "session": session,
                    "job": _job_dict(job),
                    "binding": series["binding"],
                    "executor_id": batch.target.executor_id,
                    "created_ns": time.time_ns(),
                }
                self.store.create_run(run_id, run_record)
                self.store.enqueue(batch_id, run_id)
                self.store.append_event(kind="job.submitted", run_id=run_id, data={"batch_id": batch_id, "series_id": series_id})
        return record

    def ingest_once(self) -> dict[str, int]:
        if bool(self.store.service_control().get("paused")):
            return {"accepted": 0, "rejected": 0, "deferred": len(self.store.pending())}
        accepted = rejected = deferred = 0
        for pending in self.store.pending():
            processing = self.store.claim_pending(pending)
            receipt = json.loads(processing.read_text(encoding="ascii"))
            run_id = str(receipt["run_id"])
            control = self.store.control(run_id)
            if control.get("disabled"):
                os.replace(processing, self.store.root / "inbox" / "pending" / processing.name)
                deferred += 1
                continue
            try:
                self._start_or_recover(run_id)
            except Exception as exc:
                self.store.append_event(kind="attempt.submit-failed", run_id=run_id, severity="wake", data={"error": f"{type(exc).__name__}: {exc}"})
                self.store.finish_receipt(processing, accepted=False)
                rejected += 1
            else:
                self.store.finish_receipt(processing, accepted=True)
                accepted += 1
        return {"accepted": accepted, "rejected": rejected, "deferred": deferred}

    def _start_or_recover(self, run_id: str) -> ExecutionHandle:
        run = self.store.run(run_id)
        executor_id = str(run["executor_id"])
        executor = self.executors[executor_id]
        attempt_no = self.store.next_attempt(run_id)
        control = self.store.control(run_id)
        code_ref = str(control.get("retry_code_ref") or run["job"]["code_ref"])
        workspace = self.workspace_manager.create(run_id, attempt_no, code_ref)
        if control.get("retry_code_ref") is not None:
            self.store.set_control(run_id, retry_code_ref=None)
        execution_id = f"{run_id}:a{attempt_no}"
        attempt_root = self.store.attempt_root(run_id, attempt_no)
        attempt_record = {
            "schema": "bigcherry.jobs.attempt.v1",
            "run_id": run_id,
            "attempt": attempt_no,
            "execution_id": execution_id,
            "commit": workspace.commit,
            "project_root": str(workspace.project_root),
            "shared_root": str(self.shared_root),
            "job": run["job"],
            "binding": run["binding"],
            "created_ns": time.time_ns(),
        }
        self.store.create_attempt(run_id, attempt_no, attempt_record)
        request = self._execution_request(run, attempt_record, attempt_root, executor)
        self.store.write_submission_intent(run_id, attempt_no, {
            "execution_id": execution_id,
            "request_hash": digest(dataclasses.asdict(request), person=b"bc-exec-request"),
            "commit": workspace.commit,
        })
        start = self.store.read_optional(attempt_root / "executor-start.json")
        matches: tuple[ExecutionHandle, ...]
        if start and start.get("slurm_job_id"):
            matches = (ExecutionHandle("slurm", str(start["slurm_job_id"]), execution_id),)
        else:
            matches = executor.correlate(execution_id)
        if len(matches) > 1:
            raise AmbiguousSubmission(f"multiple native executions correlate to {execution_id}")
        handle = matches[0] if matches else executor.submit(request)
        self.store.write_submission(run_id, attempt_no, {
            "execution_id": execution_id,
            "executor_id": executor_id,
            "executor": handle.executor,
            "native_id": handle.native_id,
            "submitted_ns": time.time_ns(),
        })
        self.store.append_event(kind="attempt.submitted", run_id=run_id, data={"attempt": attempt_no, "execution_id": execution_id, "native_id": handle.native_id})
        return handle

    def _execution_request(self, run: dict[str, object], attempt: dict[str, object], attempt_root: Path, executor: Executor) -> ExecutionRequest:
        binding = run["binding"]
        assert isinstance(binding, dict)
        scheduler_gpu = SchedulerGpuRequest(str(binding["scheduler_architecture"]), int(binding["scheduler_count"]))
        env = {
            "BIGCHERRY_RUN_ID": str(run["run_id"]),
            "BIGCHERRY_ATTEMPT": str(attempt["attempt"]),
            "BIGCHERRY_ATTEMPT_ROOT": str(attempt_root),
            "BIGCHERRY_SELECTED_DEVICE_IDS": ",".join(str(item) for item in binding["selected_device_ids"]),
            "BIGCHERRY_RESOURCE_POLICY": "external" if executor.name == "slurm" else "local",
            "PYTHONPATH": str(Path(str(attempt["project_root"])) / "tools"),
        }
        if executor.name == "slurm":
            launch = attempt_root / "launch.sh"
            launch.parent.mkdir(parents=True, exist_ok=True)
            python = shlex.quote(sys.executable)
            root = shlex.quote(str(attempt_root))
            launch.write_text(f"#!/bin/bash\nset -euo pipefail\nexec {python} -m bigcherry.jobs.runner --attempt-root {root}\n", encoding="utf-8")
            launch.chmod(0o755)
            command = (str(launch),)
        else:
            command = (sys.executable, "-m", "bigcherry.jobs.runner", "--attempt-root", str(attempt_root))
        return ExecutionRequest(
            execution_id=str(attempt["execution_id"]),
            command=command,
            cwd=str(attempt["project_root"]),
            env=tuple(sorted(env.items())),
            stdout_path=str(attempt_root / "stdout.log"),
            stderr_path=str(attempt_root / "stderr.log"),
            resources=ResourceRequest(
                cpu_slots=4,
                gpu=scheduler_gpu,
                activity_class="timed-measure",
                memory_bytes=None,
                timeout_seconds=int(run["job"].get("timeout_seconds", 2700)),
            ),
        )

    def _handle_for_run(self, run_id: str) -> tuple[Executor, ExecutionHandle, int]:
        run = self.store.run(run_id)
        attempt_no = self.store.latest_attempt(run_id)
        if attempt_no is None:
            raise JobServiceError("run has no attempt")
        submission = self.store.read_optional(self.store.attempt_root(run_id, attempt_no) / "submission.json")
        if not submission:
            raise JobServiceError("attempt has no submission")
        executor = self.executors[str(run["executor_id"])]
        return executor, ExecutionHandle(str(submission["executor"]), str(submission["native_id"]), str(submission["execution_id"])), attempt_no

    def status(self, run_id: str) -> dict[str, object]:
        run = self.store.run(run_id)
        attempt_no = self.store.latest_attempt(run_id)
        control = self.store.control(run_id)
        if control.get("cancelled"):
            state = "cancelled"
        elif attempt_no is None:
            state = "disabled" if control.get("disabled") else "queued"
        else:
            executor, handle, _ = self._handle_for_run(run_id)
            terminal = self.store.read_optional(self.store.attempt_root(run_id, attempt_no) / "executor-result.json")
            if terminal:
                state = "completed" if int(terminal.get("returncode", 1)) == 0 else "failed"
            else:
                state = executor.status(handle).state.value
        return {"run_id": run_id, "series_id": run["series_id"], "session": run["session"], "state": state, "attempt": attempt_no, "disabled": bool(control.get("disabled"))}

    def list_status(self) -> tuple[dict[str, object], ...]:
        return tuple(self.status(str(run["run_id"])) for run in self.store.list_runs())

    def control_run(self, run_id: str, action: str) -> dict[str, object]:
        if action == "disable":
            self.store.set_control(run_id, disabled=True)
        elif action == "enable":
            self.store.set_control(run_id, disabled=False)
        elif action == "cancel":
            executor, handle, _ = self._handle_for_run(run_id)
            executor.cancel(handle)
            self.store.set_control(run_id, cancelled=True)
        elif action in {"hold", "release"}:
            executor, handle, _ = self._handle_for_run(run_id)
            executor.control(handle, ExecutorControl.HOLD if action == "hold" else ExecutorControl.RELEASE)
        else:
            raise ValueError(f"unknown control action: {action}")
        self.store.append_event(kind=f"job.{action}", run_id=run_id)
        return self.status(run_id)

    def pause(self, paused: bool) -> dict[str, object]:
        value = self.store.set_service_control(paused=paused)
        self.store.append_event(kind="service.paused" if paused else "service.resumed")
        return value

    def retry(self, run_id: str, *, latest: bool) -> None:
        run = self.store.run(run_id)
        if not latest:
            previous = self.store.latest_attempt(run_id)
            if previous is None:
                raise JobServiceError("cannot retry same commit before first attempt")
            commit = str(self.store.attempt(run_id, previous)["commit"])
            self.store.set_control(run_id, retry_code_ref=commit, cancelled=False)
        else:
            self.store.set_control(run_id, retry_code_ref=None, cancelled=False)
        self.store.enqueue(str(run["batch_id"]), run_id, reason="retry")
        self.store.append_event(kind="job.retry-requested", run_id=run_id, data={"latest": latest})

    def logs(self, run_id: str, *, stream: str = "stdout", offset: int = 0, limit: int = 1024 * 1024) -> dict[str, object]:
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be stdout or stderr")
        attempt = self.store.latest_attempt(run_id)
        if attempt is None:
            return {"run_id": run_id, "stream": stream, "offset": offset, "next_offset": offset, "data": "", "eof": True}
        path = self.store.attempt_root(run_id, attempt) / f"{stream}.log"
        if not path.exists():
            return {"run_id": run_id, "stream": stream, "offset": offset, "next_offset": offset, "data": "", "eof": True}
        with path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read(limit)
            next_offset = handle.tell()
            eof = next_offset >= path.stat().st_size
        return {"run_id": run_id, "stream": stream, "offset": offset, "next_offset": next_offset, "data": raw.decode("utf-8", "replace"), "eof": eof}

    def artifacts(self, run_id: str) -> tuple[dict[str, object], ...]:
        attempt = self.store.latest_attempt(run_id)
        if attempt is None:
            return ()
        root = self.store.attempt_root(run_id, attempt)
        values: list[dict[str, object]] = []
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            values.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size})
        return tuple(values)

    def review(self, series_id: str) -> dict[str, object]:
        series = self.store.series(series_id)
        runs = [run for run in self.store.list_runs() if run.get("series_id") == series_id]
        statuses = [self.status(str(run["run_id"])) for run in sorted(runs, key=lambda item: int(item["session"]))]
        planned = int(series["series_material"]["planned_sessions"])
        completed = sum(1 for status in statuses if status["state"] == "completed")
        return {"series_id": series_id, "planned_sessions": planned, "completed_sessions": completed, "missing_sessions": [s["session"] for s in statuses if s["state"] != "completed"], "review_ready": completed == planned, "runs": statuses}
