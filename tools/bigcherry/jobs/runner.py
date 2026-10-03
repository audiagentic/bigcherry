"""Attempt-local runner used by Slurm/Local executors."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

from ..core.host_lock import HostFileLock
from ..hardware.inventory import InventoryCatalog, bind_gpu_requirement, HardwareBindingError, verify_series_allocation
from ..hardware.model import binding_from_mapping
from ..hardware.runtime import (
    allocation_from_visible_tokens,
    selected_launch_ordinals,
    selected_local_positions,
)
from ..tuning.journal import atomic_write
from .monitor import (
    MonitorError,
    MonitorPolicy,
    disk_guards_from_environment,
    run_monitored,
)


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        path,
        json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )


def _pairs(value) -> list[tuple[str, str]]:
    return [(str(item[0]), str(item[1])) for item in value or []]


def campaign_argv(
    job: dict[str, object],
    *,
    attempt_root: Path,
    shared_root: Path,
    local_device_indices: tuple[int, ...] | None = None,
) -> tuple[str, ...]:
    arch = str(job["architecture"])
    gpu = job.get("gpu") or {}
    count = int(gpu.get("count", 1)) if isinstance(gpu, dict) else 1
    indices = local_device_indices if local_device_indices is not None else tuple(range(count))
    if len(indices) != count or len(set(indices)) != len(indices):
        raise RuntimeError(
            f"resolved device-map positions {indices} do not match requested GPU count {count}"
        )
    if any(index < 0 for index in indices):
        raise RuntimeError("device-map positions must be non-negative")
    local_devices = ",".join(str(index) for index in indices)
    toolchain = str(job.get("hip_path") or "default")
    import hashlib

    toolchain_key = hashlib.sha256(toolchain.encode()).hexdigest()[:12]
    argv: list[str] = [
        sys.executable,
        "-m",
        "bigcherry.patch.validation_campaign",
        "--patch",
        str(job["patch"]),
        "--baseline-source",
        str(job.get("baseline_source") or "bigcherry-tuning"),
        "--amdgpu-targets",
        arch,
        "--device-map",
        f"{arch}={local_devices}",
        "--model",
        str(job["model"]),
        "--workdir",
        str(attempt_root / "campaign"),
        "--worktree-root",
        str(shared_root / "campaign-worktrees"),
        "--build-root",
        str(shared_root / "builds" / f"{arch}-{toolchain_key}"),
    ]
    if job.get("hip_path"):
        argv.extend(("--hip-path", str(job["hip_path"])))
    if job.get("producer"):
        argv.extend(("--validation-producer", str(job["producer"])))
    common = job.get("common_patches") or []
    if common:
        argv.extend(("--common-patches", ",".join(str(item) for item in common)))
    for key, value in _pairs(job.get("producer_inputs")):
        argv.extend(("--producer-input", f"{key}={value}"))
    if job.get("producer_corpus"):
        argv.extend(("--producer-corpus", str(job["producer_corpus"])))
    if job.get("production_lane"):
        argv.append("--production-lane")
    return tuple(argv)


@contextlib.contextmanager
def _device_locks() -> Iterator[None]:
    if os.environ.get("BIGCHERRY_RESOURCE_POLICY", "external") != "local":
        yield
        return
    raw = os.environ.get("BIGCHERRY_SELECTED_DEVICE_IDS", "")
    ids = tuple(sorted(item for item in raw.split(",") if item))
    jobs_root_raw = os.environ.get("BIGCHERRY_JOBS_ROOT")
    if ids and not jobs_root_raw:
        raise RuntimeError("local GPU locking requires BIGCHERRY_JOBS_ROOT")
    lock_root = Path(jobs_root_raw).resolve() / "device-locks" if jobs_root_raw else None
    locks = (
        [HostFileLock(lock_root / f"{_safe(item)}.lock") for item in ids]
        if lock_root is not None
        else []
    )
    try:
        for lock in locks:
            lock.acquire()
        yield
    finally:
        for lock in reversed(locks):
            lock.release()


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value)


def _runtime_gpu_preflight(
    attempt: dict[str, object],
    *,
    shared_root: Path,
    env: dict[str, str],
) -> tuple[tuple[int, ...], dict[str, object]]:
    binding_value = attempt.get("binding")
    if not isinstance(binding_value, dict):
        raise HardwareBindingError("attempt has no persisted GPU binding")
    binding = binding_from_mapping(binding_value)
    job_value = attempt.get("job")
    target_value = job_value.get("target") if isinstance(job_value, dict) else None
    target_executor = (
        target_value.get("executor_id") if isinstance(target_value, dict) else None
    )
    executor_id = str(
        attempt.get("executor_id")
        or env.get("BIGCHERRY_EXECUTOR_ID")
        or target_executor
        or ""
    )
    if not executor_id:
        raise HardwareBindingError("executor_id is unavailable for hardware attestation")
    hardware_root = Path(
        env.get("BIGCHERRY_HARDWARE_ROOT", str(shared_root / "hardware"))
    ).resolve()
    inventory = InventoryCatalog(hardware_root).load(executor_id)
    policy = env.get("BIGCHERRY_RESOURCE_POLICY", "external")

    if policy == "external":
        raw_visible = env.get("ROCR_VISIBLE_DEVICES", "")
        allocation = allocation_from_visible_tokens(raw_visible, inventory)
        verify_series_allocation(binding, allocation, inventory)
        positions = selected_local_positions(binding, allocation)
        return positions, {
            "policy": "external",
            "executor_id": executor_id,
            "native_gpu_ids": allocation.native_gpu_ids,
            "stable_gpu_ids": allocation.stable_gpu_ids,
            "selected_local_positions": positions,
            "accepted_inventory_hash": inventory.material_hash,
        }

    if policy != "local":
        raise HardwareBindingError(f"unknown BIGCHERRY_RESOURCE_POLICY={policy!r}")
    if inventory.material_hash != binding.accepted_inventory_hash:
        raise HardwareBindingError("accepted inventory changed after series binding")
    current = bind_gpu_requirement(binding.requirement, inventory)
    if (
        current.selected_device_ids != binding.selected_device_ids
        or current.hardware_cohort_hash != binding.hardware_cohort_hash
    ):
        raise HardwareBindingError("local hardware cohort drifted")
    ordinals = selected_launch_ordinals(binding, inventory)
    platform = inventory.platform_family.lower()
    if platform.startswith("windows"):
        env["HIP_VISIBLE_DEVICES"] = ",".join(str(item) for item in ordinals)
        visibility_key = "HIP_VISIBLE_DEVICES"
    else:
        env["ROCR_VISIBLE_DEVICES"] = ",".join(str(item) for item in ordinals)
        visibility_key = "ROCR_VISIBLE_DEVICES"
    positions = tuple(range(len(binding.selected_device_ids)))
    return positions, {
        "policy": "local",
        "executor_id": executor_id,
        "stable_gpu_ids": binding.selected_device_ids,
        "launch_ordinals": ordinals,
        "selected_local_positions": positions,
        "visibility_key": visibility_key,
        "accepted_inventory_hash": inventory.material_hash,
    }


def _failure_class(returncode: int) -> str | None:
    if returncode == 0:
        return None
    if returncode == 75:
        return "transient_environment"
    if returncode == 76:
        return "harness_error"
    if returncode == 77:
        return "invalid_contract"
    if returncode in {130, 143}:
        return "cancelled_or_interrupted"
    return "execution_error"


def run_attempt(attempt_root: Path) -> int:
    attempt = json.loads((attempt_root / "attempt.json").read_text(encoding="utf-8"))
    if not isinstance(attempt, dict) or not isinstance(attempt.get("job"), dict):
        raise RuntimeError("attempt.json missing job")
    shared_root = Path(str(attempt["shared_root"])).resolve()
    env = os.environ.copy()
    project_root = Path(str(attempt["project_root"])).resolve()
    tools = str(project_root / "tools")
    env["PYTHONPATH"] = tools + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["BIGCHERRY_PROJECT_ROOT"] = str(project_root)
    env["BIGCHERRY_WORK_ROOT"] = str(shared_root)
    if attempt["job"].get("hip_path"):
        hip = str(attempt["job"]["hip_path"])
        env["HIP_PATH"] = hip
        env["ROCM_PATH"] = hip
        env["PATH"] = str(Path(hip) / "bin") + os.pathsep + env.get("PATH", "")

    start = {
        "execution_id": attempt["execution_id"],
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_ns": time.time_ns(),
        "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
    }
    _write(attempt_root / "executor-start.json", start)
    started = time.monotonic_ns()
    incident: str | None = None
    error: str | None = None
    try:
        positions, attestation = _runtime_gpu_preflight(
            attempt, shared_root=shared_root, env=env
        )
        _write(attempt_root / "allocation-attestation.json", attestation)
        argv = campaign_argv(
            attempt["job"],
            attempt_root=attempt_root,
            shared_root=shared_root,
            local_device_indices=positions,
        )
        guards = disk_guards_from_environment(
            project_root=project_root,
            work_root=shared_root,
            environment=env,
        )
        policy = MonitorPolicy.from_environment(env)
        with _device_locks():
            monitored = run_monitored(
                argv,
                environment=env,
                guards=guards,
                progress_paths=(attempt_root,),
                policy=policy,
            )
        returncode = monitored.returncode
        incident = monitored.incident
        error = monitored.detail
    except HardwareBindingError as exc:
        print(f"hardware allocation preflight failed: {exc}", file=sys.stderr)
        returncode = 75
        incident = "hardware_attestation"
        error = str(exc)
    except MonitorError as exc:
        print(f"monitor configuration failed: {exc}", file=sys.stderr)
        returncode = 76
        incident = "monitor_configuration"
        error = str(exc)
    except ValueError as exc:
        print(f"runner configuration failed: {exc}", file=sys.stderr)
        returncode = 76
        incident = "runner_configuration"
        error = str(exc)

    result: dict[str, object] = {
        "execution_id": attempt["execution_id"],
        "returncode": returncode,
        "failure_class": _failure_class(returncode),
        "finished_ns": time.time_ns(),
        "duration_ns": time.monotonic_ns() - started,
    }
    if incident is not None:
        result["incident"] = incident
    if error is not None:
        result["error"] = error
    _write(attempt_root / "executor-result.json", result)
    return returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry jobs-runner")
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args(argv)
    root = args.attempt_root.resolve()
    if args.print_command:
        attempt = json.loads((root / "attempt.json").read_text(encoding="utf-8"))
        print(
            json.dumps(
                list(
                    campaign_argv(
                        attempt["job"],
                        attempt_root=root,
                        shared_root=Path(attempt["shared_root"]),
                    )
                )
            )
        )
        return 0
    return run_attempt(root)


if __name__ == "__main__":
    raise SystemExit(main())
