"""Attempt-local runner used by Slurm/Local executors."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

from ..core.host_lock import HostFileLock
from ..tuning.journal import atomic_write


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n")


def _pairs(value) -> list[tuple[str, str]]:
    return [(str(item[0]), str(item[1])) for item in value or []]


def campaign_argv(job: dict[str, object], *, attempt_root: Path, shared_root: Path) -> tuple[str, ...]:
    arch = str(job["architecture"])
    gpu = job.get("gpu") or {}
    count = int(gpu.get("count", 1)) if isinstance(gpu, dict) else 1
    # validation_campaign sees allocation-local positions only. Slurm/remote
    # visibility owns the mapping to physical devices.
    local_devices = ",".join(str(index) for index in range(count))
    toolchain = str(job.get("hip_path") or "default")
    import hashlib
    toolchain_key = hashlib.sha256(toolchain.encode()).hexdigest()[:12]
    argv: list[str] = [
        sys.executable, "-m", "bigcherry.patch.validation_campaign",
        "--patch", str(job["patch"]),
        "--baseline-source", str(job.get("baseline_source") or "bigcherry-tuning"),
        "--amdgpu-targets", arch,
        "--device-map", f"{arch}={local_devices}",
        "--model", str(job["model"]),
        "--workdir", str(attempt_root / "campaign"),
        "--worktree-root", str(shared_root / "campaign-worktrees"),
        "--build-root", str(shared_root / "builds" / f"{arch}-{toolchain_key}"),
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
def _device_locks(attempt_root: Path) -> Iterator[None]:
    if os.environ.get("BIGCHERRY_RESOURCE_POLICY", "external") != "local":
        yield
        return
    raw = os.environ.get("BIGCHERRY_SELECTED_DEVICE_IDS", "")
    ids = tuple(sorted(item for item in raw.split(",") if item))
    locks = [HostFileLock(attempt_root.parent.parent.parent.parent / "device-locks" / f"{_safe(item)}.lock") for item in ids]
    try:
        for lock in locks:
            lock.acquire()
        yield
    finally:
        for lock in reversed(locks):
            lock.release()


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value)


def run_attempt(attempt_root: Path) -> int:
    attempt = json.loads((attempt_root / "attempt.json").read_text(encoding="utf-8"))
    if not isinstance(attempt, dict) or not isinstance(attempt.get("job"), dict):
        raise RuntimeError("attempt.json missing job")
    shared_root = Path(str(attempt["shared_root"])).resolve()
    start = {
        "execution_id": attempt["execution_id"],
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_ns": time.time_ns(),
        "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
    }
    _write(attempt_root / "executor-start.json", start)
    argv = campaign_argv(attempt["job"], attempt_root=attempt_root, shared_root=shared_root)
    env = os.environ.copy()
    project_root = str(attempt["project_root"])
    tools = str(Path(project_root) / "tools")
    env["PYTHONPATH"] = tools + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["BIGCHERRY_PROJECT_ROOT"] = project_root
    env["BIGCHERRY_WORK_ROOT"] = str(shared_root)
    if attempt["job"].get("hip_path"):
        hip = str(attempt["job"]["hip_path"])
        env["HIP_PATH"] = hip
        env["ROCM_PATH"] = hip
        env["PATH"] = str(Path(hip) / "bin") + os.pathsep + env.get("PATH", "")
    started = time.monotonic_ns()
    with _device_locks(attempt_root):
        completed = subprocess.run(argv, env=env)
    result = {
        "execution_id": attempt["execution_id"],
        "returncode": completed.returncode,
        "finished_ns": time.time_ns(),
        "duration_ns": time.monotonic_ns() - started,
    }
    _write(attempt_root / "executor-result.json", result)
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry jobs-runner")
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args(argv)
    root = args.attempt_root.resolve()
    if args.print_command:
        attempt = json.loads((root / "attempt.json").read_text(encoding="utf-8"))
        print(json.dumps(list(campaign_argv(attempt["job"], attempt_root=root, shared_root=Path(attempt["shared_root"])))))
        return 0
    return run_attempt(root)


if __name__ == "__main__":
    raise SystemExit(main())
