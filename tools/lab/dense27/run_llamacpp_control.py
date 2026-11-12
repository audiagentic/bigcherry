"""Lease-gated, isolated llama.cpp 27B Q4 placement control.

This diagnostic is not current-pin qualification. No production endpoint,
settings or files are modified. All spawned processes are explicitly tracked.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from bench_http import run as run_http

TOPOLOGIES = {
    "xtx0": ("0", "none", None),
    "r9700": ("2", "none", None),
    "xtx_layer2": ("0,1", "layer", "1,1"),
    "xtx_tp2": ("0,1", "tensor", "1,1"),
    "layer3_equal": ("0,1,2", "layer", "1,1,1"),
    "layer3_r2": ("0,1,2", "layer", "1,1,2"),
    "layer3_r3": ("0,1,2", "layer", "1,1,3"),
}


def check_lease(*, status: str) -> None:
    if not re.search(r'gpu: held by .+ for "HD2701\b', status):
        raise ValueError("HD2701 exclusive GPU lease is not held")
    if "no leases" in status:
        raise ValueError("no GPU lease")


def command_for(*, binary: Path, model: Path, port: int, topology: str) -> tuple[list[str], dict]:
    if topology not in TOPOLOGIES:
        raise ValueError(f"unsupported topology {topology}")
    if not 18400 <= port <= 18419:
        raise ValueError("port is not in the designated lab range")
    gpu_ids, split_mode, ratio = TOPOLOGIES[topology]
    argv = [
        str(binary), "-m", str(model),
        "-sm", split_mode, "-ngl", "99", "-c", "4096", "-np", "1",
        "-fa", "on", "--fit", "off", "--host", "127.0.0.1",
        "--port", str(port), "-t", "8", "-tb", "8",
    ]
    if ratio:
        argv.extend(["-ts", ratio])
    env = {"HIP_VISIBLE_DEVICES": gpu_ids, "ROCR_VISIBLE_DEVICES": gpu_ids}
    return argv, env


def _stop_own(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def _used_vram() -> dict[str, int]:
    proc = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram"],
        capture_output=True, text=True, check=False, timeout=8,
    )
    if proc.returncode:
        raise RuntimeError(f"rocm-smi failed: {proc.stderr[:250]}")
    return {
        device: int(number)
        for device, number in re.findall(
            r"GPU\[(\d+)\]\s*: VRAM Total Used Memory \(B\): (\d+)", proc.stdout
        )
    }


def _health(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, URLError, TimeoutError):
        return False


def run_arm(*, binary: Path, model: Path, outdir: Path,
            topology: str, port: int, rounds: int, prompt_repeats: int, rocm_lib: Path) -> dict:
    lease = subprocess.run(
        ["agent-lease", "status"], capture_output=True, text=True,
        check=True, timeout=8,
    )
    check_lease(status=lease.stdout)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("binary is absent or not executable")
    if not model.is_file():
        raise ValueError("model file missing")
    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / topology
    log_path = base.with_suffix(".log")
    report_path = base.with_suffix(".json")
    if log_path.exists() or report_path.exists():
        raise ValueError("refusing to overwrite existing run evidence")
    argv, env_params = command_for(binary=binary, model=model, port=port, topology=topology)
    env = os.environ.copy()
    env.update(env_params)
    if not rocm_lib.is_dir():
        raise ValueError("ROCm library directory does not exist")
    env["LD_LIBRARY_PATH"] = str(rocm_lib) + ":" + env.get("LD_LIBRARY_PATH", "")
    report = {
        "topology": topology, "utc_start": datetime.now(UTC).isoformat(),
        "model_filename": model.name, "model_bytes": model.stat().st_size,
        "binary": str(binary), "binary_revision_class": "older diagnostic, NOT current pin",
        "command": argv, "visible_devices": env_params["HIP_VISIBLE_DEVICES"],
        "status": "not_started",
    }
    proc = None
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            proc = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT,
                env=env, start_new_session=True,
            )
            report["pid"] = proc.pid
            report["status"] = "loading"
            deadline = time.monotonic() + 55
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited during load with {proc.returncode}")
                if _health(port):
                    break
                time.sleep(0.3)
            else:
                raise TimeoutError("server did not become healthy")
            report["vram_used_bytes"] = _used_vram()
            # Existing stable workload; completion hashes are diagnostic only.
            report["workload"] = run_http(
                port=port, prompt_repeats=prompt_repeats,
                max_tokens=64, concurrency=1, rounds=rounds, timeout=90,
            )
            report["status"] = "complete"
        except (OSError, RuntimeError, ValueError, KeyError, TimeoutError) as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if proc is not None:
                _stop_own(proc)
                report["exit_code"] = proc.returncode
            report["utc_end"] = datetime.now(UTC).isoformat()
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--topology", required=True, choices=list(TOPOLOGIES))
    parser.add_argument("--port", type=int, default=18408)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--rocm-lib", required=True, type=Path)
    parser.add_argument("--prompt-repeats", type=int, default=100)
    args = parser.parse_args()
    try:
        report = run_arm(**vars(args))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: report.get(key) for key in (
        "topology", "status", "error", "vram_used_bytes"
    )}, indent=2))
    if report["status"] == "complete":
        print(json.dumps({key: report["workload"][key] for key in (
            "aggregate_generated_tokens_per_s", "p50_request_s",
            "p95_request_s",
        )}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
