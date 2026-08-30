"""PROF01/HI132: real rocprofv3 invocation + CSV parsing.

Real subprocess + real ROCm profiler -- there is no mock/simulation mode.
Column names and the ``<prefix>_kernel_trace.csv`` / ``*_agent_info.csv``
naming convention were confirmed against a real rocprofv3 run on Brutus
this session (RD33's kernel-resource diagnostic): Kind, Agent_Id,
Queue_Id, Stream_Id, Thread_Id, Dispatch_Id, Kernel_Id, Kernel_Name,
Correlation_Id, Start_Timestamp, End_Timestamp, LDS_Block_Size,
Scratch_Size, VGPR_Count, Accum_VGPR_Count, SGPR_Count,
Workgroup_Size_{X,Y,Z}, Grid_Size_{X,Y,Z}.
"""

from __future__ import annotations

import csv
import glob
import os
import statistics
import subprocess
from collections.abc import Mapping
from pathlib import Path

from .schema import GpuProfilePass, KernelStat


class RocprofError(RuntimeError):
    pass


_REDUCTION_PROVIDERS = frozenset(("auto", "rccl", "meta"))


def expected_reduction_provider(env: Mapping[str, str] | None = None) -> str:
    """Return the provider the profiled process is expected to use.

    ``GGML_HIP_REDUCE_PLAN`` is the existing runtime selector.  Keep this
    normalization in lockstep with the selector's implementation: unset and
    unknown values resolve to ``auto``.
    """
    value = (env if env is not None else os.environ).get("GGML_HIP_REDUCE_PLAN")
    return value if value in _REDUCTION_PROVIDERS else "auto"


def rocprofv3_command_prefix(*, output_dir: Path, label: str) -> tuple[str, ...]:
    """The exact flags used to validate this design (session
    ses_fec603fc33ee4089): --sys-trace for HIP/HSA API + kernel dispatch +
    memory-op coverage, --rccl-trace for the collective-API spans a real
    dual-GPU tensor-split run needs. Ends in "--" so ServerRunner's own
    command_prefix contract (prefix, then the real binary/args) holds.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        "rocprofv3",
        "--sys-trace",
        "--rccl-trace",
        "--output-format", "csv",
        "--output-directory", str(output_dir),
        "-o", label,
        "--",
    )


def rocprofv3_version() -> str | None:
    try:
        out = subprocess.run(
            ["rocprofv3", "--version"], capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or out.stderr or "").strip()
    return text or None


def _find_one(output_dir: Path, pattern: str) -> Path | None:
    matches = sorted(output_dir.glob(pattern))
    return matches[0] if matches else None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def parse_kernel_trace(kernel_trace_csv: Path) -> tuple[KernelStat, ...]:
    """Aggregate every real dispatch in a rocprofv3 kernel-trace CSV into
    one KernelStat per distinct kernel name -- durations, register/scratch
    footprint (constant per compiled kernel, so any one row's value is
    representative), and every GPU agent id the kernel actually ran on."""
    if not kernel_trace_csv.is_file():
        raise RocprofError(f"no kernel trace at {kernel_trace_csv}")

    by_name: dict[str, dict] = {}
    with kernel_trace_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Kernel_Name", "").strip()
            if not name:
                continue
            start = row.get("Start_Timestamp")
            end = row.get("End_Timestamp")
            if not start or not end:
                continue
            duration_us = (int(end) - int(start)) / 1000.0
            bucket = by_name.setdefault(name, {
                "durations": [], "agents": set(),
                "vgpr": row.get("VGPR_Count", "0"),
                "sgpr": row.get("SGPR_Count", "0"),
                "scratch": row.get("Scratch_Size", "0"),
            })
            bucket["durations"].append(duration_us)
            agent = row.get("Agent_Id", "")
            if agent:
                bucket["agents"].add(agent)

    stats = []
    for name, bucket in by_name.items():
        durations = bucket["durations"]
        stats.append(KernelStat(
            name=name,
            calls=len(durations),
            total_us=sum(durations),
            mean_us=statistics.fmean(durations) if durations else 0.0,
            p95_us=_percentile(durations, 0.95),
            vgpr_count=int(bucket["vgpr"] or 0),
            sgpr_count=int(bucket["sgpr"] or 0),
            scratch_size=int(bucket["scratch"] or 0),
            agent_ids=tuple(sorted(bucket["agents"])),
        ))
    return tuple(sorted(stats, key=lambda k: k.total_us, reverse=True))


# HI134: real-hardware evidence (2026-08-28, hi134-meta-baseline-01, Brutus
# {0,1,2}) found that ANY row in the RCCL trace is not evidence a real
# collective ran. ncclGetVersion/ncclGetUniqueId/ncclCommInitAll/
# ncclCommGetAsyncError/ncclCommDestroy all appear even under
# GGML_HIP_REDUCE_PLAN=meta, where every actual reduce call is diverted to
# META and RCCL never performs a collective (ncclCommInitAll succeeding is
# expected per HI85 -- the crash there is inside the collective kernel
# launch, not init). Only these function names indicate a real collective
# was attempted.
_RCCL_COLLECTIVE_FUNCTIONS = frozenset((
    "ncclAllReduce", "ncclBroadcast", "ncclReduce", "ncclAllGather",
    "ncclReduceScatter", "ncclSend", "ncclRecv", "ncclGroupEnd",
))


def _rccl_activity_seen(output_dir: Path) -> bool:
    rccl_csv = _find_one(output_dir, "*rccl*trace.csv") or _find_one(output_dir, "*rccl*.csv")
    if rccl_csv is None or not rccl_csv.is_file():
        return False
    with rccl_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "Function" not in reader.fieldnames:
            return False
        return any(row.get("Function") in _RCCL_COLLECTIVE_FUNCTIONS for row in reader)


def build_gpu_profile_pass(
    *, label: str, output_dir: Path, expected_gpu_count: int,
    expected_reduction_provider: str = "auto",
) -> GpuProfilePass:
    if expected_reduction_provider not in _REDUCTION_PROVIDERS:
        raise RocprofError(
            f"unsupported expected reduction provider {expected_reduction_provider!r}"
        )
    kernel_trace = _find_one(output_dir, "*_kernel_trace.csv") or _find_one(output_dir, "*kernel_trace*.csv")
    if kernel_trace is None:
        raise RocprofError(
            f"rocprofv3 produced no kernel_trace CSV under {output_dir} -- "
            "profiler invocation likely failed before capturing anything"
        )
    kernels = parse_kernel_trace(kernel_trace)
    agent_ids_seen = tuple(sorted({a for k in kernels for a in k.agent_ids}))
    rccl_seen = _rccl_activity_seen(output_dir)

    # PROF01: fail closed rather than silently PASS an incomplete capture --
    # gpt's own settling criterion from the design negotiation.
    capture_status = "complete"
    if expected_gpu_count > 1:
        if len(agent_ids_seen) < expected_gpu_count:
            capture_status = "incomplete_multi_gpu_capture"
        elif expected_reduction_provider in ("auto", "rccl") and not rccl_seen:
            capture_status = "incomplete_multi_gpu_capture"

    return GpuProfilePass(
        label=label,
        output_dir=str(output_dir),
        kernels=kernels,
        agent_ids_seen=agent_ids_seen,
        rccl_activity_seen=rccl_seen,
        expected_gpu_count=expected_gpu_count,
        capture_status=capture_status,
        expected_reduction_provider=expected_reduction_provider,
    )
