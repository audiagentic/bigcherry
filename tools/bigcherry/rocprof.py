"""HI117: wrap a real server/binary launch under rocprofv3 kernel tracing and
reduce its output into a per-kernel-family GPU-busy summary.

Why this exists: four independent real A/B campaigns today (HI109, HI114,
HI116) all showed real, correctness-verified, individually-measured kernel
candidate wins (up to 65%) producing zero measurable end-to-end change.
tune/candidate timing is an ISOLATED microbenchmark -- it says nothing about
what share of REAL production wall-clock time the tuned kernel actually
occupies, or whether GPU time is even the bottleneck (vs CPU scheduling,
synchronization, or simply not being on the critical path). This module
answers that with real hardware-profiler data instead of the calls x
isolated-time compositional model HI35's impact.py uses (dev-gpt-agent
review, req_b851acae89e64e78: "do not repeat HI107 as another calls x
isolated-time audit -- measure the native-vs-optimized delta inside the
real production execution itself").

Designed to be reusable across future campaigns, not a one-off HI117
script: any real server/binary launch can be wrapped via
``wrap_command()``, and any resulting kernel_trace.csv can be reduced via
``load_kernel_trace()`` / ``summarize()``.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_KERNEL_TRACE_SUFFIX = "_kernel_trace.csv"


def wrap_command(
    command: list[str],
    *,
    output_directory: Path,
    output_format: str = "csv",
) -> list[str]:
    """Prefix a real launch command with rocprofv3 kernel tracing.

    ``output_directory`` gets a ``<hostname>/<pid>_kernel_trace.csv`` (plus
    ``_agent_info.csv``) written to it once the wrapped process exits --
    rocprofv3 flushes on normal exit, so a server under this wrapper must
    still be shut down cleanly (e.g. via its own /shutdown endpoint) for the
    trace to appear; killing it drops the trace.
    """
    return [
        "rocprofv3",
        "--kernel-trace",
        "-f", output_format,
        "-d", str(output_directory),
        "--",
        *command,
    ]


@dataclass
class KernelDispatch:
    agent_id: str
    kernel_name: str
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


def find_kernel_trace(output_directory: Path) -> Path:
    """Locate the single kernel_trace.csv rocprofv3 wrote under
    ``output_directory`` (it nests one level under <hostname>/<pid>_...).
    Raises if none or more than one is found -- ambiguous input, not
    something to guess at."""
    matches = sorted(output_directory.rglob(f"*{_KERNEL_TRACE_SUFFIX}"))
    if not matches:
        raise FileNotFoundError(
            f"no *{_KERNEL_TRACE_SUFFIX} found under {output_directory} -- "
            "the profiled process may not have exited cleanly (rocprofv3 "
            "only flushes on normal exit)"
        )
    if len(matches) > 1:
        raise ValueError(
            f"expected exactly one {_KERNEL_TRACE_SUFFIX} under "
            f"{output_directory}, found {len(matches)}: {matches} -- "
            "point output_directory at a single run's own directory"
        )
    return matches[0]


def load_kernel_trace(path: Path) -> list[KernelDispatch]:
    """Parse rocprofv3's own kernel_trace.csv schema (Kind, Agent_Id,
    Queue_Id, Stream_Id, Thread_Id, Dispatch_Id, Kernel_Id, Kernel_Name,
    Correlation_Id, Start_Timestamp, End_Timestamp, ...) -- verified against
    a real trace on Brutus, 2026-08-25."""
    dispatches: list[KernelDispatch] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("Kind") != "KERNEL_DISPATCH":
                continue
            dispatches.append(KernelDispatch(
                agent_id=row["Agent_Id"],
                kernel_name=row["Kernel_Name"],
                start_ns=int(row["Start_Timestamp"]),
                end_ns=int(row["End_Timestamp"]),
            ))
    return dispatches


# Real kernel-name substrings observed in a live BigCherry/llama.cpp HIP
# trace on Brutus (2026-08-25) -- a small, explicit whitelist rather than a
# clever pattern, so an unrecognised kernel falls into "other" and is
# visible for triage rather than silently misclassified.
_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mmq", re.compile(r"\bmul_mat_q\b", re.IGNORECASE)),
    ("mmvq", re.compile(r"\bmul_mat_vec_q\b", re.IGNORECASE)),
    ("mmvf", re.compile(r"\bmul_mat_vec_f\b|\bmul_mat_vec\b(?!_q)", re.IGNORECASE)),
    ("mmf", re.compile(r"\bmul_mat_f\b", re.IGNORECASE)),
    ("blas", re.compile(r"\bgemm\b|rocblas|hipblas", re.IGNORECASE)),
    ("flash_attn", re.compile(r"flash_attn")),
    ("rope", re.compile(r"\brope")),
    ("norm", re.compile(r"rms_norm|l2_norm|group_norm|layer_norm")),
    ("quantize", re.compile(r"^quantize_|quantize_f32|quantize_q8")),
    ("dequantize", re.compile(r"dequantize")),
    ("elementwise", re.compile(
        r"k_bin_bcast|unary_op_kernel|unary_gated_op_kernel|scale_f32|concat")),
    ("rows", re.compile(r"k_get_rows|k_set_rows")),
    ("copy", re.compile(r"^cpy_|copy_scalar|__amd_rocclr_copyBuffer|__amd_rocclr_fillBuffer")),
    ("rccl", re.compile(r"^ncclDevKernel")),
    ("softmax", re.compile(r"softmax")),
    ("ssm", re.compile(r"\bssm_")),
)


def classify_kernel(kernel_name: str) -> str:
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(kernel_name):
            return family
    return "other"


@dataclass
class FamilySummary:
    family: str
    call_count: int = 0
    total_time_ns: int = 0
    kernel_names: set[str] = field(default_factory=set)

    @property
    def total_time_ms(self) -> float:
        return self.total_time_ns / 1e6


@dataclass
class AgentBusySummary:
    agent_id: str
    busy_ns: int
    span_ns: int

    @property
    def busy_ms(self) -> float:
        return self.busy_ns / 1e6

    @property
    def utilization_pct(self) -> float:
        return 100.0 * self.busy_ns / self.span_ns if self.span_ns else 0.0


def _union_busy_ns(intervals: Iterable[tuple[int, int]]) -> int:
    """Real GPU-busy time from a set of (possibly overlapping) [start, end)
    dispatch intervals on one agent -- sum of the interval UNION, not the
    naive sum of individual durations (which double-counts overlapping/
    concurrent-stream dispatches on the same agent)."""
    ordered = sorted(intervals)
    if not ordered:
        return 0
    busy = 0
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            busy += cur_end - cur_start
            cur_start, cur_end = start, end
    busy += cur_end - cur_start
    return busy


def summarize(dispatches: list[KernelDispatch]) -> tuple[
    dict[str, FamilySummary], dict[str, AgentBusySummary]
]:
    """Reduce a real kernel trace into (per-family time/count, per-agent
    real GPU-busy time via interval union). Both are needed together: the
    per-family breakdown answers "how much time did the tuned kernel take",
    the per-agent busy/utilization answers "was the GPU even the
    bottleneck" (per dev-gpt-agent's HI117 diagnosis fork)."""
    families: dict[str, FamilySummary] = {}
    per_agent_intervals: dict[str, list[tuple[int, int]]] = {}
    per_agent_span: dict[str, tuple[int, int]] = {}

    for d in dispatches:
        family = classify_kernel(d.kernel_name)
        summary = families.setdefault(family, FamilySummary(family=family))
        summary.call_count += 1
        summary.total_time_ns += d.duration_ns
        summary.kernel_names.add(d.kernel_name)

        per_agent_intervals.setdefault(d.agent_id, []).append((d.start_ns, d.end_ns))
        lo, hi = per_agent_span.get(d.agent_id, (d.start_ns, d.end_ns))
        per_agent_span[d.agent_id] = (min(lo, d.start_ns), max(hi, d.end_ns))

    agents: dict[str, AgentBusySummary] = {}
    for agent_id, intervals in per_agent_intervals.items():
        busy_ns = _union_busy_ns(intervals)
        lo, hi = per_agent_span[agent_id]
        agents[agent_id] = AgentBusySummary(
            agent_id=agent_id, busy_ns=busy_ns, span_ns=hi - lo)

    return families, agents


def format_summary(
    families: dict[str, FamilySummary], agents: dict[str, AgentBusySummary],
) -> str:
    lines = ["=== per-GPU busy time (real interval union, not naive sum) ==="]
    for agent_id, s in sorted(agents.items()):
        lines.append(
            f"  {agent_id}: busy={s.busy_ms:.2f}ms span={s.span_ns/1e6:.2f}ms "
            f"util={s.utilization_pct:.1f}%")
    lines.append("")
    lines.append("=== per-family kernel time ===")
    for family, s in sorted(families.items(), key=lambda kv: -kv[1].total_time_ns):
        lines.append(
            f"  {family:14s} calls={s.call_count:7d} "
            f"total={s.total_time_ms:9.3f}ms "
            f"kernels={len(s.kernel_names)}")
    return "\n".join(lines)
