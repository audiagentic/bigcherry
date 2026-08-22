"""rocprofv3 kernel-trace analysis: the matmul fraction of decode time (HI35, Part 2).

rocprofv3 kernel tracing gives per-kernel durations for a real run with no code
change and no serialisation.  This module aggregates kernel time by family,
maps kernels to the tuning families, and reports both the kernel-time share and
the wall-clock ceiling (GPU busy fraction).

The fraction is the ceiling on everything the tuning items (HI12/HI24/HI34) can
do: a 10% matmul saving is at most ``matmul_fraction x 10%`` end to end, before
Amdahl's law meets the CPU-bound parts of decode.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

from .symbol_map import demangle


class KernelFractionError(ValueError):
    pass


# rocprofv3's CSV column names have moved between point releases.  Key by
# header, never by position, and report which columns were found -- a silently
# mis-parsed trace produces a plausible percentage.
_NAME_COLUMNS = ("Kernel_Name", "Kernel Name", "KernelName")
_START_COLUMNS = ("Start_Timestamp", "BeginNs", "Start")
_END_COLUMNS = ("End_Timestamp", "EndNs", "End")
_DURATION_COLUMNS = ("Duration", "DurationNs", "End_Timestamp - Start_Timestamp")
_AGENT_COLUMNS = ("Agent_Id", "Device_Id", "gpu-id")

# Matmul families, in the same vocabulary as the tuning catalog (HI35 Part 1).
MATMUL_FAMILIES = ("mmq", "mmvq", "mmvf", "mmf", "blas")

# Family attribution.  Order matters: the more specific ``mul_mat_vec_*`` and
# ``mul_mat_f`` entries come before the generic ``mul_mat_q``/``mul_mat_f``
# substrings they share a prefix with.  The quantize kernels are the part that
# is easy to get wrong: standards 7.1 times the *complete* matmul path, and
# MMQ's activation quantisation is a large fraction of its cost.  Attributing
# quantize_mmq_q8_1 to "other" would understate the matmul fraction and put
# this method into disagreement with the record-based model.
_FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mmvq", ("mul_mat_vec_q", "quantize_q8_1")),
    ("mmvf", ("mul_mat_vec_f",)),
    ("mmq", ("mul_mat_q", "quantize_mmq_q8_1")),
    ("mmf", ("mul_mat_f",)),
    ("blas", ("Cijk_", "rocblas", "hipblas", "gemm")),
    ("attention", ("flash_attn", "soft_max")),
    ("norm/rope/act", ("rms_norm", "norm_f32", "rope", "silu", "gelu", "swiglu")),
    ("copy/other", ("cpy_", "dup_", "contiguous", "get_rows", "add_f32", "mul_f32")),
)


def classify(kernel_name: str) -> str:
    """Map a (possibly mangled) kernel symbol to a tuning family."""
    for family, needles in _FAMILY_PATTERNS:
        if any(needle in kernel_name for needle in needles):
            return family
    return "unmapped"


def gpu_busy_ns(spans: list[tuple[int, int]]) -> int:
    """Union of kernel intervals, not their sum.

    Kernels from different queues (and different agents) overlap, so summing
    durations double-counts and can exceed wall time -- which then produces a
    GPU utilisation above 100% and a wall-clock ceiling that is nonsense.
    Merge first.
    """
    if not spans:
        return 0
    spans = sorted(spans)
    busy = 0
    start, end = spans[0]
    for lo, hi in spans[1:]:
        if lo > end:
            busy += end - start
            start, end = lo, hi
        else:
            end = max(end, hi)
    return busy + (end - start)


def _pick(header: list[str], candidates: tuple[str, ...]) -> str | None:
    for wanted in candidates:
        if wanted in header:
            return wanted
    return None


def _parse_agent(value: str | None) -> int | None:
    """rocprofv3 agent ids appear as a bare number (older builds) or as
    'Agent N' (newer builds); both carry the same device index."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    match = re.search(r"(\d+)\s*$", text)
    if match:
        return int(match.group(1))
    return None


def parse_kernel_trace(paths: list[Path]) -> dict[str, Any]:
    """Parse one or more rocprofv3 kernel-trace CSVs into spans + rows.

    Returns the per-kernel rows (kernel name, family, duration, agent), the
    list of (start, end) spans, the traced wall window, and the column names
    that were actually used -- so a wrong-column parse is visible in the
    report rather than silent.
    """
    spans: list[tuple[int, int]] = []
    rows: list[dict[str, Any]] = []
    columns: dict[str, str | None] = {}

    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                raise KernelFractionError(f"{path}: empty kernel-trace CSV")
            name_col = _pick(header, _NAME_COLUMNS)
            dur_col = _pick(header, _DURATION_COLUMNS)
            start_col = _pick(header, _START_COLUMNS)
            end_col = _pick(header, _END_COLUMNS)
            agent_col = _pick(header, _AGENT_COLUMNS)
            if name_col is None or (
                dur_col is None and (start_col is None or end_col is None)
            ):
                raise KernelFractionError(
                    f"{path}: no kernel-name column (tried {list(_NAME_COLUMNS)}) "
                    f"or no timing columns (tried duration {list(_DURATION_COLUMNS)}, "
                    f"start {list(_START_COLUMNS)}, end {list(_END_COLUMNS)}); "
                    f"found headers {header}"
                )
            # Record which columns won, once, per file group.
            columns.setdefault("kernel", name_col)
            columns.setdefault("duration", dur_col)
            columns.setdefault("start", start_col)
            columns.setdefault("end", end_col)
            columns.setdefault("agent", agent_col)
            if name_col != columns.get("kernel"):
                raise KernelFractionError(
                    f"{path}: kernel-name column differs across inputs"
                )

            for line in reader:
                if not line or len(line) < len(header):
                    continue
                record = dict(zip(header, line, strict=False))
                kernel = demangle((record.get(name_col) or "").strip())
                if not kernel:
                    continue
                if dur_col is not None:
                    dur = int(float(record.get(dur_col) or 0))
                else:
                    assert (
                        start_col is not None and end_col is not None
                    )  # checked above
                    dur = int(float(record.get(end_col) or 0)) - int(
                        float(record.get(start_col) or 0)
                    )
                if dur < 0:
                    raise KernelFractionError(
                        f"{path}: negative kernel duration for {kernel!r}"
                    )
                if start_col is not None and end_col is not None:
                    start = int(float(record.get(start_col) or 0))
                    spans.append((start, start + dur))
                agent = record.get(agent_col) if agent_col else None
                rows.append(
                    {
                        "kernel": kernel,
                        "family": classify(kernel),
                        "dur_ns": dur,
                        "agent": _parse_agent(agent),
                    }
                )

    if not rows:
        raise KernelFractionError(f"no kernel rows parsed from {len(paths)} file(s)")
    if spans:
        wall_ns = max(end for _, end in spans) - min(start for start, _ in spans)
    else:
        wall_ns = sum(row["dur_ns"] for row in rows)
    return {
        "rows": rows,
        "spans": spans,
        "wall_ns": wall_ns,
        "columns": columns,
        "files": [str(p) for p in paths],
    }


def family_summary(trace: dict[str, Any]) -> dict[str, Any]:
    """Aggregate a parsed trace into per-family kernel time and the ceilings."""
    by_family: dict[str, int] = {}
    for row in trace["rows"]:
        by_family[row["family"]] = by_family.get(row["family"], 0) + row["dur_ns"]
    kernel_total = sum(by_family.values())
    matmul_ns = sum(by_family.get(family, 0) for family in MATMUL_FAMILIES)
    busy_ns = gpu_busy_ns(trace["spans"]) if trace["spans"] else kernel_total
    wall_ns = trace["wall_ns"]
    matmul_kernel_pct = 100.0 * matmul_ns / kernel_total if kernel_total else 0.0
    gpu_busy_pct = 100.0 * busy_ns / wall_ns if wall_ns else 0.0
    return {
        "by_family": by_family,
        "kernel_total_ns": kernel_total,
        "matmul_ns": matmul_ns,
        "matmul_kernel_pct": matmul_kernel_pct,
        "unmapped_ns": by_family.get("unmapped", 0),
        "unmapped_pct": (
            100.0 * by_family.get("unmapped", 0) / kernel_total if kernel_total else 0.0
        ),
        "busy_ns": busy_ns,
        "gpu_busy_pct": gpu_busy_pct,
        "wall_ns": wall_ns,
        # The wall-clock ceiling is the product of the two measured ratios,
        # NOT matmul_ns / wall_ns: overlapping kernels (separate agents or
        # queues) make family kernel time a sum that can exceed the wall
        # window, and the raw quotient would then report >100% of wall.
        # matmul_share_of_kernel_time x kernel_time_share_of_wall is bounded
        # by each factor: 45.2% x 61.3% -> 27.7% in the spec's example.
        "matmul_wall_pct": (matmul_kernel_pct * gpu_busy_pct) / 100.0,
    }


def _fmt_ns(ns: int) -> str:
    """Adaptive time formatting so short synthetic and real traces both read."""
    if ns >= 1_000_000_000:
        return f"{ns / 1e9:.3f} s"
    if ns >= 1_000_000:
        return f"{ns / 1e6:.3f} ms"
    if ns >= 1_000:
        return f"{ns / 1e3:.3f} us"
    return f"{ns} ns"


def render_report(trace: dict[str, Any], summary: dict[str, Any], phase: str) -> str:
    """Render both numbers, always: kernel-time share and the wall ceiling."""
    lines = [
        f"GPU kernel time by family ({phase} phase)",
        f"  traced wall: {_fmt_ns(summary['wall_ns'])}, "
        f"kernel rows: {len(trace['rows'])}",
    ]
    order = sorted(summary["by_family"], key=lambda f: -summary["by_family"][f])
    for family in order:
        ns = summary["by_family"][family]
        share = 100.0 * ns / summary["kernel_total_ns"]
        lines.append(f"  {family:<15} {share:5.1f}%")
    lines.append(
        f"  -- matmul       {summary['matmul_kernel_pct']:5.1f}%"
        "   <- ceiling on tuning, of kernel time"
    )
    lines.append("")
    lines.append(f"GPU busy {summary['gpu_busy_pct']:.1f}% of traced wall time")
    lines.append(
        f"  => matmul is {summary['matmul_wall_pct']:.1f}% of wall time; "
        "a 10% matmul saving is "
        f"{summary['matmul_wall_pct'] / 10.0:.1f}% end to end"
    )
    lines.append("")
    lines.append(
        f"unmapped {summary['unmapped_pct']:.1f}% of kernel time"
        + (
            "  (pattern table incomplete; matmul fraction understated)"
            if summary["unmapped_pct"] > 5.0
            else ""
        )
    )
    columns = trace["columns"]
    lines.append("")
    lines.append(
        f"columns: kernel={columns.get('kernel')!r} "
        f"timing={columns.get('duration') or (columns.get('start'), columns.get('end'))!r} "
        f"agent={columns.get('agent')!r}"
    )
    lines.append(f"files: {', '.join(trace['files'])}")
    return "\n".join(lines) + "\n"


def build_parser(subparsers) -> None:
    """Register the ``kernel-fraction`` subcommand (HI35 Part 2)."""
    cmd = subparsers.add_parser(
        "kernel-fraction",
        help="matmul fraction of traced kernel/wall time from rocprofv3 CSVs",
    )
    cmd.add_argument("csv", nargs="+", help="rocprofv3 --kernel-trace CSV file(s)")
    cmd.add_argument(
        "--phase",
        default="decode",
        help="phase label for the report (e.g. prefill, decode)",
    )
    cmd.add_argument("--output", default=None, help="write the report to this path")
    cmd.set_defaults(func=_cmd_kernel_fraction)


def _cmd_kernel_fraction(args: argparse.Namespace) -> int:
    try:
        trace = parse_kernel_trace([Path(p) for p in args.csv])
        summary = family_summary(trace)
    except (KernelFractionError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = render_report(trace, summary, args.phase)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0
