#!/usr/bin/env python3
"""Verify completed RAD50 A/B logs before computing synthetic serial ratios.

A benchmark prints RAD50_AB_CONTEXT first and RAD50_AB_COMPLETE only AFTER
GPU verification, helper exit and resource cleanup succeed. Incomplete or
failed logs are NOT performance results. Separate successful logs may be
concatenated to compare devices, sizes and CPU writing styles.

This does not measure or predict inference throughput or concurrent readiness.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

SIZES = (4096, 65536, 262144, 1048576, 4194304)
COMBINE_SIZES = (5120, 81920, 327680, 1310720, 3932160)
PATHS = {"mapped-host-pull", "dmabuf-cpu-push"}
STYLES = {"direct-scalar", "staged-memcpy", "combine-direct", "combine-staged"}
def valid_sizes(style: str) -> tuple[int, ...]:
    return COMBINE_SIZES if style.startswith("combine-") else SIZES
REQUIRED = (
    "device", "bytes", "iterations", "style", "path",
    "cpu_write_p50_us", "sync_start_p50_us", "sync_end_p50_us",
    "gpu_kernel_p50_us", "seq_p50_us", "seq_p95_us", "seq_p99_us",
)


def fields(line: str, prefix: str) -> dict[str, str]:
    row = {}
    for piece in line[len(prefix):].split():
        if "=" not in piece:
            raise ValueError(f"malformed {prefix.strip()} field {piece!r}")
        key, value = piece.split("=", 1)
        if key in row:
            raise ValueError(f"duplicate {key}")
        row[key] = value
    return row


def metadata(row: dict[str, str]) -> tuple[int, int, int, str]:
    try:
        dev = int(row["device"])
        iterations = int(row["iterations"])
        size_filter = int(row["bytes_filter"])
        style = row["style"]
    except (ValueError, KeyError) as e:
        raise ValueError(f"bad benchmark metadata: {e}") from e
    if dev < 0 or not 10 <= iterations <= 200 or style not in STYLES:
        raise ValueError("invalid benchmark device, sample count or style")
    if size_filter not in (0, *valid_sizes(style)):
        raise ValueError("invalid benchmark byte filter")
    return dev, iterations, size_filter, style


def parse(text: str) -> dict[tuple[int, int, str], dict[str, dict[str, str]]]:
    cases: dict[tuple[int, int, str], dict[str, dict[str, str]]] = {}
    active = None
    completed = 0
    for line in text.splitlines():
        if line.startswith("exitCode: ") and line.strip() != "exitCode: 0":
            raise ValueError("nonzero shell command exit")
        if line.startswith("timedOut: ") and line.strip() != "timedOut: false":
            raise ValueError("timed-out benchmark log")
        if line.startswith("truncated: ") and line.strip() != "truncated: false":
            raise ValueError("truncated benchmark log")
        if line.startswith("RAD50 A/B FAILED") or line.startswith("RAD50 A/B WRONG DATA"):
            raise ValueError("benchmark logged a failure")
        if (line.startswith("RAD50 RESULT test=dmabuf-ab ") or
            line.startswith("RAD50 RESULT test=combine-ab ")):
            result = fields(line, "RAD50 RESULT ")
            if result.get("exit") != "0":
                raise ValueError("benchmark process returned nonzero")
        if line.startswith("RAD50_AB_CONTEXT "):
            if active is not None:
                raise ValueError("previous benchmark has no completion record")
            row = fields(line, "RAD50_AB_CONTEXT ")
            dev, count, size_filter, style = metadata(row)
            active = (dev, count, size_filter, style, {}, set())
        elif line.startswith("RAD50_AB "):
            if active is None:
                raise ValueError("benchmark sample outside a run")
            row = fields(line, "RAD50_AB ")
            missing = set(REQUIRED) - row.keys()
            if missing:
                raise ValueError(f"missing fields {sorted(missing)}")
            try:
                dev, size, count = (int(row[k]) for k in ("device", "bytes", "iterations"))
            except ValueError as e:
                raise ValueError(f"invalid sample metadata: {e}") from e
            expected_dev, expected_count, size_filter, style, rows, _ = active
            if (dev != expected_dev or count != expected_count or
                    size not in valid_sizes(style) or
                    (size_filter != 0 and size != size_filter) or
                    row["style"] != style or row["path"] not in PATHS):
                raise ValueError("sample does not match benchmark context")
            for key in REQUIRED[5:]:
                try:
                    number = float(row[key])
                except ValueError as e:
                    raise ValueError(f"bad numeric field {key}") from e
                if not math.isfinite(number) or number < 0:
                    raise ValueError(f"invalid numeric field {key}")
                if key.startswith("seq_") and number <= 0:
                    raise ValueError(f"nonpositive serial latency {key}")
            pair = rows.setdefault(size, {})
            if row["path"] in pair:
                raise ValueError("duplicate sample path in one run")
            pair[row["path"]] = row
        elif line.startswith("RAD50_AB_COMPARE "):
            if active is None:
                raise ValueError("comparison outside a benchmark")
            compare = fields(line, "RAD50_AB_COMPARE ")
            dev, _, size_filter, style, rows, compared = active
            try:
                value_dev, size = int(compare["device"]), int(compare["bytes"])
            except (ValueError, KeyError) as e:
                raise ValueError("invalid comparison metadata") from e
            if (value_dev != dev or size not in rows or
                    (size_filter and size != size_filter) or
                    compare.get("method") != style or size in compared or
                    set(rows[size]) != PATHS):
                raise ValueError("unpaired, repeated or mismatched comparison")
            compared.add(size)
        elif line.startswith("RAD50_AB_COMPLETE "):
            if active is None:
                raise ValueError("completion without running benchmark")
            end = fields(line, "RAD50_AB_COMPLETE ")
            dev, count, size_filter, style, rows, compared = active
            if end.get("status") != "0" or metadata(end) != (
                    dev, count, size_filter, style):
                raise ValueError("completion status/metadata mismatch")
            sizes = {size_filter} if size_filter else set(valid_sizes(style))
            if set(rows) != sizes or compared != sizes:
                raise ValueError("incomplete size coverage or missing comparison")
            for size, pair in rows.items():
                if set(pair) != PATHS:
                    raise ValueError("incomplete path pairing")
                key = (dev, size, style)
                if key in cases:
                    raise ValueError("duplicate benchmark device/size/style")
                cases[key] = pair
            completed += 1
            active = None
    if active is not None:
        raise ValueError("benchmark truncated: missing success completion")
    if completed == 0:
        raise ValueError("no completed RAD50_AB runs")
    return cases


def format_report(text: str) -> str:
    cases = parse(text)
    output = [
        "RAD50 paired SERIAL A/B only: no inference or GPU-ready claim.",
        "Positive delta_us and speedup > 1.0 favour CPU BAR-push.",
        "device bytes style n pull_p50_us push_p50_us speedup "
        "delta_us push_sync_p50_us push_write_p50_us",
    ]
    best = {}
    for (dev, size, style), pair in sorted(cases.items()):
        pull = pair["mapped-host-pull"]
        push = pair["dmabuf-cpu-push"]
        a, b = float(pull["seq_p50_us"]), float(push["seq_p50_us"])
        overhead = float(push["sync_start_p50_us"]) + float(push["sync_end_p50_us"])
        output.append(
            f"{dev} {size} {style} {pull['iterations']} "
            f"{a:.2f} {b:.2f} {a/b:.3f} {a-b:+.2f} "
            f"{overhead:.2f} {float(push['cpu_write_p50_us']):.2f}"
        )
        workload="real-combine-synthetic-inputs" if style.startswith("combine-") else "synthetic-write"
        key = (dev, size, workload)
        if key not in best or b < best[key][0]:
            best[key] = (b, style)
    for (dev, size, workload), (latency, style) in sorted(best.items()):
        output.append(
            f"FASTEST_SUBMITTED_PUSH_METHOD device={dev} bytes={size} workload={workload} "
            f"style={style} seq_p50_us={latency:.2f} "
            "NOTE=synthetic_serial_only"
        )
    output.append("Gate 4 concurrent visibility, representative real MoE workload and paired inference tests remain required.")
    return "\n".join(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provisional", action="store_true",
                        help="report incomplete observed rows as diagnostics only")
    parser.add_argument("log", nargs="+", type=Path, help="raw RAD50 logs")
    args = parser.parse_args()
    try:
        data="\n".join(p.read_text() for p in args.log)
        if args.provisional:
            from ab_partial import provisional_report
            print(provisional_report(data))
        else:
            print(format_report(data))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"RAD50 A/B analysis rejected input: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
