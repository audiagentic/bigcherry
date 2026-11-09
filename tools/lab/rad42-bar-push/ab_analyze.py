#!/usr/bin/env python3
"""Card-free RAD50 DMA-BUF paired A/B summary. Never a serving-speed claim.

Input is raw rad50-tester --test dmabuf-ab stdout saved outside this tool.
Requires both paths at every observed (device,bytes,style); rejects duplicates,
nonpositive latencies, missing pairs and inconsistent iteration counts.
A serial microbenchmark ratio is NOT a predicted inference speedup.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

REQUIRED = ("device", "bytes", "iterations", "style", "path",
            "cpu_write_p50_us", "sync_start_p50_us", "sync_end_p50_us",
            "gpu_kernel_p50_us", "seq_p50_us", "seq_p95_us", "seq_p99_us")

def parse(text: str) -> dict[tuple[int, int, str], dict[str, dict[str, str]]]:
    cases: dict[tuple[int, int, str], dict[str, dict[str, str]]] = {}
    for line in text.splitlines():
        if not line.startswith("RAD50_AB "):
            continue
        row = {}
        for piece in line.split()[1:]:
            if "=" not in piece:
                raise ValueError(f"malformed field {piece!r}")
            k, v = piece.split("=", 1)
            if k in row:
                raise ValueError(f"duplicate field {k}")
            row[k] = v
        missing = set(REQUIRED) - row.keys()
        if missing:
            raise ValueError(f"missing fields {sorted(missing)}")
        dev, size = int(row["device"]), int(row["bytes"])
        iterations = int(row["iterations"])
        if dev < 0 or size <= 0 or iterations < 10 or iterations > 200:
            raise ValueError("invalid device/bytes/iterations")
        if row["style"] not in ("direct-scalar", "staged-memcpy"):
            raise ValueError("unknown CPU write style")
        if row["path"] not in ("mapped-host-pull", "dmabuf-cpu-push"):
            raise ValueError("unknown transfer path")
        for key in REQUIRED[5:]:
            number = float(row[key])
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"invalid number {key}")
            if key.startswith("seq_") and number <= 0:
                raise ValueError(f"zero serial latency {key}")
        key = (dev, size, row["style"])
        slot = cases.setdefault(key, {})
        if row["path"] in slot:
            raise ValueError(f"duplicate result for {key} {row['path']}")
        slot[row["path"]] = row
    if not cases:
        raise ValueError("no RAD50_AB sample rows")
    for key, pair in cases.items():
        if set(pair) != {"mapped-host-pull", "dmabuf-cpu-push"}:
            raise ValueError(f"unpaired sample {key}")
        a, b = pair.values()
        if a["iterations"] != b["iterations"]:
            raise ValueError(f"unequal repetitions {key}")
    return cases

def format_report(text: str) -> str:
    cases = parse(text)
    output = ["RAD50 paired SERIAL A/B; no inference gain, GPU contention or "
              "concurrent-ready claim.",
              "Positive delta_us and speedup > 1.0 favour CPU BAR-push.",
              "device bytes style n pull_p50_us push_p50_us speedup "
              "delta_us push_sync_p50_us push_write_p50_us"]
    best = {}
    for (dev, size, style), pair in sorted(cases.items()):
        pull = pair["mapped-host-pull"]
        push = pair["dmabuf-cpu-push"]
        a, b = float(pull["seq_p50_us"]), float(push["seq_p50_us"])
        overhead = float(push["sync_start_p50_us"]) + float(push["sync_end_p50_us"])
        output.append(f"{dev} {size} {style} {pull['iterations']} "
                      f"{a:.2f} {b:.2f} {a/b:.3f} {a-b:+.2f} "
                      f"{overhead:.2f} {float(push['cpu_write_p50_us']):.2f}")
        key = (dev, size)
        if key not in best or b < best[key][0]:
            best[key] = (b, style)
    for (dev, size), (latency, style) in sorted(best.items()):
        output.append(f"FASTEST_TESTED_PUSH_METHOD device={dev} bytes={size} "
                      f"style={style} seq_p50_us={latency:.2f} "
                      "NOTE=serial_synthetic_only")
    output.append("Stop/go requires separate concurrent visibility, real workload "
                  "trace and paired end-to-end inference runs.")
    return "\n".join(output)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="raw stdout of one or more paired runs")
    args = ap.parse_args()
    try:
        print(format_report(args.log.read_text()))
    except (ValueError, OSError) as exc:
        ap.exit(2, f"RAD50 A/B analysis rejected input: {exc}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
