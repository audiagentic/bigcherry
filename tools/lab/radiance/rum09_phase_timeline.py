"""RUM09: align native rank HIP phase diagnostics to cold/warm benchmark windows.

HIP-event intervals within a lane are device time; categories can overlap
between lanes and must not be summed to assert end-to-end wall time.
An absent phase or invalid event trace refuses the comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

CATEGORIES = ("compute_ms", "collective_ms", "host_kv_serial_ms", "host_kv_wait_ms",
              "host_kv_dma_ms", "idle_wait_ms")


def p50(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def read_steps(trace_dir: Path) -> dict[int, list[dict]]:
    if not trace_dir.is_dir():
        raise ValueError("missing step-phase trace directory")
    out: dict[int, list[dict]] = {}
    for path in sorted(trace_dir.glob("engine-rank-*.jsonl")):
        try:
            rank = int(path.stem.removeprefix("engine-rank-"))
        except ValueError as exc:
            raise ValueError(f"invalid rank trace name: {path.name}") from exc
        if rank < 0 or rank in out:
            raise ValueError(f"duplicate or invalid rank {rank}")
        data = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                step = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no}: invalid JSON") from exc
            if (step.get("schema_version") != 1 or step.get("rank") != rank or
                not isinstance(step.get("wall_start_ns"), int) or
                not isinstance(step.get("wall_end_ns"), int) or
                step["wall_start_ns"] >= step["wall_end_ns"] or
                step.get("valid") is not True):
                raise ValueError(f"{path.name}:{line_no}: invalid or incomplete HIP event measurement")
            for key in CATEGORIES:
                value = step.get(key)
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                    raise ValueError(f"{path.name}:{line_no}: invalid {key}")
            data.append(step)
        if not data:
            raise ValueError(f"{path.name}: no HIP event steps")
        out[rank] = data
    if not out:
        raise ValueError("no engine-rank HIP phase traces")
    if sorted(out) != list(range(len(out))):
        raise ValueError("missing or noncontiguous rank traces")
    return out


def analyze(bench: Path, trace_dir: Path) -> dict:
    record = json.loads(bench.read_text(encoding="utf-8"))
    rounds = record["rounds"]
    if len(rounds) != 2 or [r.get("rep") for r in rounds] != [0, 1]:
        raise ValueError("long-8 must have cold rep 0, warm rep 1")
    if not all(r.get("ok") is True for r in rounds):
        raise ValueError("benchmark contains failed rounds")
    windows = []
    for r in rounds:
        a, b = r["wall_start_ns"], r["wall_end_ns"]
        if type(a) is not int or type(b) is not int or a >= b:
            raise ValueError("invalid benchmark clock window")
        windows.append((a, b))
    if windows[0][1] >= windows[1][0]:
        raise ValueError("cold/warm benchmark clock windows overlap")
    by_rank = read_steps(trace_dir)
    outputs = {}
    for rank, steps in by_rank.items():
        results = []
        for i, (start, end) in enumerate(windows):
            # Only decode passes (RAD_PHASE_DECODE=1): prefill is reported
            # separately by the benchmark, not mixed into decode phase means.
            rows = [r for r in steps if r["phase"] == 1 and
                    start <= r["wall_start_ns"] and r["wall_end_ns"] <= end]
            if not rows:
                raise ValueError(f"rank {rank}: no decode HIP events in {'cold' if i == 0 else 'warm'} round")
            # One scheduler decode step may have several draft/verify passes.
            # Sum their serial per-pass device intervals *within the step*
            # before comparing p50s with the engine's per-step metrics.
            grouped = {}
            for r in rows:
                step_key = r.get("step")
                if type(step_key) is not int or step_key < 0:
                    raise ValueError(f"rank {rank}: missing or invalid scheduler step")
                grouped.setdefault(step_key, []).append(r)
            item = {"phase": ("cold", "warm")[i], "rank": rank, "passes": len(rows),
                    "steps": len(grouped),
                    "device_span_ms_p50": p50([
                        (max(r["wall_end_ns"] for r in group) -
                         min(r["wall_start_ns"] for r in group)) / 1e6
                        for group in grouped.values()])}
            for key in CATEGORIES:
                samples = [sum(r[key] for r in group if r.get(key) is not None)
                           for group in grouped.values()
                           if any(r.get(key) is not None for r in group)]
                item[key + "_p50"] = p50(samples)
                item[key + "_observed_steps"] = len(samples)
            results.append(item)
        delta = {}
        for key in CATEGORIES:
            a, b = results[0][key + "_p50"], results[1][key + "_p50"]
            delta[key] = round(b - a, 6) if a is not None and b is not None else None
        outputs[str(rank)] = {"rounds": results, "warm_minus_cold_ms_p50": delta}
    return {"schema_version": 1, "benchmark": bench.name, "ranks": outputs,
            "caveat": ("Profiled timings are perturbed by HIP event sync and disabled pass playback; "
                       "per-lane GPU intervals can overlap. Async DMA device activity is unavailable "
                       "rather than zero.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.benchmark, args.trace_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
