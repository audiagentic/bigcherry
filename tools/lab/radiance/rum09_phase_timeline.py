"""RUM09: align native rank HIP phase diagnostics to cold/warm benchmark windows.

HIP-event intervals within a lane are device time; categories can overlap
between lanes and must not be summed to assert end-to-end wall time.
An absent phase or invalid event trace refuses the comparison.
"""
from __future__ import annotations

import argparse
import itertools
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
            # A prefill scheduler step may run the MTP draft passes as
            # RAD_PHASE_DECODE=1. Its *model* pass (draft_pass=0) is still
            # RAD_PHASE_PREFILL=0, so filtering individual pass phase tags
            # falsely mixes 125 prefill steps into the cold decode median.
            window_rows = [r for r in steps if start <= r["wall_start_ns"] and
                           r["wall_end_ns"] <= end]
            model_phases = {r["step"]: r["phase"] for r in window_rows
                            if r.get("draft_pass") == 0}
            rows = [r for r in window_rows if model_phases.get(r.get("step")) == 1]
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
            ordered = []
            for step_id, group in grouped.items():
                group.sort(key=lambda r: r["wall_start_ns"])
                for prev, curr in itertools.pairwise(group):
                    if curr["wall_start_ns"] < prev["wall_end_ns"]:
                        raise ValueError(f"rank {rank}: overlapping passes of step {step_id}")
                ordered.append((group[0]["wall_start_ns"], group[-1]["wall_end_ns"], group))
            ordered.sort(key=lambda entry: entry[0])
            between_step = []
            for previous, following in itertools.pairwise(ordered):
                gap = following[0] - previous[1]
                if gap < 0:
                    raise ValueError(f"rank {rank}: overlapping scheduler steps")
                between_step.append(gap / 1e6)
            within_step = [sum(
                max(0, right["wall_start_ns"] - left["wall_end_ns"])/1e6
                for left, right in itertools.pairwise(group))
                for _, _, group in ordered]
            measured_steps = rounds[i].get("engine", {}).get("engine_steps_decode_total")
            if measured_steps is not None and len(grouped) != int(measured_steps):
                raise ValueError(
                    f"rank {rank}: decode-step count {len(grouped)} differs "
                    f"from benchmark counter {measured_steps}")
            item = {"phase": ("cold", "warm")[i], "rank": rank, "passes": len(rows),
                    "steps": len(grouped),
                    "rank_step_wall_ms_p50": p50([(hi-lo)/1e6 for lo, hi, _ in ordered]),
                    "inter_pass_host_gap_ms_p50": p50(within_step),
                    "between_step_host_gap_ms_p50": p50(between_step)}
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
            "caveat": ("HIP event synchronization perturbs timing; recorded-pass playback remains enabled. "
                       "Rank step spans and between-step host gaps are wall-clock measurements, not HIP events. "
                       "Per-lane GPU intervals can overlap. Async DMA device activity is unavailable "
                       "rather than zero.")}


def interval_union_ns(spans):
    """Host elapsed time across concurrent ranks, not a sum of device events."""
    total = 0
    end = -1
    for lo, hi in sorted(spans):
        if hi > max(lo, end):
            total += hi - max(lo, end)
        end = max(end, hi)
    return total


def analyze_prefill(bench: Path, trace_dir: Path, expected_ranks: int) -> dict:
    """Single-request first-token attribution, including draft and residual."""
    ranks = read_steps(trace_dir)
    if len(ranks) != expected_ranks:
        raise ValueError("rank count differs from expected topology")
    output = []
    for rnd in json.loads(bench.read_text())["rounds"]:
        if not rnd["ok"] or rnd["streams"] != 1:
            raise ValueError("first-token attribution requires a successful single request")
        req = rnd["per_stream"][0]
        start, first = req["start_wall_ns"], req["first_wall_ns"]
        if not rnd["wall_start_ns"] <= start < first <= rnd["wall_end_ns"]:
            raise ValueError("invalid first-token clock window")
        ttft = (first-start)/1e6
        if abs(ttft-req["ttft_ms"]) > 10:
            raise ValueError("wall clock and monotonic TTFT differ")
        model, draft, details = [], [], {}
        keys = None
        for rank, steps in ranks.items():
            rows = [r for r in steps if start <= r["wall_start_ns"] and r["wall_end_ns"] <= first]
            if not rows:
                raise ValueError("no trace in first-token window")
            if any(r["wall_start_ns"] < first and r["wall_end_ns"] > start
                   and r not in rows for r in steps):
                raise ValueError("pass crosses first-token window")
            model_keys = {r["step"] for r in rows if r["draft_pass"] == 0 and r["phase"] == 0}
            if not model_keys or any(r["step"] not in model_keys for r in rows):
                raise ValueError("not a pure prefill request")
            if keys is None:
                keys = model_keys
            elif keys != model_keys:
                raise ValueError("ranks disagree on prefill steps")
            count = (rnd.get("engine") or {}).get("engine_steps_prefill_total")
            if count is None or count != len(model_keys):
                raise ValueError("prefill step counter mismatch")
            ops = {}
            for r in rows:
                span = (r["wall_start_ns"], r["wall_end_ns"])
                (model if r["draft_pass"] == 0 else draft).append(span)
                for seg in r.get("tape_segments", []):
                    ms = seg["ms"]
                    if not isinstance(ms, (float,int)) or not math.isfinite(ms) or ms < 0:
                        raise ValueError("invalid device interval")
                    op = seg["op"]
                    ops[op] = ops.get(op, 0.0) + ms
            details[str(rank)] = {"passes": len(rows), "device_op_ms": ops}
        m, d = interval_union_ns(model), interval_union_ns(draft)
        u = interval_union_ns(model+draft)
        output.append({"client_ttft_ms":ttft, "model_pass_union_ms":m/1e6,
                       "draft_pass_union_ms":d/1e6, "overlap_ms":(m+d-u)/1e6,
                       "unattributed_ms":(first-start-u)/1e6, "ranks":details})
    return {"schema_version":2, "benchmark":bench.name, "rounds":output,
            "caveat":"Host pass unions include trace overhead. Device intervals can overlap; unclassified is not compute-only. Residual is not measured HTTP or admission time."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--prefill-ranks", type=int, help="single-request first-token attribution")
    args = parser.parse_args()
    report = analyze_prefill(args.benchmark, args.trace_dir, args.prefill_ranks) if args.prefill_ranks is not None else analyze(args.benchmark, args.trace_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
