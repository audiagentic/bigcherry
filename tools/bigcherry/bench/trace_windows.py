"""Helper TRACE lines bucketed by the rounds of a stream-bench record (RAD10-02), as managed code a harvest recipe can run.

The helpers print one TRACE line per finished request (R3_MOE_HELPER_TRACE=N):
    TRACE,lane,L,seq,S,layer,Y,rows,R,routes,K,local_us,A,wait_us,B,helper_gpu_us,C,enter,T,wall_ns,W
Lines are grouped by `enter` (one layer of one step; both helpers' lines of it share the value). A PAIRED group has both
helpers' lines and its exposed wait is the larger wait_us; a SINGLE group has one. `wall_ns` is the helper host's epoch
clock when the line was printed and places a layer-step in a stream-bench round (its wall_start_ns..wall_end_ns, plus slack
because a line is printed after the request). Durations on the helper clock are never compared with the engine's ticks.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path


def pct(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def read_steps(paths, small_rows: int = 8) -> list:
    groups = collections.defaultdict(list)
    for path in paths:
        for line in open(path, errors="replace"):
            if not line.startswith("TRACE,"):
                continue
            f = line.strip().split(",")
            d = {f[i]: f[i + 1] for i in range(1, len(f) - 1, 2)}
            groups[int(d["enter"])].append(d)
    steps = []
    for ds in groups.values():
        rows = int(ds[0]["rows"])
        steps.append({
            "large": rows > small_rows, "rows": rows, "paired": len(ds) > 1, "layer": int(ds[0]["layer"]),
            "wait": max(float(d["wait_us"]) for d in ds),
            "helper_max": max(float(d["helper_gpu_us"]) for d in ds),
            "helper_min": min(float(d["helper_gpu_us"]) for d in ds),
            "wall_ns": max((int(d["wall_ns"]) for d in ds if "wall_ns" in d), default=None),
        })
    return steps


def window_summary(steps, rounds, slack_ns: int = 300_000_000) -> dict:
    """Per stream count: what the helpers saw during that count's rounds."""
    out: dict = {}
    for r in rounds:
        lo, hi = r["wall_start_ns"], r["wall_end_ns"] + slack_ns
        mine = [s for s in steps if s["wall_ns"] is not None and lo <= s["wall_ns"] <= hi]
        acc = out.setdefault(r["streams"], {"steps": [], "rounds": 0})
        acc["steps"] += mine
        acc["rounds"] += 1
    result = {}
    for n, acc in sorted(out.items()):
        st = acc["steps"]
        paired = [s for s in st if s["paired"]]
        small = [s for s in st if not s["large"]]
        result[str(n)] = {
            "rounds": acc["rounds"], "layer_steps": len(st),
            "rows_p50": pct([s["rows"] for s in st], .5), "rows_p95": pct([s["rows"] for s in st], .95),
            "rows_max": max((s["rows"] for s in st), default=0),
            "small_lane_share": (len(small) / len(st)) if st else None,
            "exposed_wait_us_p50": pct([s["wait"] for s in paired], .5),
            "exposed_wait_us_p95": pct([s["wait"] for s in paired], .95),
            "exposed_wait_us_sum_per_round": sum(s["wait"] for s in paired) / acc["rounds"],
            "helper_gpu_us_p50": pct([s["helper_max"] for s in paired], .5),
            "helper_imbalance_p50": pct([(s["helper_max"] - s["helper_min"]) / s["helper_max"]
                                         for s in paired if s["helper_max"] > 0], .5),
            "single_helper_share": ((len(st) - len(paired)) / len(st)) if st else None,
        }
    return result


def summarise_cell(cell_dir: Path, small_rows: int = 8) -> dict | None:
    """The windowed helper summary of one benchmark cell (a stream-bench.json beside helper*.log), or None when the cell has
    no stream record, no helper logs, or TRACE lines without a wall clock."""
    record = cell_dir / "stream-bench.json"
    logs = sorted(cell_dir.glob("helper*.log"))
    if not record.is_file() or not logs:
        return None
    steps = read_steps([str(p) for p in logs], small_rows)
    if not steps or not any(s["wall_ns"] is not None for s in steps):
        return None
    return window_summary(steps, json.loads(record.read_text(encoding="utf-8"))["rounds"])
