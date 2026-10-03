#!/usr/bin/env python3
"""QFP09: which work makes a GPU arrive late at an AllReduce? Per-segment attribution from a rocprofv3 trace.

A segment is the span on one GPU from the end of AR k-1's consume kernel to the start of AR k's produce kernel.
For every AR (aligned by ordinal), the latest-arriving GPU's segment is compared with the earliest GPU's segment:
the per-op-class difference in kernel time (plus idle gaps) says what made it late. Results are aggregated over all
ARs, and segments are grouped by their dominant op class so recurring slow segment types stand out.

Usage: ar-segment.py <run-dir>   (rocprof/**/*kernel_trace.csv + decode.timings.json)
"""
import collections
import csv
import glob
import json
import os
import importlib.util
import sys


def _load_classify():
    """rank-census.py's op-class table (hyphenated file name, so load it by path)."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("rank_census", os.path.join(here, "rank-census.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.classify


def main():
    cls_of = _load_classify()
    run = sys.argv[1]
    t = json.load(open(f"{run}/decode.timings.json"))[-1]
    win_ns = t["predicted_n"] / t["predicted_per_second"] * 1e9
    path = sorted(glob.glob(f"{run}/rocprof/**/*kernel_trace.csv", recursive=True))[0]
    rows = [(r["Agent_Id"], int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"])
            for r in csv.DictReader(open(path))]
    end = max(r[2] for r in rows)
    rows = sorted((r for r in rows if r[1] >= end - win_ns), key=lambda r: (r[0], r[1]))
    # per GPU: list of segments, each = (arrival_ts, {class: ns}, idle_ns)
    segs = {}
    for agent in sorted({r[0] for r in rows}):
        ks = [r for r in rows if r[0] == agent]
        if not any("cpu_root_produce" in k[3] for k in ks):
            continue  # draft GPU
        out, cur, idle, prev_end, started = [], collections.Counter(), 0, None, False
        for a, s, e, name in ks:
            if "cpu_root_consume" in name:
                cur, idle, prev_end, started = collections.Counter(), 0, e, True
                continue
            if "cpu_root_produce" in name:
                if started:
                    if prev_end is not None:
                        idle += max(0, s - prev_end)
                    out.append((s, cur, idle))
                continue
            if started:
                if prev_end is not None:
                    idle += max(0, s - prev_end)
                cur[cls_of(name)] += e - s
                prev_end = e
        segs[agent] = out
    agents = sorted(segs)
    n = min(len(segs[a]) for a in agents)
    blame = collections.Counter()
    blame_idle = 0
    by_type = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    last_count = collections.Counter()
    for k in range(n):
        arr = {a: segs[a][k][0] for a in agents}
        last = max(agents, key=lambda a: arr[a])
        first = min(agents, key=lambda a: arr[a])
        last_count[last] += 1
        lc, li = segs[last][k][1], segs[last][k][2]
        fc, fi = segs[first][k][1], segs[first][k][2]
        for c in set(lc) | set(fc):
            blame[c] += lc[c] - fc[c]
        blame_idle += li - fi
        dom = max(lc, key=lc.get) if lc else "empty"
        bt = by_type[dom]
        bt[0] += 1
        bt[1] += arr[last] - arr[first]
        bt[2][last] += 1
    tok = t["predicted_n"]
    print(f"{n} aligned ARs over {tok} tokens; last to arrive: " +
          ", ".join(f"agent {a} {100 * last_count[a] / n:.0f}%" for a in agents))
    print("\nlast-minus-first per AR, summed (ms/token): what the late GPU spent extra time on")
    for c, ns in sorted(blame.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {ns / 1e6 / tok:7.3f}  {c}")
    print(f"  {blame_idle / 1e6 / tok:7.3f}  (idle gaps inside the segment)")
    print("\nsegments grouped by the late GPU's dominant class: count, mean skew, who was late")
    for dom, (cnt, skew, who) in sorted(by_type.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"  {dom:20s} {cnt:6d}  {skew / cnt / 1e3:7.1f} us  " +
              ", ".join(f"{a}:{who[a]}" for a in agents))


if __name__ == "__main__":
    main()
