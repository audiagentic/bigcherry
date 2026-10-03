#!/usr/bin/env python3
"""QFP11: decompose the per-AllReduce boundary cost in decode from a rocprofv3 kernel trace (cpu-root AR, 1291).

Each cpu-root AllReduce is a cpu_root_produce kernel (rank writes its slice to host) followed by cpu_root_consume
(rank spins until the CPU sum is published). Per AR (aligned by ordinal per GPU over the decode window):
  gap_before  previous kernel end -> produce start          (split boundary / launch latency into the AR)
  skew        latest produce start - earliest produce start (rank arrival imbalance)
  host_rt     latest produce end -> earliest consume end    (CPU-root round trip once everyone has arrived)
  consume     consume duration per rank                     (wait for slower ranks + round trip)
  gap_after   consume end -> next kernel start              (boundary back into compute)

Usage: ar-boundary.py <run-dir>   (rocprof/**/*kernel_trace.csv + decode.timings.json)
"""
import csv
import glob
import json
import statistics
import sys


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


def main():
    run = sys.argv[1]
    t = json.load(open(f"{run}/decode.timings.json"))[-1]
    win_ns = t["predicted_n"] / t["predicted_per_second"] * 1e9
    path = sorted(glob.glob(f"{run}/rocprof/**/*kernel_trace.csv", recursive=True))[0]
    rows = [(r["Agent_Id"], int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"])
            for r in csv.DictReader(open(path))]
    end = max(r[2] for r in rows)
    rows = sorted((r for r in rows if r[1] >= end - win_ns), key=lambda r: (r[0], r[1]))
    per = {}
    for agent in sorted({r[0] for r in rows}):
        ks = [r for r in rows if r[0] == agent]
        ars = []
        for i, (a, s, e, name) in enumerate(ks):
            if "cpu_root_produce" not in name:
                continue
            j = next((k for k in range(i + 1, min(i + 6, len(ks))) if "cpu_root_consume" in ks[k][3]), None)
            if j is None:
                continue
            prev_end = ks[i - 1][2] if i > 0 else s
            next_start = ks[j + 1][1] if j + 1 < len(ks) else ks[j][2]
            ars.append(dict(p_s=s, p_e=e, c_s=ks[j][1], c_e=ks[j][2], gap_before=s - prev_end,
                            gap_after=next_start - ks[j][2]))
        if ars:
            per[agent] = ars
    agents = sorted(per)
    n = min(len(per[a]) for a in agents)
    print(f"decode window {win_ns / 1e9:.2f} s, {t['predicted_n']} tokens; ARs per GPU {[len(per[a]) for a in agents]}"
          f" -> aligned {n} ({n / t['predicted_n']:.1f}/token)")
    skew, rt = [], []
    for k in range(n):
        ps = [per[a][k]["p_s"] for a in agents]
        pe = [per[a][k]["p_e"] for a in agents]
        ce = [per[a][k]["c_e"] for a in agents]
        skew.append(max(ps) - min(ps))
        rt.append(min(ce) - max(pe))
    us = lambda xs: f"median {statistics.median(xs) / 1e3:7.1f} us  p90 {pct(xs, .9) / 1e3:7.1f} us"
    print(f"arrival skew        {us(skew)}")
    print(f"host round trip     {us(rt)}  (latest produce end -> first consume end)")
    for a in agents:
        ars = per[a][:n]
        last = [k for k in range(n) if per[a][k]["p_s"] == max(per[b][k]["p_s"] for b in agents)]
        print(f"agent {a}: last to arrive {100 * len(last) / n:4.0f}% of ARs")
        print(f"  gap_before  {us([x['gap_before'] for x in ars])}")
        print(f"  consume     {us([x['c_e'] - x['c_s'] for x in ars])}")
        print(f"  gap_after   {us([x['gap_after'] for x in ars])}")
        per_tok = sum(x['gap_before'] + (x['c_e'] - x['p_s']) + x['gap_after'] for x in ars) / t['predicted_n'] / 1e6
        print(f"  boundary total {per_tok:.3f} ms/token (gap_before + produce..consume + gap_after)")


if __name__ == "__main__":
    main()
