#!/usr/bin/env python3
"""Compare two rocprofv3 kernel traces of the same Flash-Next prefill (fn-profile.sh).

For each trace, per GPU agent, inside the busiest window (the prefill): wall span, summed kernel
time, idle time between kernels, kernel count, and the kernels whose total time changed most.
A slowdown that shows in summed kernel time is on the GPU (clocks, memory); one that shows in
idle gaps is on the host side (launch, sync, copies)."""
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["Agent_Id"]), int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"][:90]))
    return rows


def window(rows, gap_ns=200_000_000):
    """The longest run of kernels with no idle gap over gap_ns: the request, not load or warm-up."""
    rows = sorted(rows, key=lambda r: r[1])
    best, cur = [], [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        if r[1] - prev[2] > gap_ns:
            best = max(best, cur, key=lambda c: c[-1][2] - c[0][1])
            cur = []
        cur.append(r)
    return max(best, cur, key=lambda c: c[-1][2] - c[0][1])


def summarise(path):
    rows = window(load(path))
    per_agent = defaultdict(list)
    for r in rows:
        per_agent[r[0]].append(r)
    out = {}
    for agent, ks in per_agent.items():
        ks.sort(key=lambda r: r[1])
        busy, end = 0, ks[0][1]
        for _, s, e, _ in ks:
            s2 = max(s, end)
            if e > s2:
                busy += e - s2
            end = max(end, e)
        span = ks[-1][2] - ks[0][1]
        names = defaultdict(int)
        for _, s, e, n in ks:
            names[n] += e - s
        out[agent] = dict(span=span, busy=busy, idle=span - busy, count=len(ks), names=names)
    return out


def main():
    a, b = summarise(Path(sys.argv[1])), summarise(Path(sys.argv[2]))
    for agent in sorted(set(a) & set(b)):
        x, y = a[agent], b[agent]
        print(f"agent {agent}: span {x['span']/1e6:.1f} -> {y['span']/1e6:.1f} ms, "
              f"kernel {x['busy']/1e6:.1f} -> {y['busy']/1e6:.1f} ms, "
              f"idle {x['idle']/1e6:.1f} -> {y['idle']/1e6:.1f} ms, kernels {x['count']} -> {y['count']}")
        delta = sorted(((y['names'].get(n, 0) - x['names'].get(n, 0), n) for n in set(x['names']) | set(y['names'])),
                       key=lambda t: -abs(t[0]))
        for d, n in delta[:8]:
            print(f"    {d/1e6:+8.2f} ms  {n}")


if __name__ == "__main__":
    main()
