#!/usr/bin/env python3
"""QFP49: where a card's idle time in prefill sits, from a rocprofv3 kernel trace.

A gap is a stretch in which a card runs no kernel. Kernels are submitted ahead of time, so a card should only run
dry where the host could not queue the next kernel yet. This lists the gaps of each card over the window from the
first to the last large AllReduce kernel (the prefill, without model load):

  by size          how much idle time is in gaps of each length: many tiny gaps are launch latency, a few long ones
                   are the host doing something between pieces of work
  by what ran before   the kernel that ended just before the gap, so a wait that always follows one kind of kernel
                   shows up (for example the host working after every cross-card sum, or between batches)

Usage: kernel-gap-stats.py <rocprof dir> [window kernel pattern, default ncclDevKernel]
"""
from __future__ import annotations

import collections
import csv
import glob
import os
import re
import sys

BUCKETS = ((0.02, "< 20 us"), (0.1, "20-100 us"), (0.5, "0.1-0.5 ms"), (2.0, "0.5-2 ms"), (10.0, "2-10 ms"),
           (float("inf"), ">= 10 ms"))


def short(name: str) -> str:
    name = re.sub(r"^void ", "", name)
    return re.split(r"[<(]", name, maxsplit=1)[0][:44]


def main() -> int:
    root = sys.argv[1]
    pattern = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"ncclDevKernel")
    files = glob.glob(os.path.join(root, "**", "*kernel_trace.csv"), recursive=True)
    if not files:
        print(f"NO_TRACE under {root}")
        return 1
    rows = collections.defaultdict(list)
    for path in files:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["Agent_Id"]].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
    marks = [(s, e) for v in rows.values() for s, e, n in v if pattern.search(n)]
    if not marks:
        print(f"no kernel matching {pattern.pattern!r}")
        return 1
    t0, t1 = min(s for s, _ in marks), max(e for _, e in marks)
    for agent in sorted(rows):
        kernels = sorted(k for k in rows[agent] if k[1] > t0 and k[0] < t1)
        if not any(pattern.search(k[2]) for k in kernels):
            continue
        by_size = collections.Counter()
        n_size = collections.Counter()
        by_prev = collections.Counter()
        n_prev = collections.Counter()
        cursor, prev = kernels[0][1], kernels[0][2]
        for s, e, name in kernels[1:]:
            if s > cursor:
                gap = (s - cursor) / 1e6
                label = next(lbl for limit, lbl in BUCKETS if gap < limit)
                by_size[label] += gap
                n_size[label] += 1
                by_prev[short(prev)] += gap
                n_prev[short(prev)] += 1
            if e > cursor:
                cursor, prev = e, name
        idle = sum(by_size.values())
        print(f"{agent}: idle {idle / 1e3:.2f} s of a {(t1 - t0) / 1e9:.2f} s window in {sum(n_size.values())} gaps")
        print("  by size:   " + "; ".join(f"{lbl} {by_size[lbl] / 1e3:.2f} s ({n_size[lbl]})" for _, lbl in BUCKETS if n_size[lbl]))
        print("  after:")
        for name, ms in by_prev.most_common(8):
            print(f"    {ms / 1e3:6.2f} s  {n_prev[name]:>7} gaps  mean {ms / n_prev[name] * 1e3:7.1f} us  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
