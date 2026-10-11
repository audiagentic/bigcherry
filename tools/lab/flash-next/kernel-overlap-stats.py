#!/usr/bin/env python3
"""QFP49: is the cross-card sum really running under compute? From a rocprofv3 kernel trace.

Every kernel in the trace has a card, a queue, a stream and a start and end time. For each card, over the window
from the first to the last sum kernel, this splits the wall time into

  compute only     a model kernel is running and no sum kernel is
  sum only         a sum kernel is running and no model kernel is: the exchange the card waits out
  sum + compute    both at once: what staggering is meant to create (0 in a sequential run by construction)
  idle             nothing is running on the card

and lists, per queue / stream of the card, how many kernels ran there and for how long - so it shows whether the
sums are on a stream of their own. It then gives the sums' count, mean and total, and the share of sum time that
was covered by compute.

Read with care: a kernel's interval is from its start to its end on the card. Two kernels overlapping in time are
both resident; the trace does not say how the card divided its compute units between them.

Usage: kernel-overlap-stats.py <rocprof dir> [sum kernel pattern, default ncclDevKernel|bc_cpu_root]
"""
from __future__ import annotations

import collections
import csv
import glob
import os
import re
import sys


def union(intervals):
    """Merged, sorted list of [start, end)."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def total(merged):
    return sum(e - s for s, e in merged)


def intersect(a, b):
    """Total length of the intersection of two merged interval lists."""
    i = j = 0
    acc = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            acc += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return acc


def main() -> int:
    root = sys.argv[1]
    pattern = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"ncclDevKernel|bc_cpu_root")
    files = glob.glob(os.path.join(root, "**", "*kernel_trace.csv"), recursive=True)
    if not files:
        print(f"NO_TRACE under {root}")
        return 1
    rows = collections.defaultdict(list)
    for path in files:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["Agent_Id"]].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"],
                                            r.get("Queue_Id", "?"), r.get("Stream_Id", "?")))
    marks = [(s, e) for v in rows.values() for s, e, n, _, _ in v if pattern.search(n)]
    if not marks:
        print(f"no kernel matching {pattern.pattern!r}")
        return 1
    t0, t1 = min(s for s, _ in marks), max(e for _, e in marks)
    ms = 1e-6
    print(f"window {((t1 - t0) * ms):.0f} ms, first to last sum kernel")
    for agent in sorted(rows):
        kernels = [(max(s, t0), min(e, t1), n, q, st) for s, e, n, q, st in rows[agent] if e > t0 and s < t1]
        sums = [(s, e) for s, e, n, _, _ in kernels if pattern.search(n)]
        if not sums:
            continue
        compute = union((s, e) for s, e, n, _, _ in kernels if not pattern.search(n))
        summed = union(sums)
        both = intersect(compute, summed)
        t_compute, t_sum = total(compute), total(summed)
        anything = total(union([(s, e) for s, e, *_ in kernels]))
        span = t1 - t0
        print(f"\n{agent}: {len(kernels)} kernels")
        for label, value in (("compute only", t_compute - both), ("sum only", t_sum - both), ("sum + compute", both),
                             ("idle", span - anything)):
            print(f"  {label:14} {value * ms:9.0f} ms  {100.0 * value / span:5.1f}%")
        print(f"  sums: {len(sums)}, mean {t_sum * ms / max(len(sums), 1):.3f} ms resident (union {t_sum * ms:.0f} ms); "
              f"{100.0 * both / max(t_sum, 1):.1f}% of sum time has compute running under it")
        per = collections.defaultdict(lambda: [0, 0, 0])
        for s, e, n, q, st in kernels:
            slot = per[(q, st)]
            slot[0] += 1
            slot[1] += e - s
            slot[2] += 1 if pattern.search(n) else 0
        for (q, st), (count, busy, n_sums) in sorted(per.items(), key=lambda kv: -kv[1][1])[:6]:
            print(f"    queue {q} stream {st}: {count} kernels ({n_sums} sums), {busy * ms:.0f} ms of kernel time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
