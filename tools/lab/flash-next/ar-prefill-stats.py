#!/usr/bin/env python3
"""QFP41: what the large (RCCL) AllReduce costs in prefill, from a rocprofv3 kernel trace.

In prefill the cross-card sums are too large for the cpu-root path and go to RCCL, whose device kernel
(ncclDevKernel_*) sits on the card's compute stream until the exchange through host memory is done. This reports,
over the window from the first to the last such kernel (the prefill, without model load):

  per card   the number of sums, their total, mean and median time, the card's kernel time in the window, its idle
             time (no kernel running) and the share of the window each takes;
  per sum    aligned by ordinal across cards: skew (latest start - earliest start: a card arriving late makes the
             others wait inside the kernel) and the time from the latest start to the latest end (the exchange
             itself once every card has arrived).

Idle time is the part an overlapped transfer cannot recover on its own: it is the card waiting for the host.

Usage: ar-prefill-stats.py <rocprof dir> [kernel name pattern, default ncclDevKernel]
"""
from __future__ import annotations

import collections
import csv
import glob
import os
import re
import statistics
import sys


def main() -> int:
    root = sys.argv[1]
    pattern = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"ncclDevKernel")
    files = glob.glob(os.path.join(root, "**", "*kernel_trace.csv"), recursive=True)
    if not files:
        print(f"NO_TRACE under {root}")
        return 1
    rows = collections.defaultdict(list)  # agent -> [(start, end, is_ar)]
    for path in files:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["Agent_Id"]].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]),
                                            bool(pattern.search(r["Kernel_Name"]))))
    ars = {a: sorted((s, e) for s, e, is_ar in v if is_ar) for a, v in rows.items()}
    ars = {a: v for a, v in ars.items() if v}
    if not ars:
        print(f"no kernel matching {pattern.pattern!r}")
        return 1
    t0, t1 = min(v[0][0] for v in ars.values()), max(v[-1][1] for v in ars.values())
    window = (t1 - t0) / 1e9
    print(f"window (first to last sum): {window:.2f} s over {len(ars)} card(s)")
    print(f"{'card':<10} {'sums':>6} {'total s':>8} {'mean ms':>8} {'median':>8} {'kernel s':>9} {'idle s':>7} "
          f"{'sum %':>6} {'other %':>8} {'idle %':>7}")
    for agent in sorted(ars):
        durations = [(e - s) / 1e6 for s, e in ars[agent]]
        inside = sorted((max(s, t0), min(e, t1)) for s, e, _ in rows[agent] if e > t0 and s < t1)
        busy, cursor = 0, t0  # union of kernel intervals: kernels on other streams may overlap
        for s, e in inside:
            if e > cursor:
                busy += e - max(s, cursor)
                cursor = e
        busy /= 1e9
        total = sum(durations) / 1e3
        print(f"{agent:<10} {len(durations):>6} {total:>8.2f} {statistics.mean(durations):>8.2f} "
              f"{statistics.median(durations):>8.2f} {busy:>9.2f} {window - busy:>7.2f} "
              f"{100 * total / window:>6.1f} {100 * (busy - total) / window:>8.1f} {100 * (window - busy) / window:>7.1f}")
    n = min(len(v) for v in ars.values())
    if len({len(v) for v in ars.values()}) == 1 and len(ars) > 1:
        skew = [(max(v[i][0] for v in ars.values()) - min(v[i][0] for v in ars.values())) / 1e6 for i in range(n)]
        exchange = [(max(v[i][1] for v in ars.values()) - max(v[i][0] for v in ars.values())) / 1e6 for i in range(n)]
        print(f"per sum over {n}: skew mean {statistics.mean(skew):.2f} ms (median {statistics.median(skew):.2f}, "
              f"total {sum(skew) / 1e3:.2f} s); exchange after the last card arrives mean "
              f"{statistics.mean(exchange):.2f} ms (median {statistics.median(exchange):.2f}, "
              f"total {sum(exchange) / 1e3:.2f} s)")
    else:
        print(f"cards ran different numbers of sums ({ {a: len(v) for a, v in ars.items()} }): no per-sum alignment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
