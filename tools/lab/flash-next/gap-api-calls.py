#!/usr/bin/env python3
"""QFP48: what the host does in the gap between two kernels, from a rocprofv3 kernel trace + HIP API trace.

kernel-step-census.py shows which adjacent kernel pairs leave a card idle (e.g. mul_mat_q -> convert_unary_cont_vec4,
239 us). This says why: for every occurrence of kernel A directly followed by kernel B on a card, it takes the host
interval from the end of the HIP call that launched A to the start of the HIP call that launched B (matched by
correlation id) and lists the HIP API calls the launching thread made in it, plus the time in no HIP call at all
(host code: graph walking, allocation bookkeeping, a lock).

Usage: gap-api-calls.py <rocprof dir> <kernel A regex> <kernel B regex> [min device gap us, default 50]
Needs a trace taken with --kernel-trace --hip-runtime-trace (long-ctx-profile.sh prefillapi).
"""
from __future__ import annotations

import bisect
import collections
import csv
import glob
import os
import re
import sys


def load(root, suffix):
    files = glob.glob(os.path.join(root, "**", "*" + suffix), recursive=True)
    if not files:
        sys.exit("no %s under %s" % (suffix, root))
    rows = []
    for path in files:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    root, pat_a, pat_b = sys.argv[1], re.compile(sys.argv[2]), re.compile(sys.argv[3])
    min_gap = float(sys.argv[4]) if len(sys.argv) > 4 else 50.0

    api = load(root, "hip_api_trace.csv")
    by_corr = {}
    by_thread = collections.defaultdict(list)
    for r in api:
        s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        by_corr[r["Correlation_Id"]] = (s, e, r["Thread_Id"])
        by_thread[r["Thread_Id"]].append((s, e, r["Function"]))
    starts = {}
    for tid, calls in by_thread.items():
        calls.sort()
        starts[tid] = [c[0] for c in calls]

    per_agent = collections.defaultdict(list)
    for r in load(root, "kernel_trace.csv"):
        per_agent[r["Agent_Id"]].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"],
                                         r["Correlation_Id"]))

    for agent in sorted(per_agent):
        kernels = sorted(per_agent[agent])
        n = 0
        dev_gap = host_gap = in_api = 0
        funcs = collections.defaultdict(lambda: [0, 0])
        worst = []
        gaps = []
        for a, b in zip(kernels, kernels[1:]):
            if not (pat_a.search(a[2]) and pat_b.search(b[2])):
                continue
            gap = (b[0] - a[1]) / 1e3
            if gap < min_gap or a[3] not in by_corr or b[3] not in by_corr:
                continue
            launch_a, launch_b = by_corr[a[3]], by_corr[b[3]]
            lo, hi, tid = launch_a[1], launch_b[0], launch_b[2]
            if hi <= lo:
                continue
            n += 1
            dev_gap += gap
            host_gap += (hi - lo) / 1e3
            calls = by_thread[tid]
            i = bisect.bisect_left(starts[tid], lo)
            spent = 0
            detail = []
            while i < len(calls) and calls[i][0] < hi:
                s, e, fn = calls[i]
                d = (min(e, hi) - s) / 1e3
                funcs[fn][0] += 1
                funcs[fn][1] += d
                spent += d
                detail.append((d, fn))
                i += 1
            in_api += spent
            worst.append(((hi - lo) / 1e3, gap, sorted(detail, reverse=True)[:4]))
            gaps.append(gap)
        if not n:
            print("%s: no %s -> %s with a gap of %g us or more" % (agent, pat_a.pattern, pat_b.pattern, min_gap))
            continue
        print("%s: %d occurrences of %s -> %s; card idle %.1f us, host interval between the two launches %.1f us "
              "(%.1f us in HIP calls, %.1f us in none)" % (agent, n, pat_a.pattern, pat_b.pattern, dev_gap / n,
                                                           host_gap / n, in_api / n, (host_gap - in_api) / n))
        gaps.sort()
        # a few first-use gaps (code object loads of hundreds of ms) can carry the whole mean: show the spread
        print("  card idle: median %.1f us, 90th percentile %.1f us, max %.1f us; the 3 longest are %.0f%% of the total"
              % (gaps[len(gaps) // 2], gaps[int(len(gaps) * 0.9)], gaps[-1], 100 * sum(gaps[-3:]) / sum(gaps)))
        print("  calls an occurrence   us an occurrence   us a call   HIP function")
        for fn, (c, us) in sorted(funcs.items(), key=lambda kv: -kv[1][1])[:12]:
            print("  %19.2f   %16.1f   %9.1f   %s" % (c / n, us / n, us / c, fn))
        worst.sort(reverse=True)
        print("  longest three (host us, card idle us, largest calls):")
        for host, gap, detail in worst[:3]:
            print("    %8.1f %8.1f  %s" % (host, gap, ", ".join("%s %.0f" % (fn, d) for d, fn in detail)))


if __name__ == "__main__":
    main()
