#!/usr/bin/env python3
"""QFP48 / QFP49: what a card launches between two cross-card sums, from a rocprofv3 kernel trace.

A "step" is the run of kernels on a card between one sum kernel and the next. Staggering hides little of the sum
once a step is many kernels (tools/lab/hip-probes: 84% at 2 kernels, 42% at 40, 13% at 80), and every launch leaves
a gap, so the number of kernels a step is the thing to bring down. This lists, per card:

  kernels a step       distribution of the launch count
  by kernel            launches a step, time a launch, share of the step's kernel time, and the gap left after it
  by adjacent pair     the commonest "A then B" sequences: what a fusion would join

Usage: kernel-step-census.py <rocprof dir> [sum kernel pattern, default ncclDevKernel|bc_cpu_root] [top N, default 14]
"""
from __future__ import annotations

import collections
import csv
import glob
import os
import re
import statistics
import sys


def short(name: str) -> str:
    name = re.sub(r"^void ", "", name)
    return re.split(r"[<(]", name, maxsplit=1)[0][:40]


def main() -> int:
    root = sys.argv[1]
    pattern = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"ncclDevKernel|bc_cpu_root")
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    files = glob.glob(os.path.join(root, "**", "*kernel_trace.csv"), recursive=True)
    if not files:
        print(f"NO_TRACE under {root}")
        return 1
    rows = collections.defaultdict(list)
    for path in files:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["Agent_Id"]].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
    us = 1e-3
    for agent in sorted(rows):
        kernels = sorted(rows[agent])
        if not any(pattern.search(k[2]) for k in kernels):
            continue
        first = next(i for i, k in enumerate(kernels) if pattern.search(k[2]))
        steps, current = [], []
        for k in kernels[first + 1:]:
            if pattern.search(k[2]):
                if current:
                    steps.append(current)
                current = []
            else:
                current.append(k)
        steps = [s for s in steps if s]
        if not steps:
            continue
        counts = [len(s) for s in steps]
        busy = [sum(e - b for b, e, _ in s) for s in steps]
        span = [s[-1][1] - s[0][0] for s in steps]
        print(f"\n{agent}: {len(steps)} steps; kernels a step median {statistics.median(counts):.0f}, mean "
              f"{statistics.mean(counts):.1f}, max {max(counts)}; a step runs {statistics.mean(busy) * us:.0f} us of kernels "
              f"in {statistics.mean(span) * us:.0f} us ({100.0 * sum(busy) / max(sum(span), 1):.0f}% of the step is kernel time)")
        # the totals above include one-off waits (first-use code object loads, the wait for the next prompt batch);
        # the median step is the steady state
        share = sorted(b / sp for b, sp in zip(busy, span) if sp > 0)
        print(f"  median step: {statistics.median(busy) * us:.0f} us of kernels in {statistics.median(span) * us:.0f} us; "
              f"kernel share of a step median {100 * share[len(share) // 2]:.0f}%, 10th percentile {100 * share[len(share) // 10]:.0f}%; "
              f"the {max(1, len(span) // 100)} longest steps (1%) hold {100.0 * sum(sorted(span)[-max(1, len(span) // 100):]) / sum(span):.0f}% of all step time")
        n_launch = collections.Counter()
        t_kernel = collections.Counter()
        t_gap = collections.Counter()
        pairs = collections.Counter()
        pair_gap = collections.Counter()
        pair_gaps = collections.defaultdict(list)
        for s in steps:
            for i, (b, e, name) in enumerate(s):
                key = short(name)
                n_launch[key] += 1
                t_kernel[key] += e - b
                if i + 1 < len(s):
                    t_gap[key] += max(0, s[i + 1][0] - e)
                    pairs[(key, short(s[i + 1][2]))] += 1
                    pair_gap[(key, short(s[i + 1][2]))] += max(0, s[i + 1][0] - e)
                    pair_gaps[(key, short(s[i + 1][2]))].append(max(0, s[i + 1][0] - e))
        total = sum(t_kernel.values())
        print("  launches a step   us a launch   share of kernel time   gap after, us   kernel")
        for key, t in t_kernel.most_common(top):
            n = n_launch[key]
            print(f"  {n / len(steps):15.2f}   {t * us / n:11.1f}   {100.0 * t / total:19.1f}%   {t_gap[key] * us / n:13.1f}   {key}")
        print("  commonest adjacent pairs:  a step   gap between them, us")
        for (a, b), n in pairs.most_common(top):
            print(f"  {n / len(steps):8.2f}   {pair_gap[(a, b)] * us / n:8.1f}   {a}  ->  {b}")
        # ranked by median gap x occurrences: a pair's mean can be a few first-use gaps (a code object load of
        # hundreds of ms after the first mul_mat_q -> convert made a 5 us pair look like 239 us, 2026-10-12)
        print("  pairs with the most steady gap time:  a step   median gap, us   us a step   (mean gap, us; largest single gap, ms)")
        steady = {pair: statistics.median(g) * len(g) for pair, g in pair_gaps.items()}
        for (a, b), g in sorted(steady.items(), key=lambda kv: -kv[1])[:top]:
            gaps = pair_gaps[(a, b)]
            print(f"  {len(gaps) / len(steps):8.2f}   {statistics.median(gaps) * us:14.1f}   {g * us / len(steps):9.1f}   "
                  f"({pair_gap[(a, b)] * us / len(gaps):.1f}; {max(gaps) * us / 1e3:.1f})   {a}  ->  {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
