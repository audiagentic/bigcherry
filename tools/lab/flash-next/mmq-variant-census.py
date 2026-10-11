#!/usr/bin/env python3
"""QFP48: which steps and which quantized matmuls carry prefill, from a rocprofv3 kernel trace.

Two tables for the first card that has sum kernels (or the agent given):

  step kinds     every step (the kernels between two cross-card sums) classified by what it contains - attention,
                 recurrent (GDN / SSM), MoE, plain feed-forward, other - with how many there are, kernels a step,
                 kernel time, wall span, and the share of all step time. This ranks which "megakernel" would matter.
  mul_mat_q      every variant of the quantized matmul kernel (template arguments from its name, plus launch grid
                 and workgroup), with launches, time a launch, and share of mul_mat_q time and of all kernel time.

Usage: mmq-variant-census.py <rocprof dir> [agent, e.g. "Agent 1"] [sum kernel pattern]
"""
from __future__ import annotations

import collections
import csv
import glob
import os
import re
import sys


def main() -> int:
    root = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    pattern = re.compile(sys.argv[3] if len(sys.argv) > 3 else r"ncclDevKernel|bc_cpu_root")
    files = glob.glob(os.path.join(root, "**", "*kernel_trace.csv"), recursive=True)
    if not files:
        print(f"NO_TRACE under {root}")
        return 1
    rows = collections.defaultdict(list)
    for path in files:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["Agent_Id"]].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"],
                                            (r["Grid_Size_X"], r["Grid_Size_Y"], r["Grid_Size_Z"]),
                                            (r["Workgroup_Size_X"], r["Workgroup_Size_Y"], r["Workgroup_Size_Z"])))
    agents = [a for a in sorted(rows) if any(pattern.search(k[2]) for k in rows[a])]
    agent = want if want in rows else agents[0]
    kernels = sorted(rows[agent])
    us = 1e-3

    first = next(i for i, k in enumerate(kernels) if pattern.search(k[2]))
    steps, current = [], []
    for k in kernels[first + 1:]:
        if pattern.search(k[2]):
            if current:
                steps.append(current)
            current = []
        else:
            current.append(k)

    def kind(step):
        names = " ".join(k[2] for k in step)
        if "flash_attn" in names:
            return "attention"
        if "gated_delta_net" in names or "gdn_" in names or "ssm_conv" in names:
            return "recurrent"
        if "moe" in names or "mm_ids" in names:
            return "MoE feed-forward"
        if "mul_mat_q" in names:
            return "matmul, no MoE"
        return "other"

    by_kind = collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # steps, kernels, kernel time, span, mmq time
    for s in steps:
        slot = by_kind[kind(s)]
        slot[0] += 1
        slot[1] += len(s)
        slot[2] += sum(e - b for b, e, *_ in s)
        slot[3] += s[-1][1] - s[0][0]
        slot[4] += sum(e - b for b, e, n, *_ in s if "mul_mat_q<" in n)
    span_all = sum(v[3] for v in by_kind.values()) or 1
    print(f"{agent}: {len(steps)} steps")
    print("  step kind            steps   kernels a step   kernel us   span us   mul_mat_q us   share of step time")
    for name, (n, nk, tk, sp, tm) in sorted(by_kind.items(), key=lambda kv: -kv[1][3]):
        print(f"  {name:18} {n:7d}   {nk / n:14.1f}   {tk * us / n:9.0f}   {sp * us / n:7.0f}   {tm * us / n:12.0f}   "
              f"{100.0 * sp / span_all:17.1f}%")

    def short(name):
        return re.split(r"[<(]", re.sub(r"^void ", "", name), maxsplit=1)[0][:38]

    print()
    print("  where the time without a kernel is, by step kind (the four largest adjacent pairs each):")
    for name in sorted(by_kind, key=lambda k: -by_kind[k][3]):
        gaps = collections.Counter()
        count = collections.Counter()
        n_steps = 0
        for s in steps:
            if kind(s) != name:
                continue
            n_steps += 1
            for i in range(len(s) - 1):
                pair = (short(s[i][2]), short(s[i + 1][2]))
                gaps[pair] += max(0, s[i + 1][0] - s[i][1])
                count[pair] += 1
        for pair, g in gaps.most_common(4):
            print(f"  {name:18} {g * us / n_steps:8.0f} us a step   {count[pair] / n_steps:5.2f} a step x {g * us / count[pair]:7.0f} us   "
                  f"{pair[0]}  ->  {pair[1]}")

    total_kernel = sum(e - b for b, e, n, *_ in kernels[first:] if not pattern.search(n)) or 1
    variants = collections.defaultdict(lambda: [0, 0])
    for b, e, n, grid, wg in kernels[first:]:
        if "mul_mat_q<" not in n:
            continue
        args = re.search(r"mul_mat_q<([^>]*)>", n)
        key = (args.group(1) if args else n[:40], "x".join(grid), "x".join(wg))
        variants[key][0] += 1
        variants[key][1] += e - b
    total_mmq = sum(v[1] for v in variants.values()) or 1
    print(f"\n  mul_mat_q: {sum(v[0] for v in variants.values())} launches, {total_mmq * 1e-6:.0f} ms, "
          f"{100.0 * total_mmq / total_kernel:.1f}% of kernel time; {len(variants)} variants")
    print("  template arguments                       grid          workgroup   launches   us a launch   of mul_mat_q   of all kernels")
    for (args, grid, wg), (n, t) in sorted(variants.items(), key=lambda kv: -kv[1][1])[:16]:
        print(f"  {args[:40]:40} {grid:13} {wg:10} {n:9d}   {t * us / n:11.1f}   {100.0 * t / total_mmq:10.1f}%   "
              f"{100.0 * t / total_kernel:13.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
