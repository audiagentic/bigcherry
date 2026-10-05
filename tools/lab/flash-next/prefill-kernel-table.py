#!/usr/bin/env python3
"""Per-kernel time table from a rocprofv3 --kernel-trace CSV directory: total GPU kernel time per device, grouped by
kernel family, with each family's share of that device's busy time. Usage: prefill-kernel-table.py <rocprof dir>"""
import collections
import csv
import glob
import os
import re
import sys

FAMILIES = (
    ("flash attention", r"flash_attn"),
    ("lightning indexer / top-k", r"indexer|top_k|topk"),
    ("MoE MMQ (mul_mat_id)", r"mul_mat_q|mmq"),
    ("MMVQ / vec dot", r"mul_mat_vec|mmvq|vec_dot"),
    ("quantize", r"quantize"),
    ("GDN / SSM / recurrent", r"gdn|ssm|delta|gated"),
    ("all-reduce / copies", r"allreduce|all_reduce|reduce|memcpy|copy|cpy"),
    ("norm / activation / elementwise", r"norm|silu|gelu|sigmoid|unary|bin_bcast|scale|add|mul"),
    ("set / get rows, mask build", r"set_rows|get_rows|fill|repeat|concat|pad"),
)


def family(name: str) -> str:
    low = name.lower()
    for label, pattern in FAMILIES:
        if re.search(pattern, low):
            return label
    return "other"


def main() -> int:
    traces = sorted(glob.glob(os.path.join(sys.argv[1], "**", "*kernel_trace.csv"), recursive=True))
    if not traces:
        print(f"no *kernel_trace.csv under {sys.argv[1]}")
        return 1
    per_dev = collections.defaultdict(lambda: collections.Counter())
    names = collections.defaultdict(lambda: collections.Counter())
    span = {}
    for path in traces:
        with open(path, newline="", errors="replace") as handle:
            for row in csv.DictReader(handle):
                start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
                dev = row.get("Agent_Id") or row.get("Queue_Id") or "?"
                fam = family(row["Kernel_Name"])
                per_dev[dev][fam] += end - start
                names[fam][row["Kernel_Name"].split("(")[0][:70]] += end - start
                lo, hi = span.get(dev, (start, end))
                span[dev] = (min(lo, start), max(hi, end))
    for dev in sorted(per_dev):
        busy = sum(per_dev[dev].values())
        wall = span[dev][1] - span[dev][0]
        print(f"device {dev}: kernel time {busy/1e9:.2f} s over a {wall/1e9:.2f} s span ({100*busy/max(1, wall):.0f}% busy)")
        for fam, t in per_dev[dev].most_common():
            print(f"  {100*t/busy:5.1f}%  {t/1e9:7.2f} s  {fam}")
    print("top kernels per family (all devices):")
    for fam in sorted(names, key=lambda f: -sum(names[f].values())):
        top = ", ".join(f"{n} {t/1e9:.2f}s" for n, t in names[fam].most_common(3))
        print(f"  {fam}: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
