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
    ("all-reduce / collectives", r"nccldevkernel|bc_cpu_root|bc_ar_|allreduce|all_reduce"),
    ("flash attention", r"flash_attn"),
    ("QSA lightning indexer / top-k", r"indexer|top_k_radix|lightning"),
    ("MoE routing", r"topk_moe|moe_block_map"),
    ("MMVQ (mul_mat_vec_q)", r"mul_mat_vec_q"),
    ("MMQ (mul_mat_q, incl. MoE)", r"mul_mat_q|quantize_mmq"),
    ("float matmul (mul_mat_f / mul_mat_vec_f / BLAS)", r"mul_mat_f|mul_mat_vec_f|cijk_"),
    ("quantize Q8_1", r"quantize"),
    ("GDN / SSM / recurrent", r"gated_delta|gdn|ssm_"),
    ("copies", r"cpy|copybuffer|memcpy|fillbuffer"),
    ("set / get rows, concat, mask build", r"set_rows|get_rows|k_fill|repeat|concat|pad"),
    ("norm / activation / elementwise", r"norm|silu|gelu|sigmoid|unary|bin_bcast|scale|k_add|k_mul|hc_pre"),
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
