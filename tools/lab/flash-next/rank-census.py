#!/usr/bin/env python3
"""Per-GPU decode census by op class (QFP09): which kernel classes make the slow rank slow.

Reads a long-ctx-profile.sh decode-mode run (rocprof kernel trace + decode.timings.json), keeps the decode window,
and prints per agent: busy ms/token, kernel count/token and time per op class, so a per-class split skew (like 1303
for attention) can target the class that dominates the R9700.

Usage: rank-census.py <run-dir>      (expects <run-dir>/rocprof/**/*kernel_trace.csv and decode.timings.json)
"""
import collections
import csv
import glob
import json
import re
import sys

CLASSES = [
    ("allreduce", r"allreduce|all_reduce|cpu_root|rccl|nccl|ncclDevKernel"),
    ("flash-attn", r"flash_attn|fattn"),
    ("qsa/indexer/topk", r"top_?k|argsort|qsa|indexer|radix"),
    ("moe-mmvq(id)", r"mul_mat_vec_q.*(moe|ids)|mmvq_moe|mul_mat_id"),
    ("mmvq", r"mul_mat_vec_q|mmvq"),
    ("mmq/gemm", r"mul_mat_q|mmq|gemm|Cijk|wmma"),
    ("mmvf/f16-matvec", r"mul_mat_vec_f|mmvf"),
    ("gdn/ssm", r"gated_delta|delta_net|ssm|conv1d|rwkv|scan"),
    ("quantize", r"quantize"),
    ("norm/rope", r"norm|rope"),
    ("get/set_rows/cpy", r"get_rows|set_rows|cpy|convert|dup"),
    ("softmax/topk-weights", r"soft_max|softmax"),
    ("elementwise", r"k_bin_bcast|unary|silu|sigmoid|glu|swiglu|add|mul|scale|fill|clamp|concat|sum"),
]


def classify(name):
    low = name.lower()
    for cls, pat in CLASSES:
        if re.search(pat, low):
            return cls
    return "other"


def main():
    run = sys.argv[1]
    t = json.load(open(f"{run}/decode.timings.json"))[-1]
    n_tok = t["predicted_n"]
    win_ns = n_tok / t["predicted_per_second"] * 1e9
    path = sorted(glob.glob(f"{run}/rocprof/**/*kernel_trace.csv", recursive=True))[0]
    rows = list(csv.DictReader(open(path)))
    end = max(int(r["End_Timestamp"]) for r in rows)
    agent_key = "Agent_Id" if "Agent_Id" in rows[0] else "Gpu_Id"
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in rows:
        s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        if s < end - win_ns:
            continue
        c = per[r[agent_key]][classify(r["Kernel_Name"])]
        c[0] += 1
        c[1] += e - s
    print(f"decode window {win_ns / 1e9:.2f} s, {n_tok} tokens ({t['predicted_per_second']:.1f} t/s); "
          f"wall {win_ns / 1e6 / n_tok:.3f} ms/token")
    for agent in sorted(per, key=lambda a: -sum(v[1] for v in per[a].values())):
        cls = per[agent]
        busy = sum(v[1] for v in cls.values())
        launches = sum(v[0] for v in cls.values())
        print(f"\nagent {agent}: busy {busy / 1e6 / n_tok:.3f} ms/token ({100 * busy / win_ns:.0f}% of wall), "
              f"{launches / n_tok:.0f} kernels/token")
        for name, (n, ns) in sorted(cls.items(), key=lambda kv: -kv[1][1]):
            print(f"  {100 * ns / busy:5.1f}%  {ns / 1e6 / n_tok:7.3f} ms/tok  {n / n_tok:7.1f} k/tok  {name}")


if __name__ == "__main__":
    main()
