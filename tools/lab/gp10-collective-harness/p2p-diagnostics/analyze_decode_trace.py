#!/usr/bin/env python3
"""GP11 re-profile: union-of-spans decode analysis of a rocprofv3 kernel trace.

Uses UNION of timestamp spans, never summed durations -- summing double-counts
whenever both GPUs run the same collective simultaneously, which they do.
"""
import csv, sys, re
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gp11prof/ktrace_kernel_trace.csv"

rows = []
with open(PATH, newline="") as f:
    r = csv.DictReader(f)
    for d in r:
        rows.append((int(d["Start_Timestamp"]), int(d["End_Timestamp"]),
                     d["Kernel_Name"], d["Agent_Id"]))
print(f"loaded {len(rows)} dispatches")
rows.sort(key=lambda x: x[0])

t0_all, t1_all = rows[0][0], max(r[1] for r in rows)
span_ns = t1_all - t0_all
print(f"full trace span: {span_ns/1e9:.3f}s")

# --- find the decode window by kernel-launch density per 100ms bin ---
BIN = 100_000_000  # 100ms
bins = defaultdict(int)
for s, e, n, a in rows:
    bins[(s - t0_all) // BIN] += 1
if not bins:
    sys.exit("no data")
peak = max(bins.values())
# decode = contiguous run of bins with >20% of peak density
active = sorted(b for b, c in bins.items() if c > peak * 0.20)
# take the longest contiguous run
best = (0, 0); cur_start = active[0]; prev = active[0]
for b in active[1:]:
    if b != prev + 1:
        if prev - cur_start > best[1] - best[0]:
            best = (cur_start, prev)
        cur_start = b
    prev = b
if prev - cur_start > best[1] - best[0]:
    best = (cur_start, prev)
w0 = t0_all + best[0] * BIN
w1 = t0_all + (best[1] + 1) * BIN
print(f"decode window: bins {best[0]}..{best[1]} = {(w1-w0)/1e9:.3f}s")

win = [r for r in rows if r[0] >= w0 and r[1] <= w1]
print(f"dispatches in window: {len(win)}")
decode_wall = w1 - w0


def classify(name):
    n = name.lower()
    if "ggml_cuda_ar_" in n or "allreduce" in n:      return "AllReduce (internal)"
    if "nccl" in n or "rccl" in n:                     return "AllReduce (RCCL)"
    if "mul_mat_vec_q" in n or "mmvq" in n:            return "MMVQ (matvec quant)"
    if "mul_mat_vec" in n or "mmvf" in n:              return "MMVF (matvec float)"
    if "mul_mat_q" in n or "mmq" in n:                 return "MMQ (matmul quant)"
    if "gemm" in n or "mul_mat" in n:                  return "other matmul"
    if "flash" in n or "fattn" in n:                   return "flash-attn"
    if "rms_norm" in n or "norm" in n:                 return "norm"
    if "rope" in n:                                    return "rope"
    if "cpy" in n or "copy" in n or "dup" in n:        return "copy/cast"
    if "soft_max" in n:                                return "softmax"
    if "bin_bcast" in n or "add" in n or "mul" in n:   return "elementwise"
    if "gated_delta" in n or "ssm" in n or "conv" in n:return "GDN/SSM"
    if "argsort" in n or "sample" in n or "topk" in n: return "sampling"
    return "other: " + name.split("(")[0][:44]


def union_ns(spans):
    if not spans: return 0
    spans = sorted(spans)
    tot = 0; cs, ce = spans[0]
    for s, e in spans[1:]:
        if s > ce:
            tot += ce - cs; cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ce - cs


fam_spans = defaultdict(list)
fam_count = defaultdict(int)
fam_sum   = defaultdict(int)
for s, e, n, a in win:
    f = classify(n)
    fam_spans[f].append((s, e))
    fam_count[f] += 1
    fam_sum[f] += e - s

all_busy = union_ns([(s, e) for s, e, _, _ in win])
print(f"\ndecode wall: {decode_wall/1e6:.1f}ms   GPU busy (union): "
      f"{all_busy/1e6:.1f}ms = {100*all_busy/decode_wall:.1f}%\n")

print(f"{'family':<32}{'union ms':>10}{'% wall':>9}{'summed ms':>11}{'count':>9}")
print("-" * 71)
res = sorted(fam_spans.items(), key=lambda kv: -union_ns(kv[1]))
for f, sp in res:
    u = union_ns(sp)
    print(f"{f:<32}{u/1e6:>10.1f}{100*u/decode_wall:>8.1f}%{fam_sum[f]/1e6:>11.1f}{fam_count[f]:>9}")

# --- overlap of AllReduce with matmul: is comms hidden behind compute? ---
ar = [sp for f, sp in fam_spans.items() if f.startswith("AllReduce") for sp in sp]
mm = [sp for f, sp in fam_spans.items()
      if f.startswith(("MMVQ", "MMVF", "MMQ", "other matmul")) for sp in sp]
if ar:
    ar_u = union_ns(ar)
    both = union_ns(ar) + union_ns(mm) - union_ns(ar + mm)
    print(f"\nAllReduce union: {ar_u/1e6:.1f}ms = {100*ar_u/decode_wall:.1f}% of decode wall")
    print(f"AllReduce overlapping any matmul: {both/1e6:.1f}ms "
          f"({100*both/ar_u:.1f}% of AllReduce time)")
    print(f"AllReduce EXCLUSIVE (pure comms, no compute): "
          f"{(ar_u-both)/1e6:.1f}ms = {100*(ar_u-both)/decode_wall:.1f}% of decode wall")

gaps = decode_wall - all_busy
print(f"\nGPU IDLE within decode window: {gaps/1e6:.1f}ms = {100*gaps/decode_wall:.1f}% "
      f"(launch gaps / host-side stalls)")
