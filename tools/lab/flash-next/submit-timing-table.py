#!/usr/bin/env python3
"""Where a prefill chunk's wall time goes on the host, from a server log taken with BIGCHERRY_SUBMIT_TIMING=1.

Reads the lines of 1319 (BIGCHERRY_SUBMIT_TIMING: one per process_ubatch - graph build, set_inputs, graph_compute host
time), 1320 (BIGCHERRY_META_TIMING: the meta backend inside graph_compute - subgraph rebuild, launches, AllReduce
enqueue) and 1325 (BIGCHERRY_SCHED_SPLIT: every scheduler split - input handling and compute host time per backend),
plus 1346's BIGCHERRY_MTP_PROMPT_TIMING line when present. Only calls with at least --min-tokens tokens are prefill
chunks; the contexts are told apart by their pointer (the one with the most prefill chunks is the target).

compute_us of a GPU split is the host time to SUBMIT the work (the device runs on asynchronously); input_us of a split
includes the synchronise that waits for the device work the split's inputs depend on, so the GPU's compute time shows
up there and in the explicit sync, not in compute_us. The CPU backend's compute_us is real host work.

Usage: submit-timing-table.py <server.log> [--min-tokens 256] [--wall-s <prefill wall seconds>]
"""
import argparse
import collections
import re

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--min-tokens", type=int, default=256)
ap.add_argument("--wall-s", type=float, default=0.0)
a = ap.parse_args()

kv = re.compile(r"(\w+)=(\S+)")
sub, meta, split, mtp = [], [], [], []
order = []   # the log order, to attach meta / split lines to the process_ubatch line that follows them
for line in open(a.log, errors="replace"):
    if "BIGCHERRY_SUBMIT_TIMING" in line:
        d = dict(kv.findall(line))
        order.append(("sub", d))
    elif "BIGCHERRY_META_TIMING" in line:
        order.append(("meta", dict(kv.findall(line))))
    elif "BIGCHERRY_SCHED_SPLIT" in line:
        order.append(("split", dict(kv.findall(line))))
    elif "BIGCHERRY_MTP_PROMPT_TIMING" in line:
        mtp.append(dict(kv.findall(line)))

# meta / split lines are printed inside graph_compute, i.e. before the SUBMIT line of the same call
pending_meta, pending_split, calls = [], [], []
for kind, d in order:
    if kind == "meta":
        pending_meta.append(d)
    elif kind == "split":
        pending_split.append(d)
    else:
        calls.append((d, pending_meta, pending_split))
        pending_meta, pending_split = [], []

by_ctx = collections.Counter(d["ctx"] for d, _, _ in calls if int(d["n_tokens"]) >= a.min_tokens)
if not by_ctx:
    raise SystemExit("no prefill chunks (n_tokens >= %d) in the log" % a.min_tokens)
print("contexts with prefill chunks:", ", ".join("%s x%d" % kv_ for kv_ in by_ctx.most_common()))


def ms(us):
    return us / 1000.0


for ctx, n in by_ctx.most_common():
    rows = [(d, m, s) for d, m, s in calls if d["ctx"] == ctx and int(d["n_tokens"]) >= a.min_tokens]
    tok = sum(int(d["n_tokens"]) for d, _, _ in rows)
    g = sum(int(d["graph_us"]) for d, _, _ in rows)
    i = sum(int(d["inputs_us"]) for d, _, _ in rows)
    c = sum(int(d["compute_us"]) for d, _, _ in rows)
    print("\n== ctx %s: %d prefill chunks, %d tokens" % (ctx, n, tok))
    print("  process_ubatch host time per chunk: graph %.1f ms, set_inputs %.1f ms, graph_compute %.1f ms  (sum %.2f s)"
          % (ms(g) / n, ms(i) / n, ms(c) / n, (g + i + c) / 1e6))
    if a.wall_s:
        print("  of %.2f s prefill wall: graph %.1f%%, set_inputs %.1f%%, graph_compute %.1f%%"
              % (a.wall_s, 100 * g / 1e6 / a.wall_s, 100 * i / 1e6 / a.wall_s, 100 * c / 1e6 / a.wall_s))
    mm = [m for _, ml, _ in rows for m in ml]
    if mm:
        print("  meta graph_compute per chunk: %d calls, rebuild %.1f ms, launch %.1f ms, allreduce enqueue %.1f ms, total %.1f ms; subgraphs %s"
              % (len(mm) / n, ms(sum(int(m["rebuild_us"]) for m in mm)) / n, ms(sum(int(m["launch_us"]) for m in mm)) / n,
                 ms(sum(int(m["allreduce_us"]) for m in mm)) / n, ms(sum(int(m["total_us"]) for m in mm)) / n,
                 sorted({int(m["n_subgraphs"]) for m in mm})))
    ss = [s for _, _, sl in rows for s in sl]
    if ss:
        agg = collections.defaultdict(lambda: [0, 0, 0, 0])
        for s in ss:
            k = s["backend"]
            agg[k][0] += 1
            agg[k][1] += int(s["input_us"])
            agg[k][2] += int(s["compute_us"])
            agg[k][3] += int(s["n_nodes"])
        print("  scheduler splits per chunk (%.1f splits):" % (len(ss) / n))
        for k, (cnt, iu, cu, nn) in sorted(agg.items(), key=lambda x: -(x[1][1] + x[1][2])):
            print("    %-28s %5.1f splits, %7.0f nodes, input %8.1f ms, compute %8.1f ms" % (k, cnt / n, nn / n, ms(iu) / n, ms(cu) / n))
        top = sorted(ss, key=lambda s: -(int(s["input_us"]) + int(s["compute_us"])))[:1]
        idx = collections.defaultdict(lambda: [0, 0, 0])
        for s in ss:
            key = (int(s["i"]), s["backend"], int(s["n_nodes"]))
            idx[key][0] += 1
            idx[key][1] += int(s["input_us"])
            idx[key][2] += int(s["compute_us"])
        print("  largest split positions (per chunk):")
        for (si, be, nn), (cnt, iu, cu) in sorted(idx.items(), key=lambda x: -(x[1][1] + x[1][2]))[:8]:
            print("    split %3d %-22s %6d nodes: input %8.1f ms, compute %8.1f ms (seen in %d chunks)" % (si, be, nn, ms(iu) / n, ms(cu) / n, cnt))
for d in mtp:
    if int(d.get("chunks", "0")) > 1:
        print("\nMTP prompt: " + " ".join("%s=%s" % kv_ for kv_ in d.items()))
