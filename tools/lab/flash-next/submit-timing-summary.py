#!/usr/bin/env python3
"""Summarise 1319 BIGCHERRY_SUBMIT_TIMING lines per context for decode-sized ubatches (n_tokens <= 8): medians of
graph build/reuse, set_inputs and graph_compute host time, and the graph reuse rate. The context with the most
multi-token (verify) calls is the target. Usage: submit-timing-summary.py <server.log>"""
import collections
import re
import statistics
import sys

pat = re.compile(r"BIGCHERRY_SUBMIT_TIMING ctx=(\S+) n_tokens=(\d+) reused=(\d) graph_us=(\d+) inputs_us=(\d+) compute_us=(\d+)")
by = collections.defaultdict(list)
for m in map(pat.search, open(sys.argv[1], errors="replace")):
    if m and int(m.group(2)) <= 8:
        by[m.group(1)].append(tuple(int(x) for x in m.groups()[1:]))
for ctx, rows in sorted(by.items(), key=lambda kv: -sum(1 for r in kv[1] if r[0] > 1)):
    rows = rows[len(rows) // 10:]
    print(f"ctx {ctx}: {len(rows)} decode-sized calls, reused {sum(r[1] for r in rows)/len(rows)*100:.0f}%, "
          f"n_tokens {collections.Counter(r[0] for r in rows).most_common(4)}")
    for k, n in ((2, "graph"), (3, "inputs"), (4, "compute")):
        col = [r[k] for r in rows]
        print(f"  {n:8s} median {statistics.median(col)/1e3:6.3f} ms  mean {statistics.mean(col)/1e3:6.3f} ms")
