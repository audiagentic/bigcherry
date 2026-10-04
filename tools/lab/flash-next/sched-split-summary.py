#!/usr/bin/env python3
"""Summarise 1325 BIGCHERRY_SCHED_SPLIT lines for the decode phase: per split position, the backend, median input and
compute host ms, so the scheduler time outside the meta backend can be attributed. Usage: sched-split-summary.py <log>"""
import collections
import re
import statistics
import sys

pat = re.compile(r"BIGCHERRY_SCHED_SPLIT i=(\d+) n_splits=(\d+) backend=(\S+) n_inputs=(\d+) n_nodes=(\d+) input_us=(\d+) compute_us=(\d+)")
rows = [m.groups() for m in map(pat.search, open(sys.argv[1], errors="replace")) if m]
rows = rows[len(rows) // 3:]  # skip prompt processing / warm-up
by = collections.defaultdict(list)
for i, ns, be, ni, nn, inp, comp in rows:
    by[(int(ns), int(i), be, int(ni), int(nn))].append((int(inp), int(comp)))
for key, vals in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    if len(vals) < 10:
        continue
    ns, i, be, ni, nn = key
    print(f"splits={ns} i={i} {be:14s} inputs={ni:3d} nodes={nn:5d} calls={len(vals):4d}  input median "
          f"{statistics.median(v[0] for v in vals)/1e3:6.3f} ms  compute median {statistics.median(v[1] for v in vals)/1e3:6.3f} ms")
