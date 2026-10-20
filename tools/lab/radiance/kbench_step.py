#!/usr/bin/env python3
"""Per-decode-step kernel time of a model on one card, from radiance's own tools (shape harness).

Inputs, both produced by the pinned radiance build with no server:
  GRAPH   `radiance --model M ... --debug-graph --debug-selection > GRAPH`: the declared graph, one block an op
          instance (op name at column 0, then its parameters), and at the end the selection table.
  REPORT  `rad-kbench -m M --kernels <libs> --max-tok T --timing-only --report REPORT`: device time of every kernel at
          the model's real shapes (weights read from the container in the layout each row asked for).

For each op the device time at the chosen step size (M, q_len or the closest smaller one the report timed) is multiplied
by the number of instances of that op in the declared graph, which gives the op's share of one forward pass at that
step size. Ops whose cost depends on routing (moe_*) are timed by rad-kbench on Zipf-skewed routing, not real traffic;
use replay-kernel-time.sh on captured requests for those.

usage: kbench_step.py GRAPH REPORT [--rows 16] [--top 30]
"""
import argparse
import collections
import re
import sys


def graph_counts(path):
    counts = collections.Counter()
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("selection table"):
            break
        m = re.match(r"^([a-z][a-z0-9_]*)\s+\S+=", line)
        if m:
            counts[m.group(1)] += 1
    return counts


def report_rows(path):
    rows = []
    in_speed = False
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("## Speed"):
            in_speed = True
            continue
        if in_speed and line.startswith("## "):
            break
        if not (in_speed and line.startswith("| `")):
            continue
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        op, kernel, lib, geom, dev = cells[0], cells[1], cells[2], cells[3], cells[4]
        try:
            us = float(dev)
        except ValueError:
            continue
        size = None
        for key in ("M", "q_len"):
            m = re.search(r"(?:^| )%s=(\d+)" % key, geom)
            if m:
                size = int(m.group(1))
                break
        rows.append((op, kernel, lib, geom, size, us))
    return rows


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("report")
    ap.add_argument("--rows", type=int, default=16, help="the step size to read times at (M or q_len)")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args(argv)
    counts = graph_counts(a.graph)
    best = {}
    for op, kernel, lib, geom, size, us in report_rows(a.report):
        if size is None or size > a.rows:
            continue
        key = (op, re.sub(r"(?:^| )(M|q_len)=\d+", "", geom))
        if key not in best or size > best[key][0]:
            best[key] = (size, us, kernel, lib)
    per_op = collections.defaultdict(list)
    for (op, g), (size, us, kernel, lib) in best.items():
        per_op[op].append((us, size, kernel, lib, g))
    total = 0.0
    table = []
    for op, cases in per_op.items():
        n = counts.get(op, 0)
        mean_us = sum(c[0] for c in cases) / len(cases)   # an op with several geometries: their mean a call
        step = mean_us * n
        total += step
        table.append((step, op, n, mean_us, cases[0][2], cases[0][3], len(cases)))
    table.sort(reverse=True)
    print("step size %d rows: %.1f us of timed kernels a forward pass" % (a.rows, total))
    print("%-28s %6s %10s %10s %6s  %s" % ("op", "calls", "us/call", "us/step", "share", "row (library), geometries"))
    for step, op, n, mean_us, kernel, lib, ng in table[: a.top]:
        print("%-28s %6d %10.1f %10.1f %5.1f%%  %s (%s), %d" % (op, n, mean_us, step, 100 * step / total if total else 0,
                                                              kernel, lib, ng))
    missing = sorted(op for op in counts if op not in per_op)
    if missing:
        print("not timed at <= %d rows: %s" % (a.rows, " ".join(missing)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
