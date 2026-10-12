#!/usr/bin/env python3
"""Per-op profile of a radiance run (--profile-ops), summed by op: where a rank's device time goes.

The engine prints one row per op site (calls, total ms, us/call, site, join, op and geometry). This sums the rows of
each rank by op name and prints the largest, with each op's share of the rank's total and its call count, so two
kernel libraries or two cards can be compared op by op. A profiled run is not comparable with an unprofiled one in
absolute speed (the engine says so); shares and per-call times are what to read.

Usage: profile_ops_summary.py <server.log> [top N, default 18]
"""
import collections
import re
import sys


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    row = re.compile(r"^ +([0-9]+) +([0-9.]+) +([0-9.]+) +(dev|host) +(-|[0-9.]+) +([A-Za-z0-9_]+)(.*)$")
    rank = None
    ops = {}
    for line in open(sys.argv[1], errors="replace"):
        m = re.match(r"per-op timing, rank ([0-9]+)", line)
        if m:
            rank = int(m.group(1))
            ops[rank] = collections.defaultdict(lambda: [0, 0.0, set()])
            continue
        if rank is None:
            continue
        m = row.match(line)
        if m:
            entry = ops[rank][m.group(6)]
            entry[0] += int(m.group(1))
            entry[1] += float(m.group(2))
            entry[2].add(m.group(4))
    if not ops:
        sys.exit("no per-op table in " + sys.argv[1])
    for rank in sorted(ops):
        total = sum(v[1] for v in ops[rank].values()) or 1.0
        print("rank %d: %.0f ms over %d ops" % (rank, total, len(ops[rank])))
        print("    share   total ms      calls   us/call  site  op")
        for name, (calls, ms, sites) in sorted(ops[rank].items(), key=lambda kv: -kv[1][1])[:top]:
            print("  %6.1f%% %10.1f %10d %9.1f  %-4s  %s" % (100 * ms / total, ms, calls, 1e3 * ms / max(calls, 1),
                                                           "/".join(sorted(sites)), name))


if __name__ == "__main__":
    main()
