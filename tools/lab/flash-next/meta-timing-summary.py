#!/usr/bin/env python3
"""Summarise 1320 BIGCHERRY_META_TIMING lines for decode-sized meta graph_compute calls (rebuild rate, n_subgraphs,
median rebuild/launch/allreduce/total host ms). Usage: meta-timing-summary.py <server.log>"""
import re
import statistics
import sys

pat = re.compile(r"BIGCHERRY_META_TIMING n_nodes=(\d+) n_subgraphs=(\d+) rebuild=(\d) rebuild_us=(\d+) launch_us=(\d+) "
                 r"allreduce_us=(\d+) total_us=(\d+)")
rows = [tuple(int(x) for x in m.groups()) for m in map(pat.search, open(sys.argv[1], errors="replace")) if m]
if not rows:
    print("no BIGCHERRY_META_TIMING lines")
    sys.exit(0)
rows = rows[len(rows) // 3:]  # skip prompt processing / warm-up
print(f"meta calls {len(rows)}  rebuilds {sum(r[2] for r in rows)}  n_subgraphs median {statistics.median(r[1] for r in rows)}"
      f"  n_nodes median {statistics.median(r[0] for r in rows)}")
for k, n in ((3, "rebuild"), (4, "launch"), (5, "allreduce"), (6, "total")):
    col = [r[k] for r in rows]
    print(f"  {n:9s} median {statistics.median(col)/1e3:6.3f} ms  mean {statistics.mean(col)/1e3:6.3f} ms")
sg = statistics.median(r[1] for r in rows)
print(f"  per subgraph: launch {statistics.median(r[4] / max(1, r[1]) for r in rows):.1f} us (3 devices), "
      f"allreduce {statistics.median(r[5] / max(1, r[1] - 1) for r in rows):.1f} us")
