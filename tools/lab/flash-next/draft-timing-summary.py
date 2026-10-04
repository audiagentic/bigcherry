#!/usr/bin/env python3
"""Summarise 1318 BIGCHERRY_DRAFT_TIMING lines: per-call medians (ms) of draft submit, draft GPU sync wait and the
host rest (sampling, nextn hidden read, batch rebuild). Usage: draft-timing-summary.py <server.log>"""
import re
import statistics
import sys

pat = re.compile(r"BIGCHERRY_DRAFT_TIMING steps=(\d+) submit_us=(\d+) sync_us=(\d+) rest_us=(-?\d+) total_us=(\d+)")
rows = [tuple(int(x) for x in m.groups()) for m in map(pat.search, open(sys.argv[1], errors="replace")) if m]
if not rows:
    print("no BIGCHERRY_DRAFT_TIMING lines")
    sys.exit(0)
rows = rows[len(rows) // 10:]
tot = sum(r[4] for r in rows)
print(f"draft calls {len(rows)}  steps/call {statistics.mean(r[0] for r in rows):.2f}  "
      f"call median {statistics.median(r[4] for r in rows)/1e3:.2f} ms")
for k, n in ((1, "submit"), (2, "gpu-sync"), (3, "host-rest")):
    col = [r[k] for r in rows]
    print(f"  {n:9s} median {statistics.median(col)/1e3:6.2f} ms/call  per step "
          f"{statistics.median(c / r[0] for c, r in zip(col, rows))/1e3:5.2f} ms  share {sum(col)/tot*100:5.1f}%")
