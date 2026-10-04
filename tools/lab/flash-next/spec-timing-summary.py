#!/usr/bin/env python3
"""Summarise 1317 BIGCHERRY_SPEC_TIMING lines (FMTP01 Gate 0): median/mean per phase in ms, share of the round,
acceptance stats. Usage: spec-timing-summary.py <server.log>"""
import re
import statistics
import sys

pat = re.compile(r"BIGCHERRY_SPEC_TIMING draft_us=(\d+) submit_us=(\d+) sync_us=(\d+) process_us=(\d+) "
                 r"sample_us=(\d+) n_draft=(\d+) n_acc=(\d+)")
rows = [tuple(int(x) for x in m.groups()) for m in map(pat.search, open(sys.argv[1], errors="replace")) if m]
if not rows:
    print("no BIGCHERRY_SPEC_TIMING lines")
    sys.exit(0)
rows = rows[len(rows) // 10:]  # drop warm-up rounds
names = ["draft", "submit", "sync", "process", "sample"]
tot = [sum(r[:5]) for r in rows]
print(f"rounds {len(rows)}  round median {statistics.median(tot)/1e3:.2f} ms  mean {statistics.mean(tot)/1e3:.2f} ms")
for k, n in enumerate(names):
    col = [r[k] for r in rows]
    print(f"  {n:8s} median {statistics.median(col)/1e3:6.2f} ms  mean {statistics.mean(col)/1e3:6.2f} ms  "
          f"share {sum(col)/sum(tot)*100:5.1f}%")
nd = [r[5] for r in rows]
na = [r[6] for r in rows]
full = sum(1 for d, a in zip(nd, na) if a == d and d > 0)
print(f"  drafted {sum(nd)} accepted {sum(na)} ({sum(na)/max(1,sum(nd))*100:.1f}%)  full-front rounds "
      f"{full}/{len(rows)} ({full/len(rows)*100:.1f}%)  tokens/round {(sum(na)+len(rows))/len(rows):.2f}")
