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
# Accepted-prefix distribution (greedy verification accepts a prefix): P(prefix >= k) and the FMTP calibration terms
# when the run drafted deeper than the production front (e.g. SPEC_N=7 vs front 3): P(full front), P(bridge | full)
# and the conditional yield of the would-be promoted tail.
FRONT = 3
nmax = max(nd) if nd else 0
if nmax > FRONT:
    def p_ge(k, pool):
        return sum(1 for a in pool if a >= k) / max(1, len(pool))
    print("  P(accepted prefix >= k): " + " ".join(f"k{k}={p_ge(k, na):.2f}" for k in range(1, nmax + 1)))
    eligible = [a for d, a in zip(nd, na) if d > FRONT]  # rounds that drafted past the front
    full = [a for a in eligible if a >= FRONT]
    bridge = [a for a in full if a >= FRONT + 1]
    tail = [min(a, nmax) - (FRONT + 1) for a in bridge]
    print(f"  FMTP calibration (front {FRONT}): P(full front) {len(full)/max(1,len(eligible)):.2f}  "
          f"P(bridge|full) {len(bridge)/max(1,len(full)):.2f}  "
          f"E[accepted promoted tail | bridge] {sum(tail)/max(1,len(tail)):.2f} of {nmax - FRONT - 1}")
