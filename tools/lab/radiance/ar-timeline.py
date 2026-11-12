#!/usr/bin/env python3
"""Summarise rad0023's all-reduce timelines (R4D_AR_TIMELINE=DIR): where a one-shot all-reduce's time goes.

Usage: ar-timeline.py DIR [--pid PID]

Reads DIR/ar-tl-r<rank>-ws<world>-p<pid>-i<n>.bin (int64; per entry: seq, 16-byte words, blocks, rank, then four
stamps for each of up to eight blocks: entry, pushed, peers seen, folded; -1 unset). Stamps are a card's wall clock
(100 MHz); card clocks are not aligned, so only differences on one card are used. Per block:

  push   = pushed - entry        issuing this rank's message to its peers' slots and draining the stores
  wait   = seen - pushed         until every peer's flag for this launch is visible
  fold   = folded - seen         reading the peers' messages and storing the sum
  total  = folded - entry

A launch's phase is the slowest block's. The instance number in a file name counts rings across every rank of the
process, so it does not name the same all-reduce on each rank; a rank's rings are matched by their order on that rank
(its k-th ring is every rank's k-th), and an entry only joins a launch whose message size agrees. The smallest wait
among a launch's ranks is the rank that arrived last, so it bounds the flag round trip; the others' extra wait is a
peer arriving later than they did. Grouped by message size.
"""
from __future__ import annotations

import argparse
import re
import statistics as st
import struct
import sys
from collections import defaultdict
from pathlib import Path

RING, BLOCKS = 16384, 8
WORDS = 4 + 4 * BLOCKS
TICK_US = 0.01  # 100 MHz


def read(path: Path):
    raw = path.read_bytes()
    n = len(raw) // 8
    vals = struct.unpack(f"<{n}q", raw[: n * 8])
    out = []
    for e in range(n // WORDS):
        w = vals[e * WORDS:(e + 1) * WORDS]
        seq, n16, nb, rank = w[:4]
        if seq < 0 or n16 <= 0:
            continue
        blocks = []
        for b in range(min(nb, BLOCKS)):
            t = w[4 + 4 * b: 8 + 4 * b]
            if min(t) < 0 or not (t[0] <= t[1] <= t[2] <= t[3]):
                continue
            blocks.append(((t[1] - t[0]) * TICK_US, (t[2] - t[1]) * TICK_US, (t[3] - t[2]) * TICK_US,
                           (t[3] - t[0]) * TICK_US))
        if blocks:
            out.append((seq, n16, rank, tuple(max(b[i] for b in blocks) for i in range(4))))
    return out


def band(n16: int) -> str:
    kib = n16 * 16 / 1024
    for lim in (64, 128, 256, 512, 1024, 4096, 16384):
        if kib <= lim:
            return f"<= {lim} KiB"
    return "> 16 MiB"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--pid", type=int)
    a = ap.parse_args()
    pat = re.compile(r"ar-tl-r(\d+)-ws(\d+)-p(\d+)-i(\d+)\.bin$")
    files = defaultdict(list)       # (pid, ws, rank) -> [(instance, path)]
    for p in sorted(a.dir.glob("ar-tl-*.bin")):
        m = pat.search(p.name)
        if not m or (a.pid and int(m.group(3)) != a.pid):
            continue
        files[(int(m.group(3)), int(m.group(2)), int(m.group(1)))].append((int(m.group(4)), p))
    if not files:
        print(f"no ar-tl-*.bin under {a.dir}", file=sys.stderr)
        return 1
    launches = defaultdict(dict)   # (pid, ws, ordinal, seq) -> rank -> phases
    sizes = {}
    unmatched = 0
    for (pid, ws, rank), insts in files.items():
        for ordinal, (_, p) in enumerate(sorted(insts)):
            for seq, n16, r, ph in read(p):
                key = (pid, ws, ordinal, seq)
                if sizes.setdefault(key, n16) != n16:
                    unmatched += 1
                    key = (pid, ws, ordinal, seq, rank)  # stands alone: counted per rank, not matched
                    sizes[key] = n16
                launches[key][rank] = ph
    if unmatched:
        print(f"{unmatched} entries disagree on size with the same launch on another rank; left unmatched",
              file=sys.stderr)
    rows = defaultdict(lambda: defaultdict(list))
    for key, per in launches.items():
        b = band(sizes[key])
        for r, (push, wait, fold, tot) in per.items():
            rows[b][f"r{r} push"].append(push)
            rows[b][f"r{r} wait"].append(wait)
            rows[b][f"r{r} fold"].append(fold)
            rows[b][f"r{r} total"].append(tot)
        if len(per) > 1:
            waits = [ph[1] for ph in per.values()]
            rows[b]["min wait (round trip)"].append(min(waits))
            rows[b]["max-min wait (late peer)"].append(max(waits) - min(waits))
    for b in sorted(rows, key=lambda s: (s.startswith(">"), float(re.findall(r"[\d.]+", s)[0]))):
        n = max(len(v) for v in rows[b].values())
        print(f"\n{b} ({n} launch-ranks)")
        print(f"  {'phase':28} {'median us':>10} {'p10':>8} {'p90':>8}")
        for k in sorted(rows[b]):
            v = rows[b][k]
            print(f"  {k:28} {st.median(v):10.1f} {pct(v, .1):8.1f} {pct(v, .9):8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
