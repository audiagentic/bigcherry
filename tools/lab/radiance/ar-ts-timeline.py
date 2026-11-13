#!/usr/bin/env python3
"""Summarise rad0031's two-shot all-reduce timelines (R4D_AR_TIMELINE=DIR): where an exact two-shot's time goes.

Usage: ar-ts-timeline.py DIR [--pid PID] [--row-bytes N] [--md]

Reads DIR/ar-ts-r<rank>-ws<world>-p<pid>-i<n>.bin (int64). A ring entry is four block records; a record is four
header words (sequence number; the message in 16-byte words; blocks | pieces << 8 | rank << 16 | world << 24; this
rank's shard in words) and the kernel's stamps: start, an (arrive, release) pair per handshake (pieces + 1 of them),
end. Stamps are a card's wall clock (100 MHz); card clocks are not aligned, so only differences on one card are used.

Per block, for a launch in K pieces (K = 1 is the lockstep form):

  push    = arrive[0] - start               first hop of piece 0 issued and drained
  wait1   = release[0] - arrive[0]          until every peer's first flag is visible
  reduce  = sum arrive[j] - release[j-1]    reduce + second hop (and the next piece's first hop, the last assemble)
  wait2   = sum release[j] - arrive[j]      until every peer's flag for that step is visible (j >= 1)
  asm     = end - release[K]                the last piece's assemble
  total   = end - start

A launch's phase is the mean over its recorded blocks; its total runs from the first block's start to the last
block's end. Ranks are matched by ring order on each rank (the k-th ring a rank made is every rank's k-th) and by
sequence number and size. The rank with the smallest wait1 of a launch arrived last: `last%` is how often that was
this rank, and `rt` (the smallest wait1 among the ranks) bounds the flag round trip. `gap` is the time from the end of
the previous launch on the same ring to this one's start, the rank's work between two all-reduces.

Grouped by message size, pieces and this rank's shard; medians in microseconds.
"""
from __future__ import annotations

import argparse
import re
import statistics as st
import struct
import sys
from collections import defaultdict
from pathlib import Path

RING, BLOCKS, TLW = 8192, 4, 36
BW = 4 + TLW
TICK_US = 0.01  # 100 MHz


def read(path: Path):
    """-> {seq: launch}, launch = dict(n16, nb, pieces, rank, own, start, end, phases)."""
    raw = path.read_bytes()
    n = len(raw) // 8
    v = struct.unpack(f"<{n}q", raw[: n * 8])
    out = {}
    for e in range(n // (BLOCKS * BW)):
        recs = []
        for b in range(BLOCKS):
            w = v[(e * BLOCKS + b) * BW:(e * BLOCKS + b + 1) * BW]
            seq, n16, packed, own = w[:4]
            if seq < 0 or n16 <= 0 or packed < 0:
                continue
            nb, k, rank = packed & 0xff, (packed >> 8) & 0xff, (packed >> 16) & 0xff
            t = w[4:4 + 4 + 2 * k]
            if len(t) < 4 + 2 * k or min(t) < 0 or any(t[i] > t[i + 1] for i in range(len(t) - 1)):
                continue
            recs.append((seq, n16, nb, k, rank, own, t))
        if not recs:
            continue
        # one launch: the records that share block 0's (or the first record's) sequence number and shape
        key = recs[0][:6]
        recs = [r for r in recs if r[:6] == key]
        seq, n16, nb, k, rank, own = key
        ph = []
        for r in recs:
            t = r[6]
            arr = [t[1 + 2 * j] for j in range(k + 1)]
            rel = [t[2 + 2 * j] for j in range(k + 1)]
            ph.append(((arr[0] - t[0]) * TICK_US, (rel[0] - arr[0]) * TICK_US,
                       sum(arr[j] - rel[j - 1] for j in range(1, k + 1)) * TICK_US,
                       sum(rel[j] - arr[j] for j in range(1, k + 1)) * TICK_US,
                       (t[-1] - rel[k]) * TICK_US))
        start = min(r[6][0] for r in recs)
        end = max(r[6][-1] for r in recs)
        out[seq] = dict(n16=n16, nb=nb, pieces=k, rank=rank, own=own, start=start, end=end,
                        phases=tuple(st.fmean(p[i] for p in ph) for i in range(5)),
                        total=(end - start) * TICK_US)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--row-bytes", type=int, default=0, help="bytes of one message row, to print rows beside KiB")
    ap.add_argument("--md", action="store_true", help="markdown table")
    ap.add_argument("--min-launches", type=int, default=8)
    a = ap.parse_args()

    rings = defaultdict(list)  # rank -> [(i, path)]
    for p in sorted(a.dir.glob("ar-ts-r*-ws*-p*-i*.bin")):
        m = re.fullmatch(r"ar-ts-r(\d+)-ws(\d+)-p(\d+)-i(\d+)\.bin", p.name)
        if not m or (a.pid is not None and int(m.group(3)) != a.pid):
            continue
        rings[int(m.group(1))].append((int(m.group(3)), int(m.group(4)), p))
    if not rings:
        print(f"no ar-ts-*.bin under {a.dir}", file=sys.stderr)
        return 1
    ranks = sorted(rings)
    per_rank = {r: [read(p) for _, _, p in sorted(rings[r])] for r in ranks}
    n_inst = min(len(v) for v in per_rank.values())

    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # (n16, pieces) -> rank -> col -> [..]
    owns = {}
    for k in range(n_inst):
        launches = {r: per_rank[r][k] for r in ranks}
        # this rank's work between two all-reduces on the ring
        for r in ranks:
            prev = None
            for seq in sorted(launches[r]):
                la = launches[r][seq]
                if prev is not None and seq - prev[0] <= 16 and la["start"] >= prev[1]:
                    la["gap"] = (la["start"] - prev[1]) * TICK_US
                prev = (seq, la["end"])
        for seq in sorted(set.intersection(*(set(launches[r]) for r in ranks))):
            ls = {r: launches[r][seq] for r in ranks}
            if len({(l["n16"], l["pieces"], l["nb"]) for l in ls.values()}) != 1:
                continue
            key = (ls[ranks[0]]["n16"], ls[ranks[0]]["pieces"])
            w1 = {r: ls[r]["phases"][1] for r in ranks}
            last = min(w1, key=w1.get)
            for r in ranks:
                g = groups[key][r]
                for i, name in enumerate(("push", "wait1", "reduce", "wait2", "asm")):
                    g[name].append(ls[r]["phases"][i])
                g["total"].append(ls[r]["total"])
                g["last"].append(1.0 if r == last else 0.0)
                if "gap" in ls[r]:
                    g["gap"].append(ls[r]["gap"])
                owns[(key, r)] = ls[r]["own"]
            groups[key]["*"]["rt"].append(min(w1.values()))

    cols = ("push", "wait1", "reduce", "wait2", "asm", "total", "gap")
    sep = " | " if a.md else "  "
    head = ["message", "pieces", "rank", "shard", "launches", *[f"{c} us" for c in cols], "last %", "rt us"]
    if a.md:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
    else:
        print(sep.join(head))
    for key in sorted(groups):
        n16, pieces = key
        n = len(groups[key]["*"]["rt"])
        if n < a.min_launches:
            continue
        size = f"{n16 * 16 / 1024:.0f} KiB"
        if a.row_bytes:
            size = f"{n16 * 16 / a.row_bytes:g} rows, " + size
        rt = st.median(groups[key]["*"]["rt"])
        for r in ranks:
            g = groups[key][r]
            row = [size, str(pieces), f"r{r}", f"{owns[(key, r)] / n16 * 100:.0f}%", str(n)]
            row += [f"{st.median(g[c]):.0f}" if g[c] else "-" for c in cols]
            row += [f"{100 * st.fmean(g['last']):.0f}", f"{rt:.1f}"]
            print(("| " + " | ".join(row) + " |") if a.md else sep.join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
