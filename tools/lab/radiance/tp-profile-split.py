#!/usr/bin/env python3
"""Where a profiled Radiance run's device time goes, per rank: trunk compute, drafter, collectives, top ops.

Usage: tp-profile-split.py SERVE_LOG [SERVE_LOG ...] [--top N]

Reads the `per-op timing, rank R` tables `--profile-ops` prints at shutdown (the last table per rank wins). An op is
the drafter's when its weight is a `dflash.` or `mtp.` tensor, a collective when its op name is all_reduce, all_gather
or an ar_ row, and the trunk's otherwise. A collective's time includes waiting for its peers, so its share rises when
a rank arrives early; compare ranks and configurations, not absolute shares against an unprofiled run.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

ROW = re.compile(r"\s+(\d+)\s+([\d.]+)\s+[\d.]+\s+\w+\s+\S+\s+\S+\s+(\S+)\s+(.*)$")
HEAD = re.compile(r"per-op timing, rank (\d+)")
DRAFT = ("dflash.", "mtp.")
COLL = ("all_reduce", "all_gather", "ar_")


def parse(path: str):
    out, rank, cur = {}, None, None
    for line in open(path, errors="replace"):
        m = HEAD.match(line)
        if m:
            rank, cur = int(m.group(1)), defaultdict(float)
            continue
        if rank is None:
            continue
        m = ROW.match(line)
        if m:
            ms, op, tail = float(m.group(2)), m.group(3), m.group(4)
            if any(d in tail for d in DRAFT):
                kind = "drafter"
            elif op.startswith(COLL):
                kind = "collectives"
            else:
                kind = "trunk"
            weight = tail.split()[-1] if tail.split() else "-"
            name = op if weight == "-" else op + " " + re.sub(r"blk\.\d+\.", "", weight)  # all layers as one
            cur[(kind, name)] += ms
        elif line.strip().startswith("total"):
            out[rank], rank = cur, None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--top", type=int, default=4)
    a = ap.parse_args()
    for path in a.logs:
        ranks = parse(path)
        print(f"== {path.rsplit('/', 1)[-1]}")
        if not ranks:
            print("  no per-op table (was the server stopped cleanly with --profile-ops?)")
            continue
        for r in sorted(ranks):
            ops = ranks[r]
            total = sum(ops.values()) or 1.0
            kinds = defaultdict(float)
            for (kind, _), ms in ops.items():
                kinds[kind] += ms
            top = sorted(((ms, name) for (kind, name), ms in ops.items() if kind == "trunk"), reverse=True)[: a.top]
            print(f"  rank {r}: {total:8.0f} ms  trunk {kinds['trunk'] / total:5.1%}  drafter "
                  f"{kinds['drafter'] / total:5.1%}  collectives {kinds['collectives'] / total:5.1%}  | "
                  + ", ".join(f"{n} {ms:.0f}" for ms, n in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
