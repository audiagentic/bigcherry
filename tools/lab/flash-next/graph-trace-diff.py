#!/usr/bin/env python3
"""QFP41: where two runs' HIP graph decisions part, from 1356's BIGCHERRY_META_DISPATCH_GRAPH_TRACE=1 lines.

Each line is one graph_compute call on one device: BIGCHERRY_1356_GRAPH device=D call=N decision=... props=... uid=...
nodes=... use=... update=... warm=... instance=... . The property hash and uid depend on addresses, so two runs are
compared on what must match: per device and context instance, the sequence of (decision, nodes, use, update, warm).

Prints, per device, the counts of each decision in both runs and the first call at which the sequences differ, with
a few calls of context from each run.

Usage: graph-trace-diff.py <server log A> <server log B> [context, default 4]
"""
from __future__ import annotations

import collections
import re
import sys

LINE = re.compile(r"BIGCHERRY_1356_GRAPH (.*)")


def load(path):
    per = collections.defaultdict(list)
    with open(path, errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if not m:
                continue
            kv = dict(p.split("=", 1) for p in m.group(1).split() if "=" in p)
            # one stream a device and context instance: calls of different contexts interleave by timing
            per[kv.get("device", "?") + "/" + kv.get("instance", "?")].append(kv)
    return per


def key(kv):
    return (kv.get("decision"), kv.get("nodes"), kv.get("use"), kv.get("update"), kv.get("warm"))


def show(kv):
    return " ".join(f"{k}={kv.get(k)}" for k in ("call", "decision", "nodes", "use", "update", "warm", "instance", "uid"))


def main() -> int:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    context = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    for dev in sorted(set(a) | set(b)):
        ca = collections.Counter(kv.get("decision") for kv in a[dev])
        cb = collections.Counter(kv.get("decision") for kv in b[dev])
        print(f"device/instance {dev}: A {len(a[dev])} calls {dict(ca)}; B {len(b[dev])} calls {dict(cb)}")
        n = min(len(a[dev]), len(b[dev]))
        first = next((i for i in range(n) if key(a[dev][i]) != key(b[dev][i])), None)
        if first is None:
            print(f"  identical for the first {n} calls")
            continue
        print(f"  first difference at index {first}")
        for i in range(max(0, first - context), min(n, first + context + 1)):
            mark = ">>" if key(a[dev][i]) != key(b[dev][i]) else "  "
            print(f"  {mark} A {show(a[dev][i])}")
            print(f"  {mark} B {show(b[dev][i])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
