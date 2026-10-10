#!/usr/bin/env python3
"""Compare the accuracy of two kernel libraries over the same rad-kbench fixture.

Each kbench.json (kdev_report.py --json) holds, per case, the kernel's relative L2 error against the host reference.
This pairs the device cases of two runs by op and geometry and prints, per kernel, the worst and the mean error of
each library and how many cases have a different error at all. A kernel whose error is the same in both libraries
computes the same thing; one whose error is larger in the second library is where a model-level difference between
the two (rad-kld.sh) comes from.

Usage: kbench_accuracy.py <a/kbench.json> <b/kbench.json>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def device_cases(path: str) -> dict[tuple[str, str], dict]:
    cases = json.load(open(path))["cases"]
    return {(c["op"], c["geometry"]): c for c in cases if c["domain"] == "device" and c["rel_l2"] is not None}


def main() -> int:
    a, b = device_cases(sys.argv[1]), device_cases(sys.argv[2])
    per: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for key in sorted(set(a) & set(b)):
        per[f"{a[key]['kernel']} / {b[key]['kernel']}" if a[key]["kernel"] != b[key]["kernel"] else a[key]["kernel"]].append(
            (a[key]["rel_l2"], b[key]["rel_l2"]))
    print(f"{len(set(a) & set(b))} paired device cases ({len(a)} in A, {len(b)} in B)")
    print(f"{'kernel':<58} {'cases':>5} {'differ':>6} {'worst A':>9} {'worst B':>9} {'mean A':>9} {'mean B':>9}")
    for kernel, pairs in sorted(per.items(), key=lambda kv: -max(p[1] for p in kv[1])):
        ea, eb = [p[0] for p in pairs], [p[1] for p in pairs]
        differ = sum(1 for x, y in pairs if x != y)
        print(f"{kernel:<58} {len(pairs):>5} {differ:>6} {max(ea):>9.2e} {max(eb):>9.2e} "
              f"{sum(ea) / len(ea):>9.2e} {sum(eb) / len(eb):>9.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
