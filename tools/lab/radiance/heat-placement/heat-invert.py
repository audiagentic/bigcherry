#!/usr/bin/env python3
"""Turn a recorded route histogram into the launcher's --heat file (RHP08).

The engine (R3_MOE_WORKER_HISTOGRAM, or the launcher's --record-heat) writes uint32[128][512] route counts. A helper given
R3_MOE_WORKER_HEAT owns the COLDEST experts first, so the hottest-first placement the launcher wants is max - count.
Usage: heat-invert.py <histogram> <out>
"""
import sys
import numpy as np

LAYERS, EXPERTS = 128, 512


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        counts = np.fromfile(sys.argv[1], dtype=np.uint32)
    except OSError as exc:
        print(f"heat-invert: {exc}", file=sys.stderr)
        return 1
    if counts.size != LAYERS * EXPERTS:
        print(f"heat-invert: {sys.argv[1]} holds {counts.size} counts, want {LAYERS * EXPERTS}", file=sys.stderr)
        return 1
    if not counts.any():
        print(f"heat-invert: {sys.argv[1]} is all zeros (no routes recorded)", file=sys.stderr)
        return 1
    wide = counts.astype(np.int64)
    (wide.max() - wide).astype(np.uint32).tofile(sys.argv[2])
    print(f"heat-invert: {int(wide.sum())} routes, max {int(wide.max())}, wrote {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
