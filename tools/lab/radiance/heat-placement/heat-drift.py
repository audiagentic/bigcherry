#!/usr/bin/env python3
"""RHP05 stage A: does the active heat file still fit what is being served? (no GPU)

Inputs are the launcher's files: the active --heat file (hot-first: max - count, uint32[128][512]) and a live route histogram
(--record-heat PREFIX writes PREFIX.rank0, uint32[128][512] counts). With the layout's split -- the heads own the hottest
HEADS experts, the tail the coldest TAIL that are left -- it reports the share of live routes each group would get under the
active file and under a file re-ranked from the live counts. On flashnext-4gpu the tail costs ~10x a head per route, so the
tail's share decides whether it holds up the engine: matched heat ~1.5%, mismatched 4-12% (RAD23 2026-11-01).

  heat-drift.py ACTIVE.hot-first LIVE.rank0 [--tail 5931] [--heads 18640] [--ratio 1.5] [--min-share 2.5] [--write NEW.hot-first]

Exit 0: the active file is good enough -- the tail's share is at most --min-share percent (below that the tail finishes
before the heads and a restart buys nothing; held-out RAD23 probe: 1.46% with the mixed file, tail 1.6 ms against the
heads' 3.7 ms a request) or at most --ratio x the re-ranked share; 3: re-rank advised (with --write the re-ranked
hot-first file is written, to restart with --heat NEW). A restart is the activation step (RHP05: never a live
owner change).
"""
import argparse
import sys
import numpy as np

LAYERS, EXPERTS = 128, 512


def load(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype=np.uint32)
    if a.size != LAYERS * EXPERTS:
        raise SystemExit(f"heat-drift: {path} holds {a.size} values, want {LAYERS * EXPERTS}")
    return a.astype(np.float64)


def shares(counts_rank: np.ndarray, live: np.ndarray, idx: np.ndarray, tail: int, heads: int) -> tuple[float, float]:
    order = idx[np.argsort(counts_rank[idx], kind="stable")]   # coldest first
    total = live[idx].sum()
    return live[order[:tail]].sum() / total, live[order[-heads:]].sum() / total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("active"); ap.add_argument("live")
    ap.add_argument("--tail", type=int, default=5931, help="experts the tail owns (its log: 'owns N of M ranked experts')")
    ap.add_argument("--heads", type=int, default=18640, help="experts the heads own together")
    ap.add_argument("--ratio", type=float, default=1.5, help="advise a re-rank when the tail share exceeds this x the re-ranked share")
    ap.add_argument("--min-share", type=float, default=2.5, help="percent of live routes on the tail below which the file is kept")
    ap.add_argument("--write", help="write the re-ranked hot-first file here when a re-rank is advised")
    a = ap.parse_args()
    hot_first, live = load(a.active), load(a.live)
    rank = hot_first.max() - hot_first                      # back to counts: higher = hotter
    layer_routes = live.reshape(LAYERS, EXPERTS).sum(1)
    if not layer_routes.any():
        print("heat-drift: the live histogram is empty (no routes recorded yet)", file=sys.stderr)
        return 1
    idx = np.concatenate([np.arange(l * EXPERTS, (l + 1) * EXPERTS) for l in np.nonzero(layer_routes)[0]])
    t_now, h_now = shares(rank, live, idx, a.tail, a.heads)
    t_best, h_best = shares(live, live, idx, a.tail, a.heads)
    print(f"live routes {int(live[idx].sum())} over {len(idx) // EXPERTS} layers")
    print(f"active file:  tail {100 * t_now:5.2f}%  heads {100 * h_now:5.2f}% of live routes")
    print(f"re-ranked:    tail {100 * t_best:5.2f}%  heads {100 * h_best:5.2f}%")
    if 100 * t_now <= a.min_share or t_now <= a.ratio * max(t_best, 1e-9):
        print("keep the active file")
        return 0
    print(f"re-rank advised: the tail takes {t_now / max(t_best, 1e-9):.1f}x the routes it would with live heat")
    if a.write:
        (live.max() - live).astype(np.uint32).tofile(a.write)
        print(f"wrote {a.write}; restart the launcher with --heat {a.write}")
    return 3


if __name__ == "__main__":
    sys.exit(main())
