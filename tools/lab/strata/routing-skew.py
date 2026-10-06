#!/usr/bin/env python3
"""MET01 / BCOP37: how skewed is expert routing, and does a frequency profile transfer between request types?

Reads Strata `--dump-routing` traces (records: int32 layer, int32 k, k int32 expert ids, k float32 weights) and, for
VRAM budgets given as a share of all (layer, expert) pairs, reports the routed-pair hit rate of four placements:

  layers   whole layers resident (what --n-cpu-moe gives): hit rate = share of layers
  global   the most frequent pairs over all layers (Strata's profile order)
  quota    the same number of slots in every layer, each layer's most frequent experts
  shipped  an existing Strata profile file (--profile), in its own order

"in" scores a placement on the trace it was built from, "x:<name>" on another trace (transfer). The last block is
the prefill upload set: distinct routed experts per layer in a window of 512 / 2048 positions, as a share of the
layer's experts - what a host-expert micro-batch has to have in VRAM at once.

Usage: routing-skew.py [--profile P] [--n-layer 48 --n-expert 512] NAME=TRACE ...
"""
import argparse
import struct
from collections import Counter
from pathlib import Path


def read_trace(path, n_layer, n_expert):
    blob = Path(path).read_bytes()
    off, recs = 0, []
    while off + 8 <= len(blob):
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        ids = struct.unpack_from("<%di" % k, blob, off)
        off += 8 * k
        if 0 <= layer < n_layer:
            recs.append((layer, [e for e in ids if 0 <= e < n_expert]))
    return recs


def freq(recs):
    c = Counter()
    for layer, ids in recs:
        for e in ids:
            c[(layer, e)] += 1
    return c


def hit(recs_freq, resident):
    total = sum(recs_freq.values())
    return sum(n for p, n in recs_freq.items() if p in resident) / total if total else 0.0


def global_set(f, slots):
    return {p for p, _ in sorted(f.items(), key=lambda kv: (-kv[1], kv[0]))[:slots]}


def quota_set(f, slots, n_layer):
    per = slots // n_layer
    out = set()
    for layer in range(n_layer):
        rows = sorted(((n, e) for (l, e), n in f.items() if l == layer), reverse=True)[:per]
        out.update((layer, e) for _, e in rows)
    return out


def read_profile(path):
    blob = Path(path).read_bytes()
    assert blob[:4] == b"STRP", "not a Strata profile"
    _, _, _, _, n = struct.unpack_from("<5I", blob, 4)
    return [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", help="NAME=TRACE")
    ap.add_argument("--profile")
    ap.add_argument("--n-layer", type=int, default=48)
    ap.add_argument("--n-expert", type=int, default=512)
    ap.add_argument("--shares", default="10,17,25,40,50,60,75")
    a = ap.parse_args()
    total_pairs = a.n_layer * a.n_expert
    traces = {}
    for t in a.traces:
        name, path = t.split("=", 1)
        recs = read_trace(path, a.n_layer, a.n_expert)
        traces[name] = (recs, freq(recs))
        f = traces[name][1]
        print(f"{name}: {len(recs)} (layer, position) records, {sum(f.values())} routed pairs, "
              f"{len(f)} distinct of {total_pairs} ({100.0 * len(f) / total_pairs:.1f}%)")
    shipped = read_profile(a.profile) if a.profile else None
    names = list(traces)
    for share in (int(s) for s in a.shares.split(",")):
        slots = total_pairs * share // 100
        print(f"\n== {share}% of pairs resident ({slots} slots); whole layers would hit {share}%")
        for src in names:
            f_src = traces[src][1]
            g, q = global_set(f_src, slots), quota_set(f_src, slots, a.n_layer)
            cells = [f"in {100 * hit(f_src, g):.1f} / {100 * hit(f_src, q):.1f}"]
            for dst in names:
                if dst != src:
                    f_dst = traces[dst][1]
                    cells.append(f"x:{dst} {100 * hit(f_dst, g):.1f} / {100 * hit(f_dst, q):.1f}")
            print(f"  profile from {src:8s} global / quota: " + "; ".join(cells))
        if len(names) > 1:   # leave-one-out: profile from every other trace
            for dst in names:
                f_rest = Counter()
                for src in names:
                    if src != dst:
                        f_rest.update(traces[src][1])
                f_dst = traces[dst][1]
                print(f"  profile from all but {dst:8s}: {100 * hit(f_dst, global_set(f_rest, slots)):.1f} / "
                      f"{100 * hit(f_dst, quota_set(f_rest, slots, a.n_layer)):.1f}")
        if shipped:
            s = set(shipped[:slots])
            print("  shipped profile: " + "; ".join(f"{n} {100 * hit(traces[n][1], s):.1f}" for n in names))
    print("\n== prefill upload set: distinct routed experts per layer in a window, share of the layer's experts")
    for name, (recs, _) in traces.items():
        by_layer = {}
        for layer, ids in recs:
            by_layer.setdefault(layer, []).append(ids)
        for win in (512, 2048):
            shares = []
            for rows in by_layer.values():
                for i in range(0, len(rows) - win + 1, win):
                    shares.append(len({e for ids in rows[i:i + win] for e in ids}) / a.n_expert)
            if shares:
                print(f"  {name:8s} window {win}: mean {100 * sum(shares) / len(shares):.1f}%, "
                      f"min {100 * min(shares):.1f}%, max {100 * max(shares):.1f}% ({len(shares)} windows)")


if __name__ == "__main__":
    main()
