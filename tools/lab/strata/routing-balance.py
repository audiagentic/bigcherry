#!/usr/bin/env python3
"""MET04: how do a token's routed experts fall across the cards when experts are split by index range?

Reads Strata `--dump-routing` traces (records: int32 layer, int32 k, k int32 expert ids, k float32 weights). For each
set of expert shares (as BIGCHERRY_MOE_EP_TS: contiguous index ranges in device order, the same in every layer) it
reports, over all (layer, position) records: the mean number of the k selected experts on each card, the mean of the
busiest card, how often one card holds at least 70% of a token's experts, and the least and most balanced layers.
An even split of k=10 over three cards has a busiest card of 4 at best.

--placed CAPS:TRAFFIC is the usage-aware placement instead of index ranges: per layer the experts are ranked by
routing frequency and dealt, hottest first, to the card that is furthest below its TRAFFIC target and still has room
(CAPS = experts each card may hold, e.g. 128,128,256:1,1,1 = hot experts spread evenly over all three cards, the cold
remainder on the last card). It is scored held out: the frequencies come from the other traces.

Usage: routing-balance.py [--n-expert 512] [--shares 0.31,0.27,0.42 ...] [--placed 128,128,256:1,1,1 ...] TRACE ...
"""
import argparse
import struct
from pathlib import Path


def read_trace(path, n_expert):
    blob = Path(path).read_bytes()
    off = 0
    while off + 8 <= len(blob):
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        ids = [e for e in struct.unpack_from("<%di" % k, blob, off) if 0 <= e < n_expert]
        off += 8 * k
        yield layer, ids


def bounds(shares, n_expert):
    total, cum, out = sum(shares), 0.0, []
    for i, s in enumerate(shares):
        cum += s
        out.append(n_expert if i + 1 == len(shares) else int(n_expert * (cum / total)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--shares", action="append", default=[])
    ap.add_argument("--placed", action="append", default=[])
    ap.add_argument("--n-expert", type=int, default=512)
    a = ap.parse_args()
    by_trace = [list(read_trace(t, a.n_expert)) for t in a.traces]
    recs = [r for t in by_trace for r in t]
    print(f"{len(recs)} (layer, position) records from {len(a.traces)} trace(s)")
    for spec in a.shares:
        shares = [float(x) for x in spec.split(",")]
        hi = bounds(shares, a.n_expert)
        n = len(shares)
        tot = [0] * n
        busiest = heavy = 0
        per_layer = {}
        for layer, ids in recs:
            c = [0] * n
            for e in ids:
                for d in range(n):
                    if e < hi[d]:
                        c[d] += 1
                        break
            for d in range(n):
                tot[d] += c[d]
            m = max(c)
            busiest += m
            heavy += m >= 0.7 * len(ids)
            s = per_layer.setdefault(layer, [0, 0])
            s[0] += m
            s[1] += 1
        k = sum(tot) / len(recs)
        held = [hi[0]] + [hi[d] - hi[d - 1] for d in range(1, n)]
        layers = sorted((s[0] / s[1], layer) for layer, s in per_layer.items())
        print(f"\nshares {spec}: experts held {held}")
        print("  mean selected per card: " + " / ".join(f"{t / len(recs):.2f}" for t in tot)
              + f" of {k:.1f}  (by share alone: " + " / ".join(f"{k * h / a.n_expert:.2f}" for h in held) + ")")
        print(f"  busiest card: mean {busiest / len(recs):.2f}; one card holds >= 70% of a token's experts in "
              f"{100.0 * heavy / len(recs):.1f}% of records")
        print(f"  per layer, busiest card: best layer {layers[0][1]} {layers[0][0]:.2f}, worst layer {layers[-1][1]} {layers[-1][0]:.2f}")
    for spec in a.placed:
        score_placed(by_trace, spec, a.n_expert)



def place(freq_recs, caps, traffic, n_expert):
    """layer -> card of each expert: hottest first to the card furthest below its traffic target with room left"""
    counts = {}
    for layer, ids in freq_recs:
        row = counts.setdefault(layer, [0] * n_expert)
        for e in ids:
            row[e] += 1
    out = {}
    total_t = sum(traffic)
    for layer, row in counts.items():
        load, room, card = [0.0] * len(caps), list(caps), [len(caps) - 1] * n_expert
        for e in sorted(range(n_expert), key=lambda x: (-row[x], x)):
            open_cards = [d for d in range(len(caps)) if room[d] > 0]
            d = min(open_cards, key=lambda c: (load[c] / (traffic[c] / total_t) if traffic[c] > 0 else float("inf"), c))
            card[e] = d
            room[d] -= 1
            load[d] += row[e]
        out[layer] = card
    return out


def score_placed(by_trace, spec, n_expert):
    caps_s, traffic_s = spec.split(":")
    caps, traffic = [int(x) for x in caps_s.split(",")], [float(x) for x in traffic_s.split(",")]
    assert sum(caps) == n_expert, "CAPS must add up to the expert count"
    n = len(caps)
    tot, busiest, heavy, count = [0] * n, 0, 0, 0
    for i, test in enumerate(by_trace):
        train = [r for j, t in enumerate(by_trace) if j != i or len(by_trace) == 1 for r in t]
        placement = place(train, caps, traffic, n_expert)
        for layer, ids in test:
            card = placement.get(layer)
            if card is None:
                continue
            c = [0] * n
            for e in ids:
                c[card[e]] += 1
            for d in range(n):
                tot[d] += c[d]
            busiest += max(c)
            heavy += max(c) >= 0.7 * len(ids)
            count += 1
    print(f"\nplaced {spec} (held out):")
    print("  mean selected per card: " + " / ".join(f"{t / count:.2f}" for t in tot))
    print(f"  busiest card: mean {busiest / count:.2f}; one card holds >= 70% of a token's experts in {100.0 * heavy / count:.1f}% of records")


if __name__ == "__main__":
    main()
