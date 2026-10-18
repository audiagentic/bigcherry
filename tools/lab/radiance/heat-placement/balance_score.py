#!/usr/bin/env python3
"""RAD06/RHP04 stepping stone: predicted helper balance, no GPU.

The two r3_moe_helper processes each own whole (layer, expert) pairs and a layer finishes when BOTH are done, so what
matters per layer is max(helper0 cost, helper1 cost). select_owned() today alternates down one global heat ranking
(helper h takes positions p with p % 2 == h) until its byte budget fills: that balances the number of experts, not the
work. This script takes the raw route-count heat file a profiling run wrote (uint32 [128][512]), reproduces today's
jointly owned set and alternation exactly, and scores it against a per-layer cost-balanced split of the SAME set.

  python3 balance_score.py HEAT.raw [--layers 49] [--per-helper 9320] [--tokens 2048] [--top-k 10]

Cost of an owned expert in one prefill request of T tokens: lam = T * top_k * share(L, e) expected rows,
    cost = (1 - exp(-lam)) * c0 + c1 * lam                (a fixed weight-tile part when active, plus rows)
and in one decode step (D tokens): lam_d = D * top_k * share, cost_d = (1 - exp(-lam_d)) * d0.
Both are scored; the sweep over (c0, d0) shows how sensitive the gain is to the model.
"""
import argparse
import math
import struct
import sys

K_EXPERTS = 512
K_MAX_LAYERS = 128


def load_heat(path):
    raw = open(path, 'rb').read()
    n = len(raw) // 4
    if n != K_MAX_LAYERS * K_EXPERTS:
        sys.exit('%s: want %d counts, read %d' % (path, K_MAX_LAYERS * K_EXPERTS, n))
    return struct.unpack('<%dI' % n, raw)


def today(heat, layers, per_helper):
    """select_owned(): hot-first key = max+1-count, stable ascending sort over the layer-major flat list, helper h takes
    every 2nd candidate until it holds per_helper experts (uniform expert bytes, so a byte budget is a count)."""
    mx = max(heat[l * K_EXPERTS + e] for l in range(layers) for e in range(K_EXPERTS))
    flat = [(mx + 1 - heat[l * K_EXPERTS + e], l, e) for l in range(layers) for e in range(K_EXPERTS)]
    flat.sort(key=lambda t: t[0])                      # python's sort is stable, like std::stable_sort
    owners = [[], []]
    for pos, (_, l, e) in enumerate(flat):
        h = pos % 2
        if len(owners[h]) < per_helper:
            owners[h].append((l, e))
    return owners


def balanced(heat, layers, per_helper, cost, joint):
    """The same jointly owned set, split per layer to minimise the larger helper sum (LPT: heaviest first onto the
    lighter helper), each helper holding exactly per_helper experts overall; ties break on (cost, expert id) so two
    processes compute the same map."""
    by_layer = {}
    for l, e in joint:
        by_layer.setdefault(l, []).append(e)
    cap = [per_helper, per_helper]
    owners = [[], []]
    for l in sorted(by_layer):
        items = sorted(by_layer[l], key=lambda e: (-cost(l, e), e))
        load = [0.0, 0.0]
        for e in items:
            c = cost(l, e)
            if cap[0] == 0:
                h = 1
            elif cap[1] == 0:
                h = 0
            else:
                h = 0 if load[0] <= load[1] else 1
            owners[h].append((l, e))
            load[h] += c
            cap[h] -= 1
    return owners


def score(owners, cost, layers):
    per = [{}, {}]
    for h in (0, 1):
        for l, e in owners[h]:
            per[h][l] = per[h].get(l, 0.0) + cost(l, e)
    total_max = sum(max(per[0].get(l, 0.0), per[1].get(l, 0.0)) for l in range(layers))
    total_sum = sum(per[0].get(l, 0.0) + per[1].get(l, 0.0) for l in range(layers))
    return total_max, total_sum / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('heat')
    ap.add_argument('--layers', type=int, default=49)
    ap.add_argument('--per-helper', type=int, default=9320)
    ap.add_argument('--tokens', type=int, default=2048, help='prefill tokens a request')
    ap.add_argument('--decode', type=int, default=11, help='tokens a decode step')
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--c1', type=float, default=0.09, help='us a routed row')
    a = ap.parse_args()

    heat = load_heat(a.heat)
    layer_sum = [sum(heat[l * K_EXPERTS:(l + 1) * K_EXPERTS]) or 1 for l in range(a.layers)]

    def share(l, e):
        return heat[l * K_EXPERTS + e] / layer_sum[l]

    cur = today(heat, a.layers, a.per_helper)
    joint = cur[0] + cur[1]
    print('layers %d, per helper %d, jointly owned %d of %d' % (a.layers, a.per_helper, len(joint), a.layers * K_EXPERTS))
    print('%-34s %12s %12s %9s %8s' % ('cost model (us)', 'alternate', 'balanced', 'bound', 'gain'))
    for c0 in (0.0, 5.0, 15.0, 30.0):
        for d0 in (0.0, 20.0):
            def cost(l, e, c0=c0, d0=d0):
                lam = a.tokens * a.top_k * share(l, e)
                lam_d = a.decode * a.top_k * share(l, e)
                return (1 - math.exp(-lam)) * c0 + a.c1 * lam + (1 - math.exp(-lam_d)) * d0

            alt = score(cur, cost, a.layers)
            bal = score(balanced(heat, a.layers, a.per_helper, cost, joint), cost, a.layers)
            print('c0=%-5.0f d0=%-5.0f (prefill+decode) %12.0f %12.0f %9.0f %7.1f%%'
                  % (c0, d0, alt[0], bal[0], bal[1], 100.0 * (alt[0] - bal[0]) / alt[0]))
    sizes = [len(cur[0]), len(cur[1])]
    print('experts a helper (today): %s' % sizes)


if __name__ == '__main__':
    main()
