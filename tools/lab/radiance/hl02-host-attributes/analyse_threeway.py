#!/usr/bin/env python3
"""Summarise HL02 Gate B sessions: per arm and size, median over sessions of each session's p50 (and p95) makespan
and per-card time. Fails on any incomplete session or verification error."""
import json
import statistics as st
import sys
from collections import defaultdict

ARMS = ["pull_all", "push_all", "push_one_pull_rest", "push_pair_pull_rest", "push_all_producers", "pull_all_producers", "cpu_sum3"]


def pct(v, q):
    v = sorted(v)
    k = (len(v) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main(paths):
    sess = []
    for p in paths:
        rows, complete, devs = [], None, {}
        for line in open(p):
            d = json.loads(line)
            if "complete" in d:
                complete = d
            elif "device" in d:
                devs[d["device"]] = d["pci"]
            else:
                rows.append(d)
        if not complete or complete["bad_words"] != 0 or any(r.get("bad", 0) for r in rows):
            sys.exit(f"{p}: incomplete or verification errors")
        sess.append(rows)
    print(f"sessions {len(sess)}, all words verified on the GPU, 0 errors")
    ser = sum(r.get("serialized", 0) for rows in sess for r in rows)
    print(f"transfers that ran behind a producer rather than beside it (excluded): {ser}")
    sess = [[r for r in rows if not r.get("serialized")] for rows in sess]
    sizes = sorted({r["bytes"] for r in sess[0]})
    print("| arm | bytes | makespan p50 us | p95 us | per card p50 us (dev 0 / 1 / 2) | per card GB/s p50 |")
    print("|---|---|---|---|---|---|")
    for arm in ARMS:
        for s in sizes:
            mk50, mk95, cards = [], [], defaultdict(list)
            for rows in sess:
                rr = [r for r in rows if r["arm"] == arm and r["bytes"] == s and r["iter"] > 0]
                if not rr:
                    continue
                mk = [r["makespan_us"] for r in rr]
                mk50.append(pct(mk, .5))
                mk95.append(pct(mk, .95))
                for c in (rr[0].get("cards") or []):
                    cards[c["dev"]].append(pct([x["us"] for r in rr for x in r["cards"] if x["dev"] == c["dev"]], .5))
            if not mk50:
                continue
            if arm == "cpu_sum3":
                per, gbs = "-", f"read {3 * s / st.median(mk50) / 1e3:.1f}"
            else:
                med = {d: st.median(v) for d, v in sorted(cards.items())}
                modes = {c["dev"]: c["mode"] for c in next(r for r in sess[0] if r["arm"] == arm)["cards"]}
                per = " / ".join(f"{med[d]:.0f}{'p' if modes[d] == 'cpu_push' else 'g'}" for d in med)
                gbs = " / ".join(f"{s / med[d] / 1e3:.2f}" for d in med)
            print(f"| {arm} | {s} | {st.median(mk50):.0f} | {st.median(mk95):.0f} | {per} | {gbs} |")


if __name__ == "__main__":
    main(sys.argv[1:])
