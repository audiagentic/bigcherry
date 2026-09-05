"""Summarise a replay-bench-balanced.sh log.

Reports per-arm means, the replay-vs-control delta, and whether the two arms'
sample ranges overlap at all. Complete separation with n=6 per arm is a
Mann-Whitney U result at p = 2/C(12,6) ~= 0.0022, which is worth far more here
than a t-test would be: the samples are small and the between-session drift on
this host is not obviously normal.

Also prints the per-position means, because the whole reason this harness
exists is that the previous unbalanced run could not distinguish an arm effect
from a position effect. If position means differ materially, the balancing is
doing real work and any unbalanced result is void.

Usage: python3 analyse.py <log-path>
"""

import re
import statistics as st
import sys
from itertools import combinations

METRICS = ["tg128", "tg512", "tg2048", "pp1024", "pp4096", "pp256"]


def parse(path):
    rows = []
    for line in open(path):
        m = re.match(r"round=(\d+) pos=(\d+) arm=(\S+)\s+(.*)", line.strip())
        if not m:
            continue
        d = {k: float(v) for k, v in re.findall(r"(\w+)_tps: ([\d.]+)", m.group(4))}
        if d:
            rows.append((int(m.group(1)), int(m.group(2)), m.group(3), d))
    return rows


def main(path):
    rows = parse(path)
    arms = sorted({a for _, _, a, _ in rows})
    base = "control" if "control" in arms else arms[0]
    print(f"n={len(rows)} cells, arms={arms}, baseline={base}\n")

    head = " ".join(f"{a:>10}" for a in arms)
    print(f"{'metric':8} {head}   vs-{base:<8} ranges")
    for mt in METRICS:
        vals = {a: [d[mt] for _, _, ar, d in rows if ar == a and mt in d] for a in arms}
        if not all(vals.values()):
            continue
        means = {a: st.mean(v) for a, v in vals.items()}
        others = [a for a in arms if a != base]
        line = f"{mt:8} " + " ".join(f"{means[a]:10.2f}" for a in arms)
        deltas = ", ".join(
            f"{a}:{100 * (means[a] - means[base]) / means[base]:+.2f}%" for a in others
        )
        seps = ",".join(
            a for a in others
            if max(vals[a]) < min(vals[base]) or min(vals[a]) > max(vals[base])
        )
        print(f"{line}   {deltas}  {'SEPARATED[' + seps + ']' if seps else 'overlap'}")

    print("\nposition means (a large spread here means order matters and any"
          "\nunbalanced run is uninterpretable):")
    positions = sorted({p for _, p, _, _ in rows})
    for mt in ("tg128", "pp4096"):
        cells = []
        for p in positions:
            v = [d[mt] for _, pp, _, d in rows if pp == p and mt in d]
            if v:
                cells.append(f"pos{p} {st.mean(v):8.2f}")
        print(f"  {mt:7} " + "  ".join(cells))

    print("\noutliers (>3% from that arm's median -- pp256 in particular is"
          "\nstartup-sensitive and should not drive any conclusion):")
    found = False
    for mt in METRICS:
        for a in arms:
            v = [d[mt] for _, _, ar, d in rows if ar == a and mt in d]
            if len(v) < 3:
                continue
            med = st.median(v)
            for x in v:
                if abs(x - med) / med > 0.03:
                    print(f"  {mt} {a}: {x:.2f} vs median {med:.2f}")
                    found = True
    if not found:
        print("  none")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "replay-balanced.log")
