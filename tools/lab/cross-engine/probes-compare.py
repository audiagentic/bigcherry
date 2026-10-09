#!/usr/bin/env python3
"""Compare next-token distributions from probe-openai.py files against a reference arm (MEN05).

Every arm saw the same prompts. For each arm against the reference: probes compared, top-1 agreement, mean and max
total-variation distance over the listed tokens (a token missing from one list counts as probability 0 there), the
mean absolute difference of the reference's top-1 probability, and the mean probability mass each side's list covers
(a list that covers little mass makes the distance an underestimate). With more than one arm, the arms are also
compared with each other.

Usage: probes-compare.py <ref-name>=<file> <name>=<file> [<name>=<file>...]
"""
import json
import math
import sys


def dist(entry):
    return {t["key"]: math.exp(t["logprob"]) for t in entry.get("top", [])}


def compare(name_a, a, name_b, b):
    n = min(len(a), len(b))
    agree, tvs, d1, mass_a, mass_b = 0, [], [], [], []
    for i in range(n):
        pa, pb = dist(a[i]), dist(b[i])
        if not pa or not pb:
            continue
        top_a = max(pa, key=pa.get)
        top_b = max(pb, key=pb.get)
        agree += top_a == top_b
        tvs.append(0.5 * sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in pa.keys() | pb.keys()))
        d1.append(abs(pa[top_a] - pb.get(top_a, 0.0)))
        mass_a.append(sum(pa.values()))
        mass_b.append(sum(pb.values()))
    if not tvs:
        print(f"{name_b} vs {name_a}: no comparable probes")
        return
    k = len(tvs)
    print(f"{name_b} vs {name_a}: {k} probes, top-1 agree {agree}/{k}, TV mean {sum(tvs)/k:.4f} max {max(tvs):.4f}, "
          f"|d p(top-1)| mean {sum(d1)/k:.4f} max {max(d1):.4f}, listed mass {sum(mass_b)/k:.3f} vs {sum(mass_a)/k:.3f}")


def main() -> int:
    arms = [(arg.split("=", 1)[0], json.load(open(arg.split("=", 1)[1]))) for arg in sys.argv[1:]]
    if len(arms) < 2:
        print(__doc__)
        return 2
    ref_name, ref = arms[0]
    for name, data in arms[1:]:
        compare(ref_name, ref, name, data)
    for i in range(1, len(arms)):
        for j in range(i + 1, len(arms)):
            compare(arms[i][0], arms[i][1], arms[j][0], arms[j][1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
