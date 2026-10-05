#!/usr/bin/env python3
"""Fidelity of next-token distributions against a reference arm.

Input: probes JSON files written by long-ctx-profile.sh in `probes` mode - one entry per probe, each the server's
completion_probabilities record for the first generated token (top_logprobs of 10). Every arm sees exactly the same
prompts, so the distributions are directly comparable.

For each arm against the reference: probes compared, top-1 agreement, mean and max total-variation distance over the
listed tokens (a token missing from one list counts as probability 0 there), and the mean absolute difference of the
reference's top-1 probability. With more than one arm it also prints the arms against each other.

Usage: probes-compare.py <ref-name>=<file> <name>=<file> [<name>=<file>...]"""
import json
import math
import sys


def dist(entry):
    return {t["id"]: math.exp(t["logprob"]) for t in entry.get("top_logprobs", [])}


def compare(name_a, a, name_b, b):
    n = min(len(a), len(b))
    agree, tvs, d1 = 0, [], []
    for i in range(n):
        pa, pb = dist(a[i]), dist(b[i])
        if not pa or not pb:
            continue
        top_a = max(pa, key=pa.get)
        top_b = max(pb, key=pb.get)
        agree += top_a == top_b
        tvs.append(0.5 * sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in pa.keys() | pb.keys()))
        d1.append(abs(pa[top_a] - pb.get(top_a, 0.0)))
    if not tvs:
        print(f"{name_b} vs {name_a}: no comparable probes")
        return
    print(f"{name_b} vs {name_a}: {len(tvs)} probes, top-1 agree {agree}/{len(tvs)}, "
          f"TV mean {sum(tvs)/len(tvs):.4f} max {max(tvs):.4f}, |d p(top-1)| mean {sum(d1)/len(d1):.4f} max {max(d1):.4f}")


def main() -> int:
    arms = [(arg.split("=", 1)[0], json.load(open(arg.split("=", 1)[1]))) for arg in sys.argv[1:]]
    ref_name, ref = arms[0]
    for name, data in arms[1:]:
        compare(ref_name, ref, name, data)
    for i in range(1, len(arms)):
        for j in range(i + 1, len(arms)):
            compare(arms[i][0], arms[i][1], arms[j][0], arms[j][1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
