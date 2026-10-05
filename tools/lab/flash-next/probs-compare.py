#!/usr/bin/env python3
"""Compare the per-token output distributions of two greedy runs (llama-server completion_probabilities JSON, as
saved by long-ctx-profile.sh with REPEAT=1). A text checksum cannot tell a near-tie flip from a wrong result; this
reports, over the positions where both runs chose the same token, the largest probability difference among the
tokens both runs list, and at the first position where they diverge the top-1/top-2 margin in each run.

Usage: probs-compare.py <label> <a.probs.json> <b.probs.json>
Prints one summary line: label, positions compared, agreed prefix length, max |dp| over the agreed prefix, and the
divergence (if any) with both margins. Exit status is always 0; the caller decides what is acceptable."""
import json
import math
import sys


def dist(entry):
    return {t["id"]: math.exp(t["logprob"]) for t in entry.get("top_logprobs", [])}


def margin(entry):
    probs = sorted((math.exp(t["logprob"]) for t in entry.get("top_logprobs", [])), reverse=True)
    return probs[0] - probs[1] if len(probs) > 1 else 1.0


def main() -> int:
    label, path_a, path_b = sys.argv[1:4]
    a, b = json.load(open(path_a)), json.load(open(path_b))
    n = min(len(a), len(b))
    max_dp, agreed, worst = 0.0, 0, None
    for i in range(n):
        if a[i]["id"] != b[i]["id"]:
            pa, pb = dist(a[i]), dist(b[i])
            print(f"{label}: {n} positions, agreed prefix {agreed}, max |dp| {max_dp:.2e}"
                  + (f" (pos {worst})" if worst is not None else "")
                  + f"; DIVERGES at {i}: A '{a[i]['token']}' p={pa.get(a[i]['id'], 0):.4f} margin {margin(a[i]):.4f}"
                  + f" (B gives it {pb.get(a[i]['id'], 0):.4f}), B '{b[i]['token']}' p={pb.get(b[i]['id'], 0):.4f}"
                  + f" margin {margin(b[i]):.4f} (A gives it {pa.get(b[i]['id'], 0):.4f})")
            return 0
        pa, pb = dist(a[i]), dist(b[i])
        for token_id in pa.keys() & pb.keys():
            dp = abs(pa[token_id] - pb[token_id])
            if dp > max_dp:
                max_dp, worst = dp, i
        agreed += 1
    print(f"{label}: {n} positions, agreed prefix {agreed} (all), max |dp| {max_dp:.2e}"
          + (f" (pos {worst})" if worst is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
