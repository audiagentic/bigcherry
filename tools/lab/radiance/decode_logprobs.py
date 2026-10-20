#!/usr/bin/env python3
"""Decode-path quality check from the server's own logprobs (RAD15).

radiance's KL mode scores prefill only, at the model's minimum chunk (64 tokens on the 35B and Flash-Next), so it never
reaches a decode-only kernel row. This compares two arms on the decode path instead: each arm serves the same prompts
greedily (temperature 0, fixed seed) with top-20 logprobs on every generated token.

  capture URL OUT [--max-tokens 256]   query a running server, write the per-token top-20 logprobs of every prompt
  compare A.json B.json                over each prompt's shared prefix (tokens until the arms first pick differently):
                                       KL(A || B) on the top-20 support (the rest of each distribution's mass as one
                                       bucket), top-1 agreement, and at the first divergence the margin between A's top
                                       two candidates in A and in B (a flip between near-tied candidates is a rounding
                                       difference; a flip with a wide margin is a defect).

After a divergence the arms decode different texts and their distributions are no longer comparable, so tokens past it
are not scored. Use several prompts so the shared prefixes add up.
"""
import argparse
import json
import math
import sys
import urllib.request

PROMPTS = [
    "Explain how a hash table handles collisions, with a short example in Python.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "List the planets of the solar system and one distinctive fact about each.",
    "What are the main causes of the French Revolution? Answer in a few paragraphs.",
    "Translate into French: 'The weather is lovely today, let us walk to the market.' Then explain the grammar.",
    "Solve step by step: a train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive?",
    "Write a C function that reverses a singly linked list, and explain its complexity.",
    "Summarise the plot of Hamlet in under 200 words.",
]


def capture(url, out, max_tokens, model):
    rows = []
    for i, prompt in enumerate(PROMPTS):
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                           "seed": 1, "max_tokens": max_tokens, "logprobs": True, "top_logprobs": 20}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            resp = json.load(r)
        content = resp["choices"][0]["logprobs"]["content"]
        rows.append({"prompt": i, "tokens": [{"token": c["token"], "logprob": c["logprob"],
                                               "top": {t["token"]: t["logprob"] for t in c["top_logprobs"]}}
                                              for c in content]})
        print("prompt %d: %d tokens" % (i, len(content)), file=sys.stderr)
    json.dump(rows, open(out, "w"))


def kl_top(pa, pb):
    """KL(a || b) over the union of the two top-k supports plus one bucket for the remaining mass of each."""
    keys = set(pa) | set(pb)
    floor_a = min(pa.values())
    floor_b = min(pb.values())
    qa = {k: math.exp(pa.get(k, floor_a)) for k in keys}
    qb = {k: math.exp(pb.get(k, floor_b)) for k in keys}
    ra, rb = max(1e-12, 1 - sum(math.exp(v) for v in pa.values())), max(1e-12, 1 - sum(math.exp(v) for v in pb.values()))
    kl = sum(p * math.log(p / max(qb[k], 1e-12)) for k, p in qa.items() if p > 0)
    return kl + ra * math.log(ra / rb)


def margin(top):
    v = sorted(top.values(), reverse=True)
    return v[0] - v[1] if len(v) > 1 else float("inf")


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    kls, agree, total, flips = [], 0, 0, []
    for ra, rb in zip(a, b):
        for n, (ta, tb) in enumerate(zip(ra["tokens"], rb["tokens"])):
            kls.append(kl_top(ta["top"], tb["top"]))
            total += 1
            if ta["token"] == tb["token"]:
                agree += 1
                continue
            flips.append((ra["prompt"], n, margin(ta["top"]), margin(tb["top"])))
            break
    kls.sort()
    pct = lambda q: kls[min(len(kls) - 1, int(q * len(kls)))] if kls else float("nan")
    print("shared-prefix tokens %d over %d prompts; top-1 agreement %.2f%%" % (total, len(a), 100.0 * agree / max(total, 1)))
    print("KL(A||B) mean %.2e median %.2e p99 %.2e max %.2e" % (sum(kls) / max(len(kls), 1), pct(0.5), pct(0.99),
                                                                kls[-1] if kls else float("nan")))
    for p, n, ma, mb in flips:
        print("prompt %d diverges at token %d: top-1/top-2 logprob margin %.3f in A, %.3f in B" % (p, n, ma, mb))
    print("prompts with no divergence: %d" % (len(a) - len(flips)))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("url")
    c.add_argument("out")
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("--model", default="default")
    d = sub.add_parser("compare")
    d.add_argument("a")
    d.add_argument("b")
    args = ap.parse_args(argv)
    if args.cmd == "capture":
        capture(args.url, args.out, args.max_tokens, args.model)
    else:
        compare(args.a, args.b)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
