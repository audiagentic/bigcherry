#!/usr/bin/env python3
"""Needle retrieval probe for a KV-cache format: a passphrase sentence at a given depth of a long natural-text prompt.

One chat request per case against a running server (the caller owns the server and the GPU lease; this only sends
requests). The haystack is `docs/**/*.md` of this checkout, a different offset per case (so no prefix-cache hit and no
two cases share text), cut to the target token count measured against the server's own tokenizer. The needle is one
sentence with a random three-part code at 10, 50 and 90% depth; the question asks for the code, thinking off,
temperature 0. A hit is the exact code in the answer. The misses a quantised cache can cause are wrong or missing
codes; a refusal is recorded as a miss with its answer.

  python3 tools/lab/radiance/needle_probe.py --url http://127.0.0.1:18450 --tokens 48000 --tokens 100000 \
      --depths 0.1,0.5,0.9 --repeats 3 --out needle.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ANSWER_KEPT = 120   # characters of the model's answer kept in a row
WORDS = ("amber", "birch", "cobalt", "delta", "ember", "falcon", "garnet", "harbor", "indigo", "juniper", "kestrel",
         "lantern", "meadow", "nectar", "onyx", "pewter", "quartz", "raven", "saffron", "tundra", "umber", "velvet",
         "willow", "zephyr")


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def corpus() -> str:
    parts = [p.read_text(encoding="utf-8", errors="replace") for p in sorted((REPO / "docs").rglob("*.md"))]
    return "\n\n".join(parts)


def chat(url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> tuple[dict, float]:
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.perf_counter()
    return post(url, body, timeout), time.perf_counter() - t0


def chars_per_token(url: str, model: str, text: str, timeout: float) -> float:
    """A first estimate from twelve evenly spaced 8K-character samples (the docs vary a lot in density: tables and code
    tokenise far worse than prose). `case` corrects it with the server's own count when a prompt lands off target."""
    step = max(1, (len(text) - 8000) // 12)
    sample = "\n".join(text[i:i + 8000] for i in list(range(0, len(text) - 8000, step))[:12])
    reply, _ = chat(url, model, sample, 1, timeout)
    tokens = int((reply.get("usage") or {}).get("prompt_tokens") or 0)
    if tokens <= 0:
        raise SystemExit("needle_probe: the server reported no prompt_tokens")
    return len(sample) / tokens


def build(text: str, cpt: float, tokens: int, depth: float, rng: random.Random, code: str) -> str:
    needle = f"The access passphrase for the vault is {code}."
    chars = int(tokens * cpt) - 400
    start = rng.randrange(0, max(1, len(text) - chars))
    hay = text[start:start + chars]
    cut = hay.find("\n\n", int(len(hay) * depth))
    cut = cut if cut >= 0 else int(len(hay) * depth)
    return (hay[:cut] + "\n\n" + needle + "\n\n" + hay[cut:] +
            "\n\nWhat is the access passphrase for the vault? Answer with the passphrase only.")


def case(url: str, model: str, text: str, cpt: float, tokens: int, depth: float, rep: int, seed: int,
         timeout: float, tolerance: float = 0.08) -> dict:
    rng = random.Random(seed * 7919 + rep * 104729 + int(depth * 1000) + tokens)
    code = "-".join([rng.choice(WORDS), str(rng.randrange(1000, 9999)), rng.choice(WORDS)])
    attempts = 0
    while True:
        attempts += 1
        prompt = build(text, cpt, tokens, depth, random.Random(rng.random()), code)
        reply, seconds = chat(url, model, prompt, 40, timeout)
        actual = int((reply.get("usage") or {}).get("prompt_tokens") or 0)
        # a prompt off target by more than the tolerance is rebuilt once with this region's own ratio
        if actual and abs(actual - tokens) / tokens > tolerance and attempts < 3:
            cpt = len(prompt) / actual
            continue
        break
    answer = (reply["choices"][0]["message"].get("content") or "").strip()
    return {"tokens_target": tokens, "prompt_tokens": actual, "depth": depth, "rep": rep, "code": code,
            "hit": code in answer, "answer": answer[:ANSWER_KEPT], "seconds": round(seconds, 2), "attempts": attempts}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="native bc-serve run directory; required for immutable measurement provenance")
    ap.add_argument("--work-root", type=Path,
                    help="BC_WORK_ROOT MRC01 record store (defaults to environment)")
    ap.add_argument("--model", default=None, help="the served model name (default: the first id of /v1/models)")
    ap.add_argument("--tokens", type=int, action="append", required=True, help="target prompt tokens (repeatable)")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    tools = str(REPO / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from bigcherry.bench.radiance_lab_records import verify_live_target
    from bigcherry.bench.served_model import served_model

    verify_live_target(args.run_dir, args.url)
    args.model = args.model or served_model(args.url)
    text = corpus()
    cpt = chars_per_token(args.url, args.model, text, args.timeout)
    rows = []
    for tokens in args.tokens:
        for depth in [float(d) for d in args.depths.split(",")]:
            for rep in range(args.repeats):
                row = case(args.url, args.model, text, cpt, tokens, depth, rep, args.seed, args.timeout)
                rows.append(row)
                print(f"{tokens:>7} depth {depth:.1f} rep {rep}: {'HIT ' if row['hit'] else 'MISS'} {row['seconds']:7.1f}s  "
                      f"{row['answer'][:60]!r}", flush=True)
    summary = {}
    for tokens in args.tokens:
        mine = [r for r in rows if r["tokens_target"] == tokens]
        summary[str(tokens)] = {"hits": sum(r["hit"] for r in mine), "of": len(mine)}
    # Original rows are retained once in the immutable record, not rewritten
    # to a second mutable JSON metrics store. --out is a digest-only pointer.
    from bigcherry.bench.radiance_lab_records import publish_probe

    result = {"chars_per_token": cpt, "summary": summary, "rows": rows}
    digest = publish_probe(
        args.run_dir, kind="needle", result=result,
        definition={"tokens": args.tokens, "depths": [float(d) for d in args.depths.split(",")],
                    "repeats": args.repeats, "seed": args.seed,
                    "tolerance": 0.08, "max_tokens": 40},
        argv=[str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
        extra_inputs={"corpus": text.encode("utf-8")},
        work_root=args.work_root,
    )
    args.out.write_text(json.dumps({"record_sha256": digest}, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "record_sha256": digest}))
    return 0 if all(r["hit"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
