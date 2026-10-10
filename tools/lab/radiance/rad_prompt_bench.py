#!/usr/bin/env python3
"""Mixed-prompt bench against a running OpenAI-compatible server (radiance or any other).

The serve smoke test's two prompts (a list of capitals, counting to ten) are so predictable that a drafter accepts
almost everything, so they overstate decode speed. This sends a fixed set of varied requests, each uncached, greedy,
streamed, and reports per request: prompt tokens, time to first token, prefill tok/s (prompt tokens / time to first
token), decode tok/s (completion tokens after the first / time from the first token to the last). Then the
geometric mean of decode tok/s over the short prompts and the prefill tok/s of the long ones.

Short prompts are chat requests with thinking off. Long prompts are the lab corpus cut to a token depth with a
summarise request after it, as tools/bigcherry/tuning/engine_bench.py builds them, so prefill here is comparable with
the engine bench.

Usage: rad_prompt_bench.py <base-url> [--tokens N] [--depths 2048,6000] [--corpus FILE] [--json out.json]
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.request
import uuid

SHORT = [
    ("code", "Write a Python function that merges overlapping intervals given as a list of (start, end) tuples. "
             "Include type hints and a short docstring, then show two example calls with their results."),
    ("explain", "Explain how a B-tree differs from a binary search tree, and why databases prefer B-trees for "
                "on-disk indexes. Keep it to three paragraphs."),
    ("story", "Write the opening of a short story about a lighthouse keeper who finds a message in a bottle "
              "written in a language nobody on the island can read."),
    ("reason", "A train leaves a station at 09:15 travelling at 84 km/h. A second train leaves the same station at "
               "09:50 on the same line at 112 km/h. At what time does the second train catch the first, and how "
               "far from the station? Show the working."),
    ("translate", "Translate into French, then into German: 'The committee postponed its decision until the "
                  "auditors had finished reviewing last year's accounts, which nobody had expected to take so long.'"),
    ("review", "Here is a C function. Point out the bugs and give a corrected version.\n\n"
               "int sum(int *a, int n) {\n    int s;\n    for (int i = 0; i <= n; i++)\n        s += a[i];\n"
               "    return s;\n}\n"),
    ("list", "Give a step-by-step plan for migrating a small web service from a single virtual machine to "
             "containers, with the risks at each step."),
    ("json", "Produce a JSON array of five fictional books, each with title, author, year, and a one-sentence "
             "summary. Output only the JSON."),
]
ASK = "\n\nSummarise the above in detail:"
CHARS_PER_TOKEN = 3.6


def stream(base: str, path: str, body: dict, timeout: int = 3600) -> dict:
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    started = time.time()
    first = last = None
    usage, text = {}, []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                piece = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                if piece:
                    now = time.time()
                    first = first or now
                    last = now
                    text.append(piece)
    prompt, completion = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    ttft = (first - started) if first else None
    span = (last - first) if first and last and last > first else None
    return {"prompt_tokens": prompt, "completion_tokens": completion, "ttft_s": ttft,
            "prefill_tps": prompt / ttft if ttft else None,
            "decode_tps": (completion - 1) / span if span and completion > 1 else None,
            "text": "".join(text)}


def geomean(values: list[float]) -> float | None:
    values = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        print(__doc__)
        return 2
    base = args[0].rstrip("/")

    def opt(name: str, default: str) -> str:
        return args[args.index(name) + 1] if name in args else default

    tokens = int(opt("--tokens", "256"))
    depths = [int(d) for d in opt("--depths", "2048,6000").split(",") if d]
    corpus_path = opt("--corpus", "/mnt/data/bigcherry-work/corpus/kld-docs.txt")
    model = json.load(urllib.request.urlopen(base + "/v1/models", timeout=60))["data"][0]["id"]
    rows = []
    for name, prompt in SHORT:
        nonce = uuid.uuid4().hex[:8]
        r = stream(base, "/v1/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": f"[{nonce}] {prompt}"}],
            "max_tokens": tokens, "temperature": 0, "reasoning_effort": "none"})
        r["name"], r["kind"] = name, "short"
        rows.append(r)
    corpus = open(corpus_path, errors="replace").read() if depths else ""
    for depth in depths:
        r = stream(base, "/v1/completions", {
            "model": model, "prompt": f"[{uuid.uuid4().hex}]\n" + corpus[: int(depth * CHARS_PER_TOKEN)] + ASK,
            "max_tokens": tokens, "temperature": 0})
        r["name"], r["kind"] = f"corpus-{depth}", "long"
        rows.append(r)

    def show(value, digits=1):
        return "-" if value is None else f"{value:.{digits}f}"

    print(f"{'request':<14}{'prompt':>8}{'gen':>6}{'ttft s':>9}{'prefill t/s':>13}{'decode t/s':>12}  first words")
    for r in rows:
        print(f"{r['name']:<14}{r['prompt_tokens']:>8}{r['completion_tokens']:>6}{show(r['ttft_s'], 3):>9}"
              f"{show(r['prefill_tps']):>13}{show(r['decode_tps']):>12}  {' '.join(r['text'].split())[:48]!r}")
    short = [r["decode_tps"] for r in rows if r["kind"] == "short"]
    long_ = [r for r in rows if r["kind"] == "long"]
    print(f"decode, geometric mean over {len(short)} short prompts: {show(geomean(short))} tok/s "
          f"(min {show(min([s for s in short if s], default=None))}, max {show(max([s for s in short if s], default=None))})")
    for r in long_:
        print(f"prefill at {r['prompt_tokens']} tokens: {show(r['prefill_tps'])} tok/s; decode after it {show(r['decode_tps'])} tok/s")
    if "--json" in args:
        json.dump({"model": model, "rows": rows}, open(args[args.index("--json") + 1], "w", encoding="utf-8"), indent=1)
    return 0 if all(r["completion_tokens"] > 0 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
