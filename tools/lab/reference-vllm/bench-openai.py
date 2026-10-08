#!/usr/bin/env python3
"""Prefill and decode speed of an OpenAI-compatible server, measured the way the BigCherry lab measures llama-server.

For each prompt depth: one uncached request (unique nonce first, so no prefix cache), streamed, greedy.
  prefill t/s = prompt tokens / time to first token
  decode t/s  = (completion tokens - 1) / (time of last token - time of first token)
Prompt text is the lab corpus (kld-docs.txt) cut to the requested size, so depths are comparable with
long-ctx-profile.sh runs. Usage: bench-openai.py <base-url> <model> <out.json> <depth-tokens>... [--decode N] [--reps N]
"""
import json
import sys
import time
import urllib.request
import uuid

CORPUS = "/mnt/data/bigcherry-work/corpus/kld-docs.txt"
CHARS_PER_TOKEN = 3.6  # corpus average for the Qwen tokenizer; the server reports the real prompt token count


def stream(base, body):
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    n_chunks = 0
    usage = {}
    text = []
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            item = json.loads(payload)
            if item.get("usage"):
                usage = item["usage"]
            choices = item.get("choices") or []
            if choices and choices[0].get("text"):
                now = time.time()
                first = first or now
                last = now
                n_chunks += 1
                text.append(choices[0]["text"])
    return t0, first, last, usage, "".join(text), n_chunks


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = sys.argv[1:]
    decode = int(opts[opts.index("--decode") + 1]) if "--decode" in opts else 512
    reps = int(opts[opts.index("--reps") + 1]) if "--reps" in opts else 2
    base, model, out = args[0].rstrip("/"), args[1], args[2]
    depths = [int(a) for a in args[3:] if a.isdigit()]
    corpus = open(CORPUS, errors="replace").read()
    rows = []
    for depth in depths:
        for rep in range(reps):
            nonce = uuid.uuid4().hex
            prompt = f"[{nonce}]\n" + corpus[: int(depth * CHARS_PER_TOKEN)] + "\n\nSummarise the above in detail:"
            body = {"model": model, "prompt": prompt, "max_tokens": decode, "temperature": 0, "stream": True,
                    "stream_options": {"include_usage": True}}
            try:
                t0, first, last, usage, text, n_chunks = stream(base, body)
            except Exception as exc:  # report and continue with the next depth
                print(f"d{depth} r{rep}: REQUEST_FAILED {exc}", flush=True)
                break
            p = usage.get("prompt_tokens") or 0
            c = usage.get("completion_tokens") or 0
            ttft = (first - t0) if first else float("nan")
            dec = (c - 1) / (last - first) if first and last and last > first and c > 1 else float("nan")
            row = {"depth": depth, "rep": rep, "prompt_tokens": p, "completion_tokens": c, "ttft_s": ttft,
                   "prefill_tps": p / ttft if first else float("nan"), "decode_tps": dec, "text_head": text[:160]}
            rows.append(row)
            print(f"d{depth} r{rep}: prompt {p} tok, ttft {ttft:.2f} s = {row['prefill_tps']:.1f} t/s prefill, "
                  f"decode {c} tok at {dec:.1f} t/s", flush=True)
    json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
