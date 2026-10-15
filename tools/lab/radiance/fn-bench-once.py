#!/usr/bin/env python3
"""One streamed greedy completion against a radiance server: prefill and decode rates, and the completion text
(written to <out-dir>/completion.txt so two arms can be checked for identical output).
Usage: fn-bench-once.py <base-url> <prompt-file> <max-tokens> <out-dir>"""
import json
import sys
import time
import urllib.request

base, prompt_path, n, out_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
text = open(prompt_path, encoding="utf-8", errors="replace").read()[:24000]
body = {"prompt": text, "max_tokens": n, "temperature": 0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True}}
req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t0 = time.monotonic()
first = last = None
usage = {}
texts = []
with urllib.request.urlopen(req, timeout=1800) as resp:
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        d = json.loads(data)
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices") or []:
            s = ch.get("text") or ""
            if s:
                texts.append(s)
                now = time.monotonic()
                first = first or now
                last = now
open(f"{out_dir}/completion.txt", "w", encoding="utf-8").write("".join(texts))
prompt = int(usage.get("prompt_tokens") or 0)
gen = int(usage.get("completion_tokens") or 0)
if not first or prompt <= 0 or gen <= 1:
    raise SystemExit(f"BENCH_INCOMPLETE prompt={prompt} gen={gen}")
print(f"prefill {prompt / (first - t0):.1f} tok/s ({prompt} tokens)")
print(f"decode {(gen - 1) / max(last - first, 1e-9):.1f} tok/s ({gen} tokens)")
