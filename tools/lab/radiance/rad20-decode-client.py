#!/usr/bin/env python3
"""RAD20 decode client: per-request prefill and decode timings and MTP acceptance from a radiance server.

The bench's tg128 is the 128-token decode that follows each pp256/pp1024/pp4096 prompt, so a prefill-side change
(e.g. R3_MOE_WORKER_COMBINE, which fires only on prefill steps) can move it either by changing the generated text or
by leaving something behind that slows the decode. Each request records the server's own timings (prompt and
predicted ms and rates, draft_n / draft_n_accepted) and a hash of the generated text, so two arms can be compared
prompt by prompt: the same text with a different decode rate is not a content effect.
Prompts: N slices of PROMPT_FILE, each CHARS characters, the same slices in every arm.
Usage: rad20-decode-client.py URL PROMPT_FILE N CHARS MAX_TOKENS OUT.jsonl
"""
import hashlib
import json
import sys
import urllib.request


def run(url, prompt, max_tokens):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    text = resp["choices"][0]["text"]
    t = resp.get("timings", {})
    keep = ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second",
            "draft_n", "draft_n_accepted")
    out = {k: t.get(k) for k in keep}
    out["text_sha"] = hashlib.sha256(text.encode()).hexdigest()[:16]
    out["text_head"] = text[:60]
    return out


def main():
    url, prompt_file, n, chars, max_tokens, out_path = sys.argv[1:7]
    n, chars, max_tokens = int(n), int(chars), int(max_tokens)
    src = open(prompt_file, errors="replace").read()
    step = max(1, (len(src) - chars) // max(1, n))
    with open(out_path, "w") as out:
        for i in range(n):
            r = run(url, src[i * step:i * step + chars] + "\n\nContinue:", max_tokens)
            r["prompt"] = i
            out.write(json.dumps(r) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
