#!/usr/bin/env python3
"""Next-token distributions from any OpenAI-compatible server, for comparing engines on the same prompts (MEN05).

The prompts are the ones long-ctx-profile.sh uses in `probes` mode: the lab corpus cut to the requested depth, then
PROBES natural-text continuations appended one at a time. For each probe the server is asked for one token at
temperature 0 with its top alternatives, through /v1/completions, so llama-server, radiance and vLLM answer the same
request. Only standard request fields are sent.

Output: a JSON list, one entry per probe: {"top": [{"key": <token bytes as hex, or the token text>, "token": <text>,
"logprob": <float>}, ...]}. The key is the token's bytes where the server reports them, so two engines agree on a
token even if they print it differently. Compare files with probes-compare.py.

Usage: probe-openai.py <base-url> <model> <out.json> <depth-tokens> [--probes N] [--top N]
"""
import json
import sys
import urllib.request

CORPUS = "/mnt/data/bigcherry-work/corpus/kld-docs.txt"


def post(base, body):
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=7200).read())


def alternatives(choice):
    """The first generated token's alternatives as [(key, text, logprob)], from either logprobs shape in use."""
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content")
    if content:  # chat-style shape: content[0].top_logprobs = [{token, logprob, bytes?, id?}, ...]
        first = content[0]
        rows = first.get("top_logprobs") or [first]
        out = []
        for row in rows:
            raw = row.get("bytes")
            key = bytes(raw).hex() if isinstance(raw, list) else "t:" + row["token"]
            out.append((key, row["token"], float(row["logprob"])))
        return out
    top = logprobs.get("top_logprobs")
    if top:  # legacy completions shape: top_logprobs[0] = {token text: logprob}
        return [(text.encode("utf-8").hex(), text, float(lp)) for text, lp in top[0].items()]
    return []


def main() -> int:
    opts = sys.argv[1:]
    probes = int(opts[opts.index("--probes") + 1]) if "--probes" in opts else 24
    top = int(opts[opts.index("--top") + 1]) if "--top" in opts else 20
    args = [a for i, a in enumerate(opts) if not a.startswith("--") and (i == 0 or opts[i - 1] not in ("--probes", "--top"))]
    base, model, out, depth = args[0].rstrip("/"), args[1], args[2], int(args[3])
    corpus = open(CORPUS, errors="replace").read()
    text = (corpus * (1 + 4 * depth // max(1, len(corpus))))[: 4 * depth]  # ~4 chars/token, as long-ctx-profile.sh
    results = []
    prompt_tokens = None
    for i in range(probes):
        off = (i * 104729) % max(1, len(corpus) - 2000)
        body = {"model": model, "prompt": text + "\n\n" + corpus[off:off + 600], "max_tokens": 1, "temperature": 0,
                "logprobs": top}
        reply = post(base, body)
        prompt_tokens = (reply.get("usage") or {}).get("prompt_tokens", prompt_tokens)
        rows = alternatives(reply["choices"][0])
        results.append({"top": [{"key": k, "token": t, "logprob": lp} for k, t, lp in rows]})
    json.dump(results, open(out, "w"))
    empty = sum(1 for r in results if not r["top"])
    print(f"d{depth}: {len(results)} probes saved ({empty} without alternatives), last prompt {prompt_tokens} tokens", flush=True)
    return 1 if empty else 0


if __name__ == "__main__":
    sys.exit(main())
