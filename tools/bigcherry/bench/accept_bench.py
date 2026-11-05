"""Draft acceptance by prompt kind, one request at a time, temperature 0 (RAD23's acceptance probe, kept).

Speculative decode is only as fast as the text lets it be: the same build accepts 49% of its drafts on a story and 96%
on a counting list. A decode figure is therefore only comparable across builds when the prompt is pinned, and the
recorded Flash-Next bests were not. This sends five fixed prompts to /v1/completions (no chat template, `ignore_eos`,
temperature 0, 512 tokens unless told otherwise), reads the engine's counters before and after each, and records per
prompt kind the drafted and accepted tokens, tokens and milliseconds per decode step, the SHA-256 of the prompt and of
the text produced (so identity between two builds is a comparison of digests). Prompts are the ones RAD23 used, so its
recorded 65 / 49 / 91 / 96 / 89 % (hello / story / code / list / pi digits) can be reproduced on any later build."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .stream_bench import engine_delta, scrape_metrics

SCHEMA = "bigcherry/accept-bench/v1"
#: kind -> prompt, exactly as RAD23 sent them (a change here makes a run incomparable with its record)
KINDS = {
    "hello": "Hello",
    "story": "Write a long story about a lighthouse keeper.",
    "code": "Write a Python red-black tree with insertion and deletion.",
    "list": "Count from 1 to 500, one number per line.",
    "random": "3 1 4 1 5 9 2 6 5 3 5 8 9 7 9 3 2 3 8 4 6 2 6 4 3 3 8 3 2 7 9 5",
}


class AcceptBenchError(RuntimeError):
    pass


def _complete(url: str, model: str, prompt: str, max_tokens: int, temperature: float, timeout: float) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
                       "ignore_eos": True}).encode("utf-8")
    request = urllib.request.Request(url.rstrip("/") + "/v1/completions", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)["choices"][0]["text"]
    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as exc:
        raise AcceptBenchError(f"completion failed: {exc}") from exc


def run(url: str, model: str = "default", kinds: tuple[str, ...] = tuple(KINDS), max_tokens: int = 512,
        temperature: float = 0.0, timeout: float = 900.0) -> dict:
    unknown = [k for k in kinds if k not in KINDS]
    if unknown or not kinds or max_tokens < 1:
        raise AcceptBenchError(f"kinds must be among {sorted(KINDS)} and max-tokens at least 1 (got {unknown or kinds})")
    if scrape_metrics(url) is None:
        raise AcceptBenchError("the server exposes no /metrics: acceptance cannot be read")
    _complete(url, model, "Hi", 8, temperature, timeout)          # warm-up: first request pays one-time costs
    results = {}
    for kind in kinds:
        before = scrape_metrics(url)
        text = _complete(url, model, KINDS[kind], max_tokens, temperature, timeout)
        delta = engine_delta(before, scrape_metrics(url), [])
        if not delta or "draft_acceptance" not in delta:
            raise AcceptBenchError(f"{kind}: the engine drafted nothing, so there is no acceptance to report")
        results[kind] = {
            "drafted": int(delta["draft_tokens_total"]), "accepted": int(delta["draft_accepted_total"]),
            "acceptance": delta["draft_acceptance"],
            "decode_tokens_per_step": delta.get("decode_tokens_per_step"),
            "decode_ms_per_step": delta.get("decode_ms_per_step"),
            "prompt_sha256": hashlib.sha256(KINDS[kind].encode("utf-8")).hexdigest(),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_head": text[:60],
        }
    return {"schema": SCHEMA,
            "config": {"url": url, "model": model, "kinds": list(kinds), "max_tokens": max_tokens,
                       "temperature": temperature, "ignore_eos": True, "endpoint": "/v1/completions"},
            "kinds": results}


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, add_help=add_help)
    ap.add_argument("--url", required=True, help="server base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--model", default="default")
    ap.add_argument("--kinds", default=",".join(KINDS), help=f"comma-separated, from {','.join(KINDS)}")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--output", help="write the JSON record here instead of stdout")
    return ap


def run_from_namespace(args) -> int:
    try:
        record = run(args.url, args.model, tuple(k for k in args.kinds.split(",") if k), args.max_tokens,
                     args.temperature, args.timeout)
    except AcceptBenchError as exc:
        print(f"accept-bench: ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(record, indent=1, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    for kind, r in record["kinds"].items():
        print(f"accept-bench: {kind:7} drafted {r['drafted']:5} accepted {r['accepted']:5} ({r['acceptance'] * 100:3.0f}%)  "
              f"{r['decode_tokens_per_step']} tok/step  {r['decode_ms_per_step']} ms/step", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    return run_from_namespace(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
