"""Concurrent-stream benchmark against an already-running OpenAI-compatible server (RAD10-01).

``completion_bench`` (server_completion.py) is deliberately single-slot: one request at a time so every counter is
attributable. Serving is not single-stream, so this module measures the other regime: N simultaneous streaming chat
completions, each with its own prompt, and reports what a client sees:

  per stream   time to first token, tokens, decode tokens/s, inter-token gaps
  per round    aggregate tokens/s over the round's wall time, per-stream mean, TTFT and inter-token percentiles
  identity     a hash of every stream's text, so a run with helpers on can be compared with helpers off at temperature 0

It connects to a server the caller launched and configured (lifecycle belongs to the campaign/build code), uses only the
standard library, and writes one JSON record (``bigcherry.stream-benchmark.v1``). A stream that fails is recorded with its
error and makes the run exit non-zero; nothing is retried, and a round with a failed stream reports no aggregate.

  bigcherry stream-bench --url http://127.0.0.1:8080 --streams 1,2,4,8 --prompt-tokens 512 --max-tokens 256 --reps 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "bigcherry.stream-benchmark.v1"

_WORDS = ("amber basalt cedar delta ember fjord garnet harbor indigo juniper kestrel lantern meadow nectar onyx prairie "
          "quartz ridge summit thistle umber valley willow xenon yarrow zephyr").split()


class StreamBenchError(RuntimeError):
    pass


@dataclass
class StreamResult:
    stream: int
    ok: bool = False
    error: str = ""
    t_start: float = 0.0
    t_first: float = 0.0
    t_end: float = 0.0
    tokens: int = 0
    prompt_tokens: int = 0
    text_sha256: str = ""
    gaps_ms: list = field(default_factory=list)

    def ttft_ms(self) -> float:
        return (self.t_first - self.t_start) * 1000.0 if self.ok else float("nan")

    def decode_tps(self) -> float:
        span = self.t_end - self.t_first
        return (self.tokens - 1) / span if self.ok and self.tokens > 1 and span > 0 else float("nan")


def make_prompt(stream: int, prompt_tokens: int, seed: int, shared_prefix: bool) -> str:
    """A deterministic prompt of about `prompt_tokens` tokens (one short word is about one token). Streams differ unless
    `shared_prefix`, so the server cannot answer one from another's cache by accident."""
    rng = random.Random(seed if shared_prefix else seed * 1000003 + stream)
    words = [rng.choice(_WORDS) for _ in range(max(1, prompt_tokens))]
    return "Continue this list of words with a short story.\n" + " ".join(words)


def one_stream(url: str, model: str, stream: int, prompt: str, max_tokens: int, temperature: float, timeout: float,
               seed: int) -> StreamResult:
    res = StreamResult(stream)
    body = json.dumps({
        "model": model, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
        "temperature": temperature, "seed": seed, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    text, chunks, last = [], 0, None
    res.t_start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    res.prompt_tokens = int(event["usage"].get("prompt_tokens", 0))
                    res.tokens = int(event["usage"].get("completion_tokens", 0)) or res.tokens
                for choice in event.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        now = time.perf_counter()
                        if not chunks:
                            res.t_first = now
                        else:
                            res.gaps_ms.append((now - last) * 1000.0)
                        last = now
                        chunks += 1
                        text.append(piece)
        res.t_end = time.perf_counter()
        if not chunks:
            raise StreamBenchError("the stream produced no content")
        if not res.tokens:
            res.tokens = chunks                      # no usage block: count content chunks (one token each)
        res.text_sha256 = hashlib.sha256("".join(text).encode()).hexdigest()
        res.ok = True
    except (urllib.error.URLError, OSError, ValueError, StreamBenchError) as exc:
        res.t_end = time.perf_counter()
        res.error = f"{type(exc).__name__}: {exc}"
    return res


def pct(values, p):
    v = sorted(x for x in values if x == x)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def run_round(url: str, model: str, streams: int, prompt_tokens: int, max_tokens: int, temperature: float,
              timeout: float, seed: int, shared_prefix: bool) -> dict:
    """N simultaneous streams released together; returns the round's record."""
    results: list = [None] * streams
    gate = threading.Barrier(streams)

    def work(i):
        gate.wait()
        results[i] = one_stream(url, model, i, make_prompt(i, prompt_tokens, seed, shared_prefix), max_tokens,
                                temperature, timeout, seed)

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = all(r.ok for r in results)
    wall = max(r.t_end for r in results) - min(r.t_start for r in results)
    record = {"streams": streams, "ok": ok, "wall_s": round(wall, 4),
              "per_stream": [{"stream": r.stream, "ok": r.ok, "error": r.error, "tokens": r.tokens,
                              "prompt_tokens": r.prompt_tokens, "ttft_ms": round(r.ttft_ms(), 2),
                              "decode_tps": round(r.decode_tps(), 3), "text_sha256": r.text_sha256} for r in results]}
    if ok:
        total = sum(r.tokens for r in results)
        gaps = [g for r in results for g in r.gaps_ms]
        record["aggregate"] = {
            "tokens": total, "aggregate_tps": round(total / wall, 3),
            "mean_stream_decode_tps": round(statistics.fmean(r.decode_tps() for r in results if r.tokens > 1), 3)
            if any(r.tokens > 1 for r in results) else None,
            "ttft_ms_p50": round(pct([r.ttft_ms() for r in results], 0.50), 2),
            "ttft_ms_p95": round(pct([r.ttft_ms() for r in results], 0.95), 2),
            "itl_ms_p50": round(pct(gaps, 0.50), 3), "itl_ms_p95": round(pct(gaps, 0.95), 3),
            "itl_ms_p99": round(pct(gaps, 0.99), 3)}
    return record


def run(url: str, model: str, stream_counts: list, prompt_tokens: int, max_tokens: int, reps: int,
        temperature: float = 0.0, timeout: float = 600.0, seed: int = 1, shared_prefix: bool = False) -> dict:
    if not stream_counts or any(n < 1 for n in stream_counts) or reps < 1:
        raise StreamBenchError("stream counts and reps must be at least 1")
    rounds = []
    for n in stream_counts:
        for rep in range(reps):
            rec = run_round(url, model, n, prompt_tokens, max_tokens, temperature, timeout, seed, shared_prefix)
            rec["rep"] = rep
            rounds.append(rec)
    summary = {}
    for n in stream_counts:
        good = [r["aggregate"] for r in rounds if r["streams"] == n and r["ok"]]
        if good:
            summary[str(n)] = {"reps_ok": len(good),
                               "aggregate_tps_median": statistics.median(g["aggregate_tps"] for g in good),
                               "ttft_ms_p95_median": statistics.median(g["ttft_ms_p95"] for g in good),
                               "itl_ms_p95_median": statistics.median(g["itl_ms_p95"] for g in good)}
    return {"schema": SCHEMA,
            "config": {"url": url, "model": model, "streams": stream_counts, "prompt_tokens": prompt_tokens,
                       "max_tokens": max_tokens, "reps": reps, "temperature": temperature, "seed": seed,
                       "shared_prefix": shared_prefix},
            "ok": all(r["ok"] for r in rounds), "rounds": rounds, "summary": summary}


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, add_help=add_help)
    ap.add_argument("--url", required=True, help="server base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--model", default="default")
    ap.add_argument("--streams", default="1,2,4,8", help="comma-separated concurrent stream counts")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shared-prefix", action="store_true", help="give every stream the same prompt (prefix-cache probe)")
    ap.add_argument("--output", help="write the JSON record here (default: stdout)")
    return ap


def run_from_namespace(args) -> int:
    try:
        counts = [int(x) for x in str(args.streams).split(",") if x.strip()]
        record = run(args.url, args.model, counts, args.prompt_tokens, args.max_tokens, args.reps, args.temperature,
                     args.timeout, args.seed, args.shared_prefix)
    except (StreamBenchError, ValueError) as exc:
        print(f"stream-bench: ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(record, indent=1, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    for n, s in record["summary"].items():
        print(f"stream-bench: {n:>2} streams  {s['aggregate_tps_median']:9.2f} tok/s aggregate  "
              f"ttft p95 {s['ttft_ms_p95_median']:8.1f} ms  itl p95 {s['itl_ms_p95_median']:7.2f} ms", file=sys.stderr)
    if not record["ok"]:
        print("stream-bench: FAIL: at least one stream failed (see per_stream errors)", file=sys.stderr)
        return 1
    return 0


def main(argv: list | None = None) -> int:
    return run_from_namespace(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
