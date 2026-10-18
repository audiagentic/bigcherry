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
    t_last: float = 0.0              # last text event
    t_end: float = 0.0               # response end (the stream closed)
    tokens: int = 0                  # completion tokens: server usage, or an event-count estimate if allowed
    usage_source: str = ""           # "server" | "event_count_estimate"
    finish_reason: str = ""
    events: int = 0                  # SSE events that carried text (one event may carry several tokens)
    prompt_tokens: int = 0
    text_sha256: str = ""
    gaps_ms: list = field(default_factory=list)

    def ttft_ms(self) -> float:
        return (self.t_first - self.t_start) * 1000.0 if self.ok else float("nan")

    def decode_tps_est(self) -> float:
        """Client-observed decode rate: (completion tokens - 1) over first-to-last text event. An estimate, because the
        stream's timestamps do not identify when each token was produced (one event can carry several)."""
        span = self.t_last - self.t_first
        return (self.tokens - 1) / span if self.ok and self.tokens > 1 and span > 0 else float("nan")


def make_prompt(stream: int, prompt_tokens: int, seed: int, shared_prefix: bool) -> str:
    """A deterministic prompt of about `prompt_tokens` tokens (one short word is about one token). Streams differ unless
    `shared_prefix`, so the server cannot answer one from another's cache by accident."""
    rng = random.Random(seed if shared_prefix else seed * 1000003 + stream)
    words = [rng.choice(_WORDS) for _ in range(max(1, prompt_tokens))]
    return "Continue this list of words with a short story.\n" + " ".join(words)


def one_stream(url: str, model: str, stream: int, prompt: str, max_tokens: int, temperature: float, timeout: float,
               seed: int, allow_estimate: bool = False) -> StreamResult:
    res = StreamResult(stream)
    body = json.dumps({
        "model": model, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
        "temperature": temperature, "seed": seed, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    text, chunks, last, raw_head, done = [], 0, None, [], False
    usage_tokens = 0
    res.t_start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("usage"):
                    res.prompt_tokens = int(event["usage"].get("prompt_tokens", 0))
                    usage_tokens = int(event["usage"].get("completion_tokens", 0)) or usage_tokens
                if len(raw_head) < 4:
                    raw_head.append(data[:300])
                for choice in event.get("choices", []):
                    if choice.get("finish_reason"):
                        res.finish_reason = str(choice["finish_reason"])
                    delta = choice.get("delta") or {}
                    # a thinking model streams its tokens as reasoning_content / reasoning before any content
                    piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                    if piece:
                        now = time.perf_counter()
                        if not chunks:
                            res.t_first = now
                        else:
                            res.gaps_ms.append((now - last) * 1000.0)
                        last = res.t_last = now
                        chunks += 1
                        text.append(piece)
        res.t_end = time.perf_counter()
        if not chunks:
            raise StreamBenchError("the stream produced no content; first events: " + " | ".join(raw_head))
        if not done and not res.finish_reason:
            raise StreamBenchError("the stream ended without [DONE] or a finish reason (truncated)")
        res.events = chunks
        if usage_tokens:
            res.tokens, res.usage_source = usage_tokens, "server"
        elif allow_estimate:
            res.tokens, res.usage_source = chunks, "event_count_estimate"      # one token an event: an estimate only
        else:
            raise StreamBenchError("no usage block (the server must honour stream_options.include_usage); "
                                   "token counts are never inferred from event counts unless --allow-estimate")
        res.text_sha256 = hashlib.sha256("".join(text).encode()).hexdigest()
        res.ok = True
    except (urllib.error.URLError, OSError, ValueError, StreamBenchError) as exc:
        res.t_end = time.perf_counter()
        res.error = f"{type(exc).__name__}: {exc}"
    return res


_COUNTERS = ("radiance:engine_steps_decode_total", "radiance:engine_step_seconds_decode_total",
             "radiance:decode_tokens_total", "radiance:draft_tokens_total", "radiance:draft_accepted_total",
             "radiance:engine_steps_prefill_total", "radiance:prefill_tokens_total", "vllm:num_preemptions_total")
_GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


def scrape_metrics(url: str, timeout: float = 5.0):
    """The server's Prometheus text as {name: value}, label sets summed; None when the endpoint is not there."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    out: dict = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition("{")
        if not rest:
            name, _, value = line.partition(" ")
        else:
            value = rest.rpartition("}")[2]
        try:
            out[name.strip()] = out.get(name.strip(), 0.0) + float(value.split()[0])
        except (ValueError, IndexError):
            continue
    return out


def engine_delta(before, after, running):
    """What the engine did during one round, from two /metrics readings and the sampled gauges. None when the engine
    does not expose them. Steps are decode steps; acceptance is accepted/drafted tokens over the round."""
    if not before or not after:
        return None
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in _COUNTERS if k in after}
    steps = d.get("radiance:engine_steps_decode_total", 0.0)
    out = {k.split(":", 1)[1]: round(v, 6) for k, v in d.items()}
    if steps > 0:
        out["decode_ms_per_step"] = round(d.get("radiance:engine_step_seconds_decode_total", 0.0) / steps * 1000.0, 4)
        out["decode_tokens_per_step"] = round(d.get("radiance:decode_tokens_total", 0.0) / steps, 4)
        seconds = d.get("radiance:engine_step_seconds_decode_total", 0.0)
        if seconds > 0:
            # throughput of the decode phase alone: the client-side aggregate also carries every prefill and the ramp
            out["decode_tps_engine"] = round(d.get("radiance:decode_tokens_total", 0.0) / seconds, 3)
    drafted = d.get("radiance:draft_tokens_total", 0.0)
    if drafted > 0:
        out["draft_acceptance"] = round(d.get("radiance:draft_accepted_total", 0.0) / drafted, 4)
    if running:
        out["requests_running_mean"] = round(statistics.fmean(r for r, _ in running), 3)
        out["requests_running_max"] = max(r for r, _ in running)
        out["requests_waiting_max"] = max(w for _, w in running)
    return out


def pct(values, p):
    v = sorted(x for x in values if x == x)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def run_round(url: str, model: str, streams: int, prompt_tokens: int, max_tokens: int, temperature: float,
              timeout: float, seed: int, shared_prefix: bool, allow_estimate: bool = False) -> dict:
    """N simultaneous streams released together by a controller; returns the round's record.

    Aggregate throughput here is finite-batch makespan throughput: all tokens over the time from the first request start to
    the last response end. The last straggler can dominate that denominator, so it is not sustained serving throughput."""
    results: list = [None] * streams
    gate = threading.Barrier(streams + 1)
    before = scrape_metrics(url)
    running: list = []
    stop = threading.Event()

    def sample():
        while True:
            m = scrape_metrics(url, timeout=2.0)
            if m and "vllm:num_requests_running" in m:
                running.append((m["vllm:num_requests_running"], m.get("vllm:num_requests_waiting", 0.0)))
            if stop.wait(0.25):
                break

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    def work(i):
        gate.wait()
        results[i] = one_stream(url, model, i, make_prompt(i, prompt_tokens, seed, shared_prefix), max_tokens,
                                temperature, timeout, seed, allow_estimate)

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(streams)]
    for t in threads:
        t.start()
    release = time.perf_counter()
    wall_start_ns = time.time_ns()                 # epoch clock, so helper TRACE lines can be bucketed by round
    gate.wait()
    time.sleep(0.03)                       # let the streams get admitted, then read the gauges once more
    mid = scrape_metrics(url, timeout=2.0)
    if mid and "vllm:num_requests_running" in mid:
        running.append((mid["vllm:num_requests_running"], mid.get("vllm:num_requests_waiting", 0.0)))
    for t in threads:
        t.join()
    stop.set()
    sampler.join()
    after = scrape_metrics(url)
    wall_end_ns = time.time_ns()
    ok = all(r.ok for r in results)
    wall = max(r.t_end for r in results) - min(r.t_start for r in results)
    record = {"streams": streams, "ok": ok, "wall_s": round(wall, 4),
              "wall_start_ns": wall_start_ns, "wall_end_ns": wall_end_ns,
              "start_skew_ms": round((max(r.t_start for r in results) - min(r.t_start for r in results)) * 1000.0, 3),
              "release_to_first_start_ms": round((min(r.t_start for r in results) - release) * 1000.0, 3),
              "per_stream": [{"stream": r.stream, "ok": r.ok, "error": r.error, "tokens": r.tokens,
                              "usage_source": r.usage_source, "finish_reason": r.finish_reason, "events": r.events,
                              "prompt_tokens": r.prompt_tokens, "ttft_ms": round(r.ttft_ms(), 2),
                              "decode_tps_est": round(r.decode_tps_est(), 3),
                              "response_tail_ms": round((r.t_end - r.t_last) * 1000.0, 2) if r.ok else None,
                              "text_sha256": r.text_sha256} for r in results]}
    record["engine"] = engine_delta(before, after, running)
    if ok:
        total = sum(r.tokens for r in results)
        gaps = [g for r in results for g in r.gaps_ms]
        per = [r.decode_tps_est() for r in results if r.tokens > 1]
        record["aggregate"] = {
            "tokens": total, "aggregate_tps": round(total / wall, 3),
            "mean_stream_decode_tps_est": round(statistics.fmean(per), 3) if per else None,
            "ttft_ms_p50": round(pct([r.ttft_ms() for r in results], 0.50), 2),
            "ttft_ms_p95": round(pct([r.ttft_ms() for r in results], 0.95), 2),
            "ttft_ms_max": round(max(r.ttft_ms() for r in results), 2),
            "event_gap_ms_p50": round(pct(gaps, 0.50), 3), "event_gap_ms_p95": round(pct(gaps, 0.95), 3),
            "event_gap_ms_p99": round(pct(gaps, 0.99), 3), "event_gap_ms_max": round(max(gaps), 3) if gaps else None,
            "estimated_tokens": any(r.usage_source != "server" for r in results)}
    return record


def run(url: str, model: str, stream_counts: list, prompt_tokens: int, max_tokens: int, reps: int,
        temperature: float = 0.0, timeout: float = 600.0, seed: int = 1, shared_prefix: bool = False,
        allow_estimate: bool = False) -> dict:
    if not stream_counts or any(n < 1 for n in stream_counts) or reps < 1:
        raise StreamBenchError("stream counts and reps must be at least 1")
    rounds = []
    for n in stream_counts:
        for rep in range(reps):
            rec = run_round(url, model, n, prompt_tokens, max_tokens, temperature, timeout, seed, shared_prefix,
                            allow_estimate)
            rec["rep"] = rep
            rounds.append(rec)
    summary = {}
    for n in stream_counts:
        good = [r["aggregate"] for r in rounds if r["streams"] == n and r["ok"]]
        if good:
            summary[str(n)] = {"reps_ok": len(good),
                               "aggregate_tps_median": statistics.median(g["aggregate_tps"] for g in good),
                               "ttft_ms_max_median": statistics.median(g["ttft_ms_max"] for g in good),
                               "event_gap_ms_p95_median": statistics.median(g["event_gap_ms_p95"] for g in good)}
            eng = [r["engine"] for r in rounds if r["streams"] == n and r["ok"] and r.get("engine")]
            for key in ("decode_ms_per_step", "decode_tokens_per_step", "decode_tps_engine", "draft_acceptance", "requests_running_mean"):
                vals = [e[key] for e in eng if key in e]
                if vals:
                    summary[str(n)][key + "_median"] = statistics.median(vals)
    return {"schema": SCHEMA,
            "config": {"url": url, "model": model, "streams": stream_counts, "prompt_tokens": prompt_tokens,
                       "max_tokens": max_tokens, "reps": reps, "temperature": temperature, "seed": seed,
                       "shared_prefix": shared_prefix, "allow_estimate": allow_estimate,
                       "throughput_definition": "finite-batch makespan: tokens / (first request start to last response end)"},
            "ok": all(r["ok"] for r in rounds), "rounds": rounds, "summary": summary}


def flat_metrics(record: dict) -> dict:
    """The record as flat named numbers, so a stream cell can feed the A/B harness like any other measurement:
    sN_agg_tps (finite-batch throughput), sN_stream_tps (mean client-observed per-stream decode estimate),
    sN_ttft_max_ms and sN_gap_p95_ms for each stream count N (medians over the repetitions). With 2 to 6 streams a p95 is
    just the slowest stream, so the TTFT figure is the explicit maximum. A run with any failed stream has no metrics."""
    if not record.get("ok"):
        raise StreamBenchError("a stream failed; no metrics")
    out = {}
    for n in record["config"]["streams"]:
        good = [r["aggregate"] for r in record["rounds"] if r["streams"] == n and r["ok"]]
        out[f"s{n}_agg_tps"] = statistics.median(g["aggregate_tps"] for g in good)
        out[f"s{n}_ttft_max_ms"] = statistics.median(g["ttft_ms_max"] for g in good)
        out[f"s{n}_gap_p95_ms"] = statistics.median(g["event_gap_ms_p95"] for g in good)
        per = [g["mean_stream_decode_tps_est"] for g in good if g["mean_stream_decode_tps_est"] is not None]
        if per:
            out[f"s{n}_stream_tps"] = statistics.median(per)
        eng = [r["engine"] for r in record["rounds"] if r["streams"] == n and r["ok"] and r.get("engine")]
        for key in ("decode_ms_per_step", "decode_tokens_per_step", "decode_tps_engine", "draft_acceptance", "requests_running_mean"):
            vals = [e[key] for e in eng if key in e]
            if vals:
                out[f"s{n}_{key}"] = statistics.median(vals)
    return out


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
    ap.add_argument("--allow-estimate", action="store_true",
                    help="count one token per text event when the server sends no usage block (marked as an estimate)")
    ap.add_argument("--output", help="write the JSON record here (default: stdout)")
    return ap


def run_from_namespace(args) -> int:
    try:
        counts = [int(x) for x in str(args.streams).split(",") if x.strip()]
        record = run(args.url, args.model, counts, args.prompt_tokens, args.max_tokens, args.reps, args.temperature,
                     args.timeout, args.seed, args.shared_prefix, args.allow_estimate)
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
              f"ttft max {s['ttft_ms_max_median']:8.1f} ms  event gap p95 {s['event_gap_ms_p95_median']:7.2f} ms"
              + (f"  step {s['decode_ms_per_step_median']:6.2f} ms  {s['decode_tokens_per_step_median']:5.2f} tok/step"
                 if "decode_ms_per_step_median" in s else ""), file=sys.stderr)
    if not record["ok"]:
        print("stream-bench: FAIL: at least one stream failed (see per_stream errors)", file=sys.stderr)
        return 1
    return 0


def main(argv: list | None = None) -> int:
    return run_from_namespace(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
