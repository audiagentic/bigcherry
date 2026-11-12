"""Small deterministic 27B HTTP workload runner for an already running lab server.

Only loopback ports 18400-18419 are accepted, preventing accidental traffic
to the default production port. This is not an inference service launcher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen


def prompt_for(index: int, repetitions: int) -> str:
    if isinstance(repetitions, bool) or repetitions < 1 or repetitions > 10000:
        raise ValueError("repetitions must be 1..10000")
    return (
        ("Explain what determines memory bandwidth requirements for inference. "
         "Include the roles of weights, activations and the KV cache. ") * repetitions
        + f"\nRespond with one short paragraph about scenario {index}."
    )


def _one(port: int, prompt: str, max_tokens: int, timeout: int) -> dict:
    payload = {"model": "default", "prompt": prompt, "temperature": 0,
               "max_tokens": max_tokens}
    request = Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    before = time.monotonic()
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP status {response.status}")
        obj = json.load(response)
    elapsed = time.monotonic() - before
    text = obj["choices"][0]["text"]
    usage = obj["usage"]
    if not isinstance(text, str) or not isinstance(usage["completion_tokens"], int):
        raise TypeError("unexpected server response shape")
    return {
        "latency_s": round(elapsed, 5),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "finish_reason": obj["choices"][0].get("finish_reason"),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def run(*, port: int, prompt_repeats: int, max_tokens: int,
        concurrency: int, rounds: int, timeout: int) -> dict:
    if not (18400 <= port <= 18419):
        raise ValueError("port must be an allocated 18400..18419 lab port")
    if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0
               for x in (max_tokens, concurrency, rounds, timeout)):
        raise ValueError("positive integer generation/round/timeout settings required")
    if concurrency > 8 or rounds > 20 or max_tokens > 512:
        raise ValueError("this small harness is limited to 8 sessions, 20 rounds, 512 tokens")
    # Warm once with a DISTINCT prompt to avoid counting cold graph capture as steady state.
    warmup = _one(port, prompt_for(900000, prompt_repeats), min(16, max_tokens), timeout)
    rows = []
    round_elapsed = []
    for cycle in range(rounds):
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            before = time.monotonic()
            tasks = [
                pool.submit(_one, port, prompt_for(cycle * concurrency + i, prompt_repeats),
                            max_tokens, timeout)
                for i in range(concurrency)
            ]
            round_rows = [f.result() for f in tasks]
            elapsed = time.monotonic() - before
            round_elapsed.append(elapsed)
            rows.extend(round_rows)
    total = sum(row["completion_tokens"] for row in rows)
    elapsed = sum(round_elapsed)
    latencies = sorted(row["latency_s"] for row in rows)
    p95_index = min(len(latencies) - 1, max(0, int(len(latencies) * 0.95 + 0.999999) - 1))
    return {
        "recorded_utc": datetime.now(UTC).isoformat(),
        "port": port,
        "workload": {"prompt_repeats": prompt_repeats, "max_tokens": max_tokens,
                     "concurrency": concurrency, "rounds": rounds, "temperature": 0},
        "warmup": warmup,
        "requests": rows,
        "round_seconds": [round(v, 5) for v in round_elapsed],
        "completion_tokens": total,
        "total_round_seconds": round(elapsed, 5),
        "aggregate_generated_tokens_per_s": round(total / elapsed, 3),
        "p50_request_s": round(statistics.median(latencies), 5),
        "p95_request_s": round(latencies[p95_index], 5),
        "warnings": [
            "HTTP end-to-end time includes scheduling, prompt and generation.",
            "Text hashes are diagnostics, not a quality or KL metric.",
            "The default prompt is deterministic but does not control token-level graph capture.",
            "A single arm is not an A/B performance qualification.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prompt-repeats", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(port=args.port, prompt_repeats=args.prompt_repeats,
                 max_tokens=args.max_tokens, concurrency=args.concurrency,
                 rounds=args.rounds, timeout=args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "completion_tokens", "total_round_seconds", "aggregate_generated_tokens_per_s",
        "p50_request_s", "p95_request_s"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
