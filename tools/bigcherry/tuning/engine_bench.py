"""MEN03: one prefill/decode measurement for any declared engine, with one result record.

The engine is launched through its declared serve specification (engines/<name>/engine.toml, [serve]) by
ServerRunner, asked for completions over the OpenAI-compatible route every engine here serves, and stopped the way it
declares. The record written is the same for every engine, so two engines, or two builds of one, are compared from
files of one shape.

For each prompt depth and repeat: one uncached streamed greedy request (a unique nonce leads the prompt, so no
prefix cache applies).
  prefill t/s = prompt tokens / time to first token
  decode t/s  = (completion tokens - 1) / (time of last token - time of first token)
The prompt is the lab corpus cut to the requested size, as long-ctx-profile.sh cuts it.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..core import engines, paths
from .server_runner import ServerError, ServerRunner

SCHEMA = 1
RESULT = "engine-bench.json"
CHARS_PER_TOKEN = 3.6  # corpus average for the Qwen tokenizer; the server reports the real prompt token count
ASK = "\n\nSummarise the above in detail:"


@dataclass(frozen=True)
class RequestResult:
    depth: int
    rep: int
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float | None
    prefill_tps: float | None
    decode_tps: float | None
    text_head: str
    error: str | None = None


def _rate(numerator: float, seconds: float | None) -> float | None:
    if seconds is None or seconds <= 0 or not math.isfinite(seconds):
        return None
    return numerator / seconds


def summarise(*, depth: int, rep: int, started: float, first: float | None, last: float | None,
              usage: dict, text: str) -> RequestResult:
    """One request's timings as the normalised row. Times are seconds on one clock."""
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    ttft = (first - started) if first is not None else None
    decode_span = (last - first) if first is not None and last is not None else None
    return RequestResult(
        depth=depth, rep=rep, prompt_tokens=prompt, completion_tokens=completion, ttft_s=ttft,
        prefill_tps=_rate(prompt, ttft),
        decode_tps=_rate(completion - 1, decode_span) if completion > 1 else None,
        text_head=text[:160],
    )


def stream_completion(base_url: str, body: dict, *, timeout_s: int = 7200,
                      clock: Callable[[], float] = time.time) -> tuple[float, float | None, float | None, dict, str]:
    """POST a streamed /v1/completions request; (started, first token time, last token time, usage, text)."""
    request = urllib.request.Request(
        base_url + "/v1/completions", json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"},
    )
    started = clock()
    first = last = None
    usage: dict = {}
    text: list[str] = []
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw in response:
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
                now = clock()
                first = first if first is not None else now
                last = now
                text.append(choices[0]["text"])
    return started, first, last, usage, "".join(text)


def served_model_id(base_url: str, *, timeout_s: int = 60) -> str:
    with urllib.request.urlopen(base_url + "/v1/models", timeout=timeout_s) as response:
        return str(json.loads(response.read().decode("utf-8"))["data"][0]["id"])


def prompt_for(corpus: str, depth: int, nonce: str) -> str:
    return f"[{nonce}]\n" + corpus[: int(depth * CHARS_PER_TOKEN)] + ASK


def measure(base_url: str, model_id: str, corpus: str, depths: Iterable[int], *, reps: int, decode: int,
            report: Callable[[str], None] = print) -> list[RequestResult]:
    rows: list[RequestResult] = []
    for depth in depths:
        for rep in range(reps):
            body = {
                "model": model_id, "prompt": prompt_for(corpus, depth, uuid.uuid4().hex), "max_tokens": decode,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            }
            try:
                started, first, last, usage, text = stream_completion(base_url, body)
            except (urllib.error.URLError, OSError, TimeoutError, ValueError, KeyError) as exc:
                # a failed depth is recorded and the larger repeats of it are skipped; smaller depths stand
                rows.append(RequestResult(depth, rep, 0, 0, None, None, None, "", error=str(exc)))
                report(f"d{depth} r{rep}: REQUEST_FAILED {exc}")
                break
            row = summarise(depth=depth, rep=rep, started=started, first=first, last=last, usage=usage, text=text)
            rows.append(row)
            report(
                f"d{depth} r{rep}: prompt {row.prompt_tokens} tok, "
                f"ttft {_show(row.ttft_s, 2)} s = {_show(row.prefill_tps, 1)} t/s prefill, "
                f"decode {row.completion_tokens} tok at {_show(row.decode_tps, 1)} t/s"
            )
    return rows


def _show(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def run(*, engine: str, binary: Path, model: str, out_dir: Path, depths: Sequence[int], corpus_path: Path,
        reps: int = 2, decode: int = 512, extra_args: Sequence[str] = (), env: dict[str, str] | None = None,
        env_unset: Sequence[str] = (), label: str = "run", health_timeout_s: int = 2400, port: int | None = None,
        report: Callable[[str], None] = print) -> dict:
    """Launch ``engine``'s server from ``binary``, measure ``depths``, stop it, and write the result record."""
    serve = engines.load(paths.REPO_ROOT, engine).serve
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{label}.server.log"
    runner = ServerRunner(
        serve=serve, binary=Path(binary), model=Path(model) if Path(model).exists() else model, port=port,
        extra_args=tuple(extra_args), env_overrides=dict(env or {}), env_unset=tuple(env_unset), log_path=log_path,
    )
    record: dict = {
        "schema": SCHEMA, "engine": engine, "label": label, "binary": str(binary), "model": str(model),
        "extra_args": list(extra_args), "env": dict(env or {}), "depths": list(depths), "reps": reps,
        "decode": decode, "rows": [], "draft": None, "server_error": None, "shutdown": None,
    }
    runner.launch()
    try:
        runner.wait_healthy(timeout_s=health_timeout_s)
        base_url = f"http://{runner.host}:{runner.port}"
        record["served_model"] = served_model_id(base_url)
        corpus = Path(corpus_path).read_text(encoding="utf-8", errors="replace")
        rows = measure(base_url, record["served_model"], corpus, depths, reps=reps, decode=decode, report=report)
        record["rows"] = [asdict(row) for row in rows]
        metrics = serve.draft_stats is not None and serve.draft_stats.source == "metrics"
        if metrics:  # counters are only reachable while the server is up
            record["draft"] = _draft(runner)
    except (ServerError, urllib.error.URLError, OSError) as exc:
        record["server_error"] = str(exc)
        report(f"SERVER_FAILED {exc}")
    finally:
        stopped = runner.shutdown()
        if stopped is not None:
            record["shutdown"] = {"method": stopped.method, "clean": stopped.clean, "forced": stopped.forced,
                                  "returncode": stopped.returncode}
    if record["draft"] is None and record["server_error"] is None:
        record["draft"] = _draft(runner)  # log-reported totals are complete once the server has exited
    (out_dir / f"{label}.{RESULT}").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    if record["draft"]:
        drafted, accepted = record["draft"]["drafted"], record["draft"]["accepted"]
        report(f"draft: accepted {accepted} of {drafted} ({100.0 * accepted / drafted:.1f}%)" if drafted
               else "draft: none drafted")
    return record


def _draft(runner: ServerRunner) -> dict | None:
    try:
        stats = runner.draft_stats()
    except ServerError:
        return None
    return None if stats is None else {"drafted": stats[0], "accepted": stats[1]}


def cmd_engine_bench(args) -> int:
    env: dict[str, str] = {}
    for item in args.env or ():
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise SystemExit(f"--env takes NAME=VALUE, got {item!r}")
        env[name] = value
    record = run(
        engine=args.engine, binary=Path(args.binary), model=args.model, out_dir=Path(args.out),
        depths=[int(d) for d in args.depth], corpus_path=Path(args.corpus), reps=args.reps, decode=args.decode,
        extra_args=tuple(args.server_args or ()), env=env, env_unset=tuple(args.unset or ()), label=args.label,
        health_timeout_s=args.health_timeout, port=args.port,
    )
    failed = record["server_error"] is not None or any(row.get("error") for row in record["rows"])
    return 1 if failed else 0
