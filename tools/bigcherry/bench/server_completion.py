"""Repeated-completion benchmark harness against an already-running
llama-server, for statistically real MTP speculative-decode throughput
and acceptance numbers -- replacing the single-sample "one completion
request" liveness checks this project has relied on so far.

Deliberately narrow (MVP scope, gpt-dev-agent-reviewed design, 2026-08-31):
connects to a server the caller already launched and configured (no
lifecycle/build orchestration here -- that's pin_bump.py/campaign/build.py's
job, not this module's); requires ``--parallel 1`` (exactly one slot, so
every /metrics delta is attributable to exactly one request, never mixed
with concurrent traffic) and ``--metrics`` (the only way to get real
per-draft-position acceptance counts -- ``/completion``'s own response
only exposes the aggregate draft_n/draft_n_accepted, not per-position).

Output is a NEW, separate JSONL schema (``bigcherry.completion-benchmark.v1``)
-- deliberately NOT analysis/report.py's or analysis/impact.py's
measurement-record schema. Those represent kernel-signature candidate
measurements and paired tuning evidence; an HTTP completion run against a
live server has different semantics (session/request shape, wall-clock
HTTP timing, speculative-decode counters) and forcing it into the existing
schema would make both worse, not share anything real.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

SCHEMA = "bigcherry.completion-benchmark.v1"


class BenchmarkError(RuntimeError):
    pass


# ------------------------------------------------------------------ transport


class Transport(Protocol):
    def get_text(self, path: str) -> str: ...
    def get_json(self, path: str) -> Any: ...
    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]: ...


class HttpTransport:
    """Real transport -- urllib only, matching tuning/server_runner.py's
    established pattern in this codebase (no new HTTP dependency)."""

    def __init__(self, base_url: str, *, timeout_s: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def get_text(self, path: str) -> str:
        try:
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=self.timeout_s) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise BenchmarkError(f"GET {path} failed: {exc}") from exc

    def get_json(self, path: str) -> Any:
        return json.loads(self.get_text(path))

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise BenchmarkError(f"POST {path} failed: {exc}") from exc
        return json.loads(raw.decode("utf-8")) if raw else {}


# ------------------------------------------------------------------ corpus


@dataclass(frozen=True)
class CorpusPrompt:
    id: str
    seed: int
    category: str
    prompt: str


def load_corpus(path: Path) -> tuple[list[CorpusPrompt], str]:
    """Load a fixed, committed prompt corpus (one JSON object per line:
    id/seed/category/prompt). Returns (prompts, sha256-of-raw-bytes) --
    the hash is recorded in every session so a corpus edit is visible in
    the output, not silently comparing apples to oranges across runs.
    """
    raw = Path(path).read_bytes()
    corpus_sha256 = hashlib.sha256(raw).hexdigest()
    prompts: list[CorpusPrompt] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"{path}: line {line_number} is malformed JSON: {exc}") from exc
        for field_name in ("id", "seed", "category", "prompt"):
            if field_name not in entry:
                raise BenchmarkError(f"{path}: line {line_number} missing required field {field_name!r}")
        if entry["id"] in seen_ids:
            raise BenchmarkError(f"{path}: duplicate prompt id {entry['id']!r}")
        seen_ids.add(entry["id"])
        prompts.append(CorpusPrompt(
            id=entry["id"], seed=entry["seed"], category=entry["category"], prompt=entry["prompt"],
        ))
    if not prompts:
        raise BenchmarkError(f"{path}: corpus is empty")
    return prompts, corpus_sha256


# ------------------------------------------------------------------ metrics


@dataclass(frozen=True)
class Metrics:
    """One /metrics snapshot, parsed. Only the fields this harness needs
    -- see parse_prometheus_metrics for the exact real counter names
    (verified against vendor/llama.cpp/tools/server/server-task.cpp, not
    guessed)."""

    draft_tokens_total: float
    accepted_tokens_total: float
    drafts_total: float
    accepted_per_position: dict[int, float] = field(default_factory=dict)


_SCALAR_NAMES = {
    "llamacpp:spec_decode_num_draft_tokens_total": "draft_tokens_total",
    "llamacpp:spec_decode_num_accepted_tokens_total": "accepted_tokens_total",
    "llamacpp:spec_decode_num_drafts_total": "drafts_total",
}
_PER_POSITION_PREFIX = "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""


def parse_prometheus_metrics(text: str) -> Metrics:
    """Parse llama-server's Prometheus text exposition format for exactly
    the speculative-decode counters this harness needs. Ignores every
    other line (HELP/TYPE comments, unrelated counters/gauges) rather
    than trying to be a general Prometheus parser.
    """
    scalars: dict[str, float] = {}
    per_position: dict[int, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(_PER_POSITION_PREFIX):
            rest = line[len(_PER_POSITION_PREFIX):]
            try:
                position_str, value_str = rest.split("\"} ", 1)
                per_position[int(position_str)] = float(value_str)
            except (ValueError, IndexError) as exc:
                raise BenchmarkError(f"malformed per-position metric line: {line!r}") from exc
            continue
        for prefix, field_name in _SCALAR_NAMES.items():
            if line.startswith(prefix + " "):
                scalars[field_name] = float(line[len(prefix) + 1:])
                break
    missing = set(_SCALAR_NAMES.values()) - set(scalars)
    if missing:
        raise BenchmarkError(
            f"/metrics response is missing required counter(s): {sorted(missing)} -- "
            "was the server launched with --metrics?"
        )
    return Metrics(
        draft_tokens_total=scalars["draft_tokens_total"],
        accepted_tokens_total=scalars["accepted_tokens_total"],
        drafts_total=scalars["drafts_total"],
        accepted_per_position=per_position,
    )


def fetch_metrics(transport: Transport) -> Metrics:
    return parse_prometheus_metrics(transport.get_text("/metrics"))


def metrics_delta(before: Metrics, after: Metrics) -> dict[str, Any]:
    """Delta between two snapshots -- must be non-negative (a real
    completion request only ever increments these counters); a negative
    delta means the server restarted or counters otherwise reset mid-run,
    and the caller must fail closed rather than record a nonsensical
    negative count."""
    draft_generated = after.draft_tokens_total - before.draft_tokens_total
    draft_accepted = after.accepted_tokens_total - before.accepted_tokens_total
    verification_cycles = after.drafts_total - before.drafts_total
    if draft_generated < 0 or draft_accepted < 0 or verification_cycles < 0:
        raise BenchmarkError(
            "metrics counters decreased between snapshots -- server restart or "
            "counter reset mid-request; refusing to record a negative delta"
        )
    all_positions = sorted(set(before.accepted_per_position) | set(after.accepted_per_position))
    accepted_by_position: dict[int, float] = {}
    for position in all_positions:
        delta = after.accepted_per_position.get(position, 0.0) - before.accepted_per_position.get(position, 0.0)
        if delta < 0:
            raise BenchmarkError(
                f"per-position accepted-tokens counter decreased at position {position} "
                "between snapshots -- server restart or counter reset mid-request"
            )
        accepted_by_position[position] = delta
    return {
        "draft_generated": draft_generated,
        "draft_accepted": draft_accepted,
        "verification_cycles": verification_cycles,
        "accepted_count_by_position": [accepted_by_position[p] for p in all_positions],
    }


# ------------------------------------------------------------------ session


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    top_k: int


@dataclass(frozen=True)
class SessionConfig:
    session_id: str
    corpus_id: str
    corpus_sha256: str
    bigcherry_revision: str
    llama_pin: str
    llama_revision: str
    model_id: str
    server_argv: tuple[str, ...]
    spec_type: str
    spec_n_max: int
    spec_draft_k: str
    spec_draft_v: str
    sampling: SamplingConfig
    n_predict: int
    order_seed: int


def validate_server(transport: Transport) -> None:
    """Fail closed before running anything: exactly 1 slot (--parallel 1,
    required so every metrics delta is attributable to one request) and a
    reachable /metrics endpoint (required for per-position acceptance)."""
    try:
        slots = transport.get_json("/slots")
    except BenchmarkError as exc:
        raise BenchmarkError(
            f"could not reach /slots -- is the server running with slot "
            f"introspection enabled? ({exc})"
        ) from exc
    if not isinstance(slots, list) or len(slots) != 1:
        raise BenchmarkError(
            f"server must be launched with --parallel 1 (found "
            f"{len(slots) if isinstance(slots, list) else '?'} slot(s)) -- "
            "otherwise a metrics delta cannot be attributed to one request"
        )
    fetch_metrics(transport)  # raises BenchmarkError if /metrics is unreachable/incomplete


def _deterministic_order(prompt_ids: list[str], seed: int) -> list[int]:
    indices = list(range(len(prompt_ids)))
    random.Random(seed).shuffle(indices)
    return indices


def run_request(
    transport: Transport, prompt: CorpusPrompt, config: SessionConfig, *, pass_number: int, order_index: int,
) -> dict[str, Any]:
    before = fetch_metrics(transport)
    start = time.monotonic()
    response = transport.post_json("/completion", {
        "prompt": prompt.prompt,
        "n_predict": config.n_predict,
        "temperature": config.sampling.temperature,
        "top_p": config.sampling.top_p,
        "top_k": config.sampling.top_k,
        "seed": prompt.seed,
        "cache_prompt": False,
        "ignore_eos": True,
    })
    wall_s = time.monotonic() - start
    after = fetch_metrics(transport)

    timings = response.get("timings", {})
    tokens_predicted = response.get("tokens_predicted")
    if not isinstance(tokens_predicted, int) or tokens_predicted <= 0:
        raise BenchmarkError(
            f"prompt {prompt.id!r}: /completion returned no usable tokens_predicted "
            f"({tokens_predicted!r}) -- treating as a failed request, not a 0-token sample"
        )
    if tokens_predicted != config.n_predict:
        raise BenchmarkError(
            f"prompt {prompt.id!r}: generated {tokens_predicted} tokens, expected exactly "
            f"{config.n_predict} (ignore_eos=True should prevent early stopping) -- this "
            f"benchmark measures decode machinery, not completion-length variance, so a "
            f"short/long response is a failed run, not a data point"
        )

    delta = metrics_delta(before, after)
    completion_draft_n = response.get("timings", {}).get("draft_n")
    completion_draft_accepted = response.get("timings", {}).get("draft_n_accepted")
    if completion_draft_n is not None and completion_draft_n != delta["draft_generated"]:
        raise BenchmarkError(
            f"prompt {prompt.id!r}: /completion draft_n={completion_draft_n} != "
            f"/metrics delta draft_generated={delta['draft_generated']} -- concurrent "
            f"traffic, a server restart, or a schema drift between the two; refusing "
            f"to record disagreeing evidence"
        )
    if completion_draft_accepted is not None and completion_draft_accepted != delta["draft_accepted"]:
        raise BenchmarkError(
            f"prompt {prompt.id!r}: /completion draft_n_accepted={completion_draft_accepted} != "
            f"/metrics delta draft_accepted={delta['draft_accepted']} -- concurrent "
            f"traffic, a server restart, or a schema drift between the two; refusing "
            f"to record disagreeing evidence"
        )

    verification_cycles = delta["verification_cycles"]
    mean_accepted_length = (
        1.0 + delta["draft_accepted"] / verification_cycles if verification_cycles else None
    )
    draft_acceptance = (
        delta["draft_accepted"] / delta["draft_generated"] if delta["draft_generated"] else None
    )
    total_by_position = sum(delta["accepted_count_by_position"]) or None
    acceptance_rate_by_position = (
        [count / total_by_position for count in delta["accepted_count_by_position"]]
        if total_by_position
        else None
    )

    predicted_ms = timings.get("predicted_ms", wall_s * 1000.0)
    predicted_tps = timings.get("predicted_per_second", (
        1000.0 * tokens_predicted / predicted_ms if predicted_ms else None
    ))

    return {
        "schema": SCHEMA,
        "kind": "request",
        "session_id": config.session_id,
        "pass": pass_number,
        "order_index": order_index,
        "prompt_id": prompt.id,
        "prompt_sha256": hashlib.sha256(prompt.prompt.encode("utf-8")).hexdigest(),
        "seed": prompt.seed,
        "tokens_prompt": response.get("tokens_evaluated"),
        "tokens_predicted": tokens_predicted,
        "predicted_ms": predicted_ms,
        "predicted_tps": predicted_tps,
        "draft_generated": delta["draft_generated"],
        "draft_accepted": delta["draft_accepted"],
        "verification_cycles": verification_cycles,
        "draft_acceptance": draft_acceptance,
        "mean_accepted_length": mean_accepted_length,
        "accepted_count_by_position": delta["accepted_count_by_position"],
        "acceptance_rate_by_position": acceptance_rate_by_position,
    }


def run_session(transport: Transport, prompts: list[CorpusPrompt], config: SessionConfig) -> list[dict[str, Any]]:
    """Warmup (unmeasured) + two passes in independent deterministic
    shuffled orders. Fails closed (raises BenchmarkError) on the first
    bad request rather than silently skipping it and continuing --
    a partial/corrupted benchmark run is worse than an obvious failure.
    """
    validate_server(transport)

    records: list[dict[str, Any]] = [{
        "schema": SCHEMA,
        "kind": "session",
        "session_id": config.session_id,
        "corpus_id": config.corpus_id,
        "corpus_sha256": config.corpus_sha256,
        "bigcherry_revision": config.bigcherry_revision,
        "llama_pin": config.llama_pin,
        "llama_revision": config.llama_revision,
        "model_id": config.model_id,
        "server_argv": list(config.server_argv),
        "spec": {
            "type": config.spec_type,
            "n_max": config.spec_n_max,
            "draft_k": config.spec_draft_k,
            "draft_v": config.spec_draft_v,
        },
        "sampling": {
            "temperature": config.sampling.temperature,
            "top_p": config.sampling.top_p,
            "top_k": config.sampling.top_k,
        },
        "n_predict": config.n_predict,
        "parallel": 1,
        "order_seed": config.order_seed,
    }]

    # One unmeasured warmup request -- first-request cold-cache effects
    # (draft context, KV allocation) must not pollute pass 1's numbers.
    run_request(transport, prompts[0], config, pass_number=0, order_index=0)

    for pass_number in (1, 2):
        order = _deterministic_order([p.id for p in prompts], config.order_seed + pass_number)
        for order_index, prompt_index in enumerate(order):
            records.append(
                run_request(
                    transport, prompts[prompt_index], config,
                    pass_number=pass_number, order_index=order_index,
                )
            )
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


# ------------------------------------------------------------------ CLI


def _build_parser() -> argparse.ArgumentParser:
    # add_help=False: this parser is also composed via parents=[...] into
    # bigcherry's own top-level `completion-bench` subcommand
    # (cli/main.py) -- two -h/--help definitions in one parser chain
    # raises ArgumentError, so only the parent that actually owns
    # dispatch (standalone CLI use below, or the top-level bigcherry
    # parser) should define -h.
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--server-url", required=True, help="base URL of an already-running llama-server")
    parser.add_argument("--corpus", required=True, type=Path, help="path to a corpus JSONL (see corpora/mtp-27b-v1.jsonl)")
    parser.add_argument("--out", required=True, type=Path, help="output path for the completion-benchmark JSONL")
    parser.add_argument("--session-id", default=None, help="default: derived from the current time")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--bigcherry-revision", required=True)
    parser.add_argument("--llama-pin", required=True)
    parser.add_argument("--llama-revision", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--server-argv", default="", help="the server's own launch argv, space-separated, for provenance only")
    parser.add_argument("--spec-type", default="draft-mtp")
    parser.add_argument("--spec-n-max", type=int, required=True)
    parser.add_argument("--spec-draft-k", required=True)
    parser.add_argument("--spec-draft-v", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-predict", type=int, default=512)
    parser.add_argument("--order-seed", type=int, default=12345)
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Shared by main() (standalone CLI) and bigcherry's own `completion-
    bench` subcommand (cli/main.py, composed via parents=[_build_parser()])
    -- one implementation, so the two entry points can never drift apart."""
    prompts, corpus_sha256 = load_corpus(args.corpus)
    config = SessionConfig(
        session_id=args.session_id or f"session-{int(time.time())}",
        corpus_id=args.corpus_id,
        corpus_sha256=corpus_sha256,
        bigcherry_revision=args.bigcherry_revision,
        llama_pin=args.llama_pin,
        llama_revision=args.llama_revision,
        model_id=args.model_id,
        server_argv=tuple(args.server_argv.split()) if args.server_argv else (),
        spec_type=args.spec_type,
        spec_n_max=args.spec_n_max,
        spec_draft_k=args.spec_draft_k,
        spec_draft_v=args.spec_draft_v,
        sampling=SamplingConfig(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k),
        n_predict=args.n_predict,
        order_seed=args.order_seed,
    )
    transport = HttpTransport(args.server_url)
    try:
        records = run_session(transport, prompts, config)
    except BenchmarkError as exc:
        print(f"server-completion: {exc}", file=sys.stderr)
        return 2
    write_jsonl(records, args.out)
    print(f"wrote {len(records)} record(s) ({len(records) - 1} requests) to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Standalone CLI entry point (`python -m bigcherry.bench.server_completion`)."""
    parser = argparse.ArgumentParser(description=__doc__, parents=[_build_parser()])
    return run_from_namespace(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
