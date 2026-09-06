"""Shared server-benchmark runner -- the DOCUMENTED measurement harness.

Promoted out of patch/validation_campaign.py (where it was reachable only by
importing patch internals) because it is not patch-specific: any lane that
needs real end-to-end throughput from a running llama-server uses it.

WHY NOT llama-bench. llama-bench cannot measure what production runs: it has
no speculative/MTP flags at all, and on this project's real 27B/dual-GPU
-sm tensor config it has repeatedly failed outright (OOM under contention, and
a hard argument-parse error for --fit, which it does not register). The
documented path is docs/reference/testing/TEST.md's "Server benchmark (Brutus
bench runner)": start a server ourselves, then drive it with
`bench/run_bench.py --bench-type server-bench`.

Bench configs live in the harness's own bench/config/bench-configs.json --
`mtp-dual` is the MTP-speculative set matching the production dual-XTX
baseline, `default` is pp512+tg128.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import subprocess
import sys
from pathlib import Path


class BenchRunnerError(RuntimeError):
    """The documented bench harness did not produce a usable measurement."""


# Compatibility export retained for callers that import the old name.  The
# default is resolved at call time so importing this module does not require a
# machine-local environment document to exist.
BENCH_RUNNER_ROOT: Path | None = None
_BENCH_RUNNER_AGGREGATED_RESULT_PATTERN = re.compile(r"^\s*(\w+_tps):\s+([0-9.]+)\s*$")


def _resolve_runner_root(runner_root: Path | None) -> Path:
    if runner_root is not None:
        return Path(runner_root)
    harness = os.environ.get("BC_BENCH_HARNESS", "").strip()
    if harness:
        # BC_BENCH_HARNESS points at bench/, while the runner's cwd is the
        # harness repository root containing bench/run_bench.py.
        return Path(harness).expanduser().parent
    # Delayed import/load is intentional: missing config/environment.toml is
    # reported when a benchmark is requested, not while importing the module.
    from bigcherry.core.environment import load_default
    return Path(load_default().host().bench_harness).expanduser().parent


def run_bench_runner_server_bench(
    *, server_url: str, bench_configs: str, repetitions: int = 1, timeout_s: int = 300,
    runner_root: Path | None = None,
    model_label: str = "rd73-va06", evidence_dir: Path | None = None,
    required_metrics: tuple[str, ...] = (),
) -> dict[str, float]:
    """VA06 (user redirect, 2026-09-01): drive an already-running
    llama-server via the documented Brutus bench harness
    (docs/reference/testing/TEST.md's "Server benchmark (Brutus bench
    runner)" section: `cd .../llamacpp && python3 bench/run_bench.py
    --bench-type server-bench --server-url ... --bench-configs ...`) --
    NOT a raw llama-bench subprocess. llama-bench itself has proven
    unworkable for RD73's real 27B/dual-GPU/-sm-tensor config on real
    Brutus hardware (repeated real crashes this session: OOM under
    resource contention with production traffic, and a hard
    argument-parse error for --fit, which llama-bench does not even
    register). Parses the real "Extracted Results"/"Aggregated Results"
    stdout blocks (bench/runners/server_base.py and
    bench/lib/bench_orchestrator.py print one or the other depending on
    bench type -- server-bench mode, used here, prints "Extracted
    Results"; both share the same "  <name>_tps: <value>" per-config
    line format, confirmed directly against a real Brutus run) for every
    <name>_tps metric. Fails closed on a missing runner script, nonzero
    exit, or no parseable metric at all."""
    if repetitions < 1 or timeout_s <= 0 or not model_label.strip():
        raise BenchRunnerError("positive repetitions/timeout and a nonempty model label are required")
    runner_root = _resolve_runner_root(runner_root)
    runner_path = runner_root / "bench" / "run_bench.py"
    if not runner_path.is_file():
        raise BenchRunnerError(f"bench runner not found at {runner_path}")
    command = [
        sys.executable, str(runner_path),
        "--bench-type", "server-bench", "--server-url", server_url,
        "--model", model_label, "--bench-configs", bench_configs,
        "--toggles", json.dumps({"repetitions": repetitions}),
    ]
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=False)
        config_path = runner_root / "bench" / "config" / "bench-configs.json"
        (evidence_dir / "request.json").write_text(json.dumps({
            "command": command, "cwd": str(runner_root),
            "runner_sha256": hashlib.sha256(runner_path.read_bytes()).hexdigest(),
            "bench_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()
            if config_path.is_file() else None,
            "required_metrics": list(required_metrics),
        }, indent=2) + "\n", encoding="utf-8")

    def retain(stdout, stderr, returncode, *, timed_out=False):
        if evidence_dir is None:
            return
        for name, value in (("stdout", stdout), ("stderr", stderr)):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            (evidence_dir / f"{name}.log").write_text(value or "", encoding="utf-8")
        (evidence_dir / "exit.json").write_text(json.dumps({
            "returncode": returncode, "timed_out": timed_out,
        }) + "\n", encoding="utf-8")

    try:
        completed = subprocess.run(
            command, cwd=str(runner_root), capture_output=True, text=True,
            check=False, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        retain(exc.stdout, exc.stderr, None, timed_out=True)
        raise BenchRunnerError(f"bench runner timed out after {timeout_s}s") from exc
    retain(completed.stdout, completed.stderr, completed.returncode)
    if completed.returncode != 0:
        raise BenchRunnerError(
            f"bench runner failed (exit {completed.returncode}) against {server_url}: "
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    metrics: dict[str, float] = {}
    in_block = False
    for line in completed.stdout.splitlines():
        if "Aggregated Results" in line or "Extracted Results" in line:
            in_block = True
            continue
        if in_block:
            match = _BENCH_RUNNER_AGGREGATED_RESULT_PATTERN.match(line)
            if match:
                metrics[match.group(1)] = float(match.group(2))
    if not metrics:
        raise BenchRunnerError(
            f"bench runner produced no parseable <name>_tps metric against {server_url}; "
            f"stdout tail:\n{completed.stdout[-2000:]}"
        )
    missing = set(required_metrics) - metrics.keys()
    if missing:
        raise BenchRunnerError(f"bench runner omitted required metrics: {', '.join(sorted(missing))}")
    if any(not math.isfinite(value) or value <= 0 for value in metrics.values()):
        raise BenchRunnerError("bench runner returned non-positive or non-finite throughput")
    return metrics
