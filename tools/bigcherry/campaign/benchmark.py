"""Paired, interleaved native-versus-replay end-to-end benchmarks.

The command under test is deliberately identical for both arms.  This module
changes only ``GGML_HIP_DISPATCH_MODE`` (and adds the replay cache), alternates
the order within successive pairs, preserves every stdout/stderr log, and
reports the comparison using the pairs.  That keeps thermal and clock drift
from quietly becoming a dispatch result.

Example::

    python -m bigcherry ab-benchmark --cache dispatch.cache --output artifacts/tuning-runs/ab-2026-08-08 \
      --pairs 3 --metric "pp256=pp256.*?([0-9.]+)" -- \
      /home/audumla/bc-build/bin/llama-bench -m model.gguf -p 256 -n 0 -r 5 -ngl 99
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def pair_modes(pair_index: int) -> tuple[str, str]:
    """Return a balanced native/replay order for a zero-based pair index."""
    return ("native", "replay") if pair_index % 2 == 0 else ("replay", "native")


def schedule(rounds: int, *, include_stock: bool, seed: int = 0) -> list[tuple[str, ...]]:
    """Return reproducible complete six-order blocks for three-arm work."""
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if not include_stock:
        return [pair_modes(index) for index in range(rounds)]
    orders = list(itertools.permutations(("stock", "native", "replay")))
    rng = random.Random(seed)
    result: list[tuple[str, ...]] = []
    for _ in range(math.ceil(rounds / len(orders))):
        block = list(orders)
        rng.shuffle(block)
        result.extend(block)
    return result[:rounds]


def round_modes(round_index: int, *, include_stock: bool) -> tuple[str, ...]:
    """Compatibility-free public schedule accessor using the fixed seed."""
    return schedule(round_index + 1, include_stock=include_stock)[round_index]


def schedule_named_arms(rounds: int, arm_names: list[str], seed: int = 0) -> list[tuple[str, ...]]:
    """GP08: generalized version of schedule() for an arbitrary, named set
    of arms (not just native/replay/stock) -- same balanced-complete-
    permutation-block methodology, generalized to N arms. Used by
    collective_benchmark.py to compare an arbitrary provider/build arm set
    (native-rccl, bigcherry-rccl, internal, hybrid, meta, layer-split, ...)
    rather than this module's own fixed three-arm vocabulary."""
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if len(arm_names) < 2:
        raise ValueError("need at least 2 arms to compare")
    if len(set(arm_names)) != len(arm_names):
        raise ValueError("arm_names must be unique")
    orders = list(itertools.permutations(arm_names))
    rng = random.Random(seed)
    result: list[tuple[str, ...]] = []
    for _ in range(math.ceil(rounds / len(orders))):
        block = list(orders)
        rng.shuffle(block)
        result.extend(block)
    return result[:rounds]


_CMAKE_PARITY_KEYS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_C_COMPILER",
    "CMAKE_CXX_COMPILER",
    "GGML_HIP",
    "GGML_HIP_RCCL",
    "AMDGPU_TARGETS",
)


def cmake_settings(path: Path) -> dict[str, str]:
    """Read the compilation settings that must agree between stock and patched."""
    if not path.is_file():
        raise ValueError(f"CMake cache does not exist: {path}")
    settings: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("//") or line.startswith("#") or "=" not in line or ":" not in line:
            continue
        name_type, value = line.split("=", 1)
        name, _ = name_type.split(":", 1)
        if name in _CMAKE_PARITY_KEYS:
            settings[name] = value
    missing = [name for name in _CMAKE_PARITY_KEYS if not settings.get(name)]
    if missing:
        raise ValueError(f"CMake cache lacks parity settings: {', '.join(missing)}")
    return settings


def validate_build_parity(stock_cache: Path, patched_cache: Path) -> dict[str, str]:
    """Reject a stock comparison whose common compilation settings differ."""
    stock = cmake_settings(stock_cache)
    patched = cmake_settings(patched_cache)
    mismatches = [
        f"{name}: stock={stock[name]!r}, patched={patched[name]!r}"
        for name in _CMAKE_PARITY_KEYS
        if stock[name] != patched[name]
    ]
    if mismatches:
        raise ValueError("stock and patched CMake settings differ: " + "; ".join(mismatches))
    return {name: stock[name] for name in _CMAKE_PARITY_KEYS}


def build_fingerprint(settings: dict[str, str]) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16, person=b"bc-build-v1").hexdigest()


def parse_metric_specs(specs: list[str]) -> dict[str, re.Pattern[str]]:
    """Parse ``name=regex`` metric selectors, rejecting ambiguous regexes."""
    parsed: dict[str, re.Pattern[str]] = {}
    for spec in specs:
        name, separator, pattern = spec.partition("=")
        if not separator or not name or not pattern:
            raise ValueError("metric must be NAME=REGEX")
        if name in parsed:
            raise ValueError(f"metric named {name!r} was supplied more than once")
        compiled = re.compile(pattern, re.MULTILINE)
        if compiled.groups != 1:
            raise ValueError(
                f"metric {name!r} must have exactly one capture group for its value"
            )
        parsed[name] = compiled
    return parsed


def extract_metrics(text: str, patterns: dict[str, re.Pattern[str]]) -> dict[str, float]:
    """Extract the last reported value for every configured benchmark metric."""
    values: dict[str, float] = {}
    for name, pattern in patterns.items():
        matches = pattern.findall(text)
        if not matches:
            raise ValueError(f"could not find metric {name!r} in command output")
        raw = matches[-1]
        if isinstance(raw, tuple):  # defensive; parse_metric_specs rejects this.
            raw = raw[0]
        try:
            values[name] = float(raw.replace(",", ""))
        except ValueError as exc:
            raise ValueError(f"metric {name!r} is not numeric: {raw!r}") from exc
    return values


def extract_structured_metrics(text: str) -> dict[str, list[float]]:
    """Read current benchmark-result JSONL and preserve every repetition."""
    result: dict[str, list[float]] = {}
    seen = False
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("schema_version") != 1 or record.get("kind") != "benchmark_result":
            continue
        seen = True
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"structured result line {line_number} lacks metrics")
        for name, repetitions in metrics.items():
            if (not isinstance(name, str) or not isinstance(repetitions, list) or
                    not repetitions):
                raise ValueError(f"structured metric {name!r} has no repetitions")
            values = [float(value) for value in repetitions]
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError(f"structured metric {name!r} contains invalid values")
            result.setdefault(name, []).extend(values)
    if not seen or not result:
        raise ValueError("current benchmark_result JSONL was not found")
    return result


def sanitize_environment(source: dict[str, str], mode: str, cache: Path | None = None,
                         coverage: Path | None = None) -> dict[str, str]:
    """Construct an uncontaminated arm environment."""
    env = dict(source)
    exact = {
        "GGML_HIP_DISPATCH_MODE", "GGML_HIP_DISPATCH_CACHE",
        "GGML_HIP_DISPATCH_COVERAGE", "GGML_HIP_DISPATCH_MISS",
        "GGML_HIP_DISPATCH_OVERRIDE", "GGML_HIP_AUTOTUNE_MODE",
    }
    for key in list(env):
        if key in exact or key.startswith(("GGML_HIP_FORCE_", "GGML_HIP_TUNE_")):
            env.pop(key, None)
    if mode != "stock":
        env["GGML_HIP_DISPATCH_MODE"] = mode
    if mode == "replay":
        if cache is None or coverage is None:
            raise ValueError("replay arm requires cache and coverage")
        env["GGML_HIP_DISPATCH_CACHE"] = str(cache)
        env["GGML_HIP_DISPATCH_COVERAGE"] = str(coverage)
        env["GGML_HIP_DISPATCH_MISS"] = "native-record"
    return env


def _arm_metric(run: dict[str, Any], name: str) -> float | None:
    repetitions = run.get("repetitions", {}).get(name)
    if repetitions:
        return statistics.mean(float(value) for value in repetitions)
    value = run.get("metrics", {}).get(name)
    return float(value) if value is not None else None


def block_bootstrap_effect(
    runs: list[dict[str, Any]], candidate: str, reference: str, metric: str,
    *, lower_is_better: bool = False, seed: int = 0, resamples: int = 10_000,
) -> dict[str, Any]:
    """Estimate paired geometric effect and deterministic block-bootstrap CI."""
    by_round: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        by_round.setdefault(int(run["pair"]), {})[str(run["mode"])] = run
    ratios: list[float] = []
    for arms in by_round.values():
        if candidate not in arms or reference not in arms:
            continue
        candidate_value = _arm_metric(arms[candidate], metric)
        reference_value = _arm_metric(arms[reference], metric)
        if candidate_value and reference_value:
            ratio = candidate_value / reference_value
            ratios.append(1.0 / ratio if lower_is_better else ratio)
    if not ratios:
        raise ValueError(f"no complete paired rounds for metric {metric!r}")
    logs = [math.log(value) for value in ratios]
    effect = 100.0 * (math.exp(statistics.mean(logs)) - 1.0)
    rng = random.Random(seed)
    samples = sorted(
        100.0 * (math.exp(statistics.mean(rng.choice(logs) for _ in logs)) - 1.0)
        for _ in range(resamples)
    )
    return {
        "paired_rounds": len(ratios), "geometric_effect_pct": effect,
        "ci95_low_pct": samples[int(0.025 * resamples)],
        "ci95_high_pct": samples[min(resamples - 1, int(0.975 * resamples))],
        "resamples": resamples, "seed": seed,
        # VA24: the ordered per-pair ratio vector is the SUFFICIENT STATISTIC
        # for re-running this estimator -- the bootstrap's input is defined to
        # be these ratios, so retaining them allows an aggregate (multi-lane)
        # interval to be computed later without re-running the benchmark, and
        # without persisting 10,000 replicate values. Raw per-arm values are
        # deliberately NOT carried: they are unnecessary for reproducing a
        # ratio-based estimator. Note they WOULD be required for a future
        # estimator using absolute differences or weighting, so this is
        # sufficiency for the current estimator, not for all time
        # (dev-gpt-agent, req_a667633429fa4c9e).
        "pair_ratios": tuple(ratios),
    }


def validate_replay_coverage(path: Path) -> dict[str, Any]:
    """Return replay provenance, rejecting coverage that cannot support a claim."""
    if not path.is_file():
        raise ValueError("replay command did not write its coverage file")
    try:
        coverage = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read replay coverage: {exc}") from exc
    dispatched = coverage.get("total_dispatched")
    executed = coverage.get("total_executed")
    replay = coverage.get("replay")
    if not isinstance(dispatched, int) or not isinstance(executed, int):
        raise ValueError("coverage lacks total_dispatched/total_executed")
    if dispatched != executed:
        raise ValueError(f"coverage incomplete: dispatched={dispatched}, executed={executed}")
    if not isinstance(replay, dict):
        raise ValueError("coverage lacks replay provenance; use a replay-capable build")
    if replay.get("schema_version") != 2:
        raise ValueError("coverage is not from current replay v2; rerun required")
    outcomes = {
        name: replay.get(name)
        for name in ("exact", "candidate_unavailable", "rerun_required", "incompatible", "misses")
    }
    if any(not isinstance(value, int) for value in outcomes.values()):
        raise ValueError("replay coverage lacks current resolution outcomes")
    unexplained = sum(outcomes[name] for name in outcomes if name != "exact")
    if unexplained:
        raise ValueError(
            "replay has non-exact resolution outcomes: " +
            ", ".join(f"{name}={value}" for name, value in outcomes.items() if name != "exact" and value)
        )
    return {
        "entries": replay.get("entries"),
        **outcomes,
        "total_dispatched": dispatched,
        "total_executed": executed,
    }


def paired_summary(
    runs: list[dict[str, Any]], directions: dict[str, str]
) -> dict[str, dict[str, float | int]]:
    """Summarise replay's paired advantage; positive percentages favour replay."""
    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        by_pair.setdefault(int(run["pair"]), {})[str(run["mode"])] = run

    metric_pairs: dict[str, list[float]] = {}
    for pair in by_pair.values():
        if set(pair) != {"native", "replay"}:
            continue
        native_metrics = pair["native"].get("metrics", {})
        replay_metrics = pair["replay"].get("metrics", {})
        for name in sorted(set(native_metrics) & set(replay_metrics)):
            native = float(native_metrics[name])
            replay = float(replay_metrics[name])
            if native == 0:
                continue
            delta = 100.0 * (replay - native) / native
            if directions.get(name, "higher") == "lower":
                delta = -delta
            metric_pairs.setdefault(name, []).append(delta)

    result: dict[str, dict[str, float | int]] = {}
    for name, deltas in metric_pairs.items():
        result[name] = {
            "pairs": len(deltas),
            "mean_pct": statistics.mean(deltas),
            "median_pct": statistics.median(deltas),
            "min_pct": min(deltas),
            "max_pct": max(deltas),
            "stdev_pct": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        }
    return result


def comparison_summaries(
    runs: list[dict[str, Any]], directions: dict[str, str], *, include_stock: bool
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Produce all paired comparisons required by this run.

    The existing native/replay comparison remains the tuning result; with stock
    enabled, native/stock exposes patch overhead and replay/stock is the release
    result users will actually observe.
    """
    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        by_pair.setdefault(int(run["pair"]), {})[str(run["mode"])] = run

    def compare(candidate: str, reference: str) -> dict[str, dict[str, float | int]]:
        selected: list[dict[str, Any]] = []
        for pair_id, arms in by_pair.items():
            if candidate not in arms or reference not in arms:
                continue
            selected.append({"pair": pair_id, "mode": "native", "metrics": arms[reference].get("metrics", {})})
            selected.append({"pair": pair_id, "mode": "replay", "metrics": arms[candidate].get("metrics", {})})
        return paired_summary(selected, directions)

    results = {"replay_vs_native": compare("replay", "native")}
    if include_stock:
        results["native_vs_stock"] = compare("native", "stock")
        results["replay_vs_stock"] = compare("replay", "stock")
    return results


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# End-to-end native versus replay comparison",
        "",
        f"Command: `{' '.join(summary['command'])}`",
        f"Rounds: {summary['pairs']} (interleaved; settle {summary['settle_seconds']} s)",
        f"Replay cache: `{summary['cache']}`",
        "",
        "Positive deltas favour replay. Results are paired by adjacent native/replay runs.",
        "",
        "| Metric | Replay vs native | Native vs stock | Replay vs stock |",
        "| --- | ---: | ---: | ---: |",
    ]
    metric_names = sorted({name for comparison in summary["comparisons"].values() for name in comparison})
    for name in metric_names:
        values = []
        for comparison in ("replay_vs_native", "native_vs_stock", "replay_vs_stock"):
            data = summary["comparisons"].get(comparison, {}).get(name)
            values.append(
                "-" if data is None else "{:+.2f}% (median {:+.2f}%, n={})".format(
                    float(data["mean_pct"]), float(data["median_pct"]), int(data["pairs"])
                )
            )
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    if not metric_names:
        lines.append("| wall_clock_s | no parsed benchmark metric supplied | - | - |")
    if summary.get("decisions"):
        lines.extend(["", "## Decision-grade intervals", "",
                      "| Comparison | Metric | Geometric effect | 95% interval | Decision |",
                      "| --- | --- | ---: | ---: | --- |"])
        for comparison, metrics in summary["decisions"].items():
            for name, evidence in metrics.items():
                lines.append(
                    f"| {comparison} | {name} | {evidence['geometric_effect_pct']:+.2f}% | "
                    f"[{evidence['ci95_low_pct']:+.2f}%, {evidence['ci95_high_pct']:+.2f}%] | "
                    f"{evidence.get('decision', 'inconclusive')} |"
                )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "Each arm's stdout/stderr and replay coverage JSON are retained beside this report. "
            "Reject a replay result with cache misses, stale winners, or a non-zero command exit.",
            "",
        ]
    )
    return "\n".join(lines)


def run_arm_capture(
    *,
    command: list[str],
    cwd: Path | None,
    output: Path,
    pair: int,
    side: str,
    env: dict[str, str],
    patterns: dict[str, re.Pattern[str]],
    structured: bool = False,
    replay: bool = False,
) -> dict[str, Any]:
    """Run one arm and retain complete evidence even when it fails.

    RE12: extracted from the original ``_run_arm`` so comparisons.py's
    identity-aware, descriptor-backed arm runner can reuse the exact same
    subprocess/evidence-capture/statistics code instead of a second
    benchmark framework -- ``side`` is an opaque label (``"native"``,
    ``"replay"``, ``"left"``, ``"right"``, ...) used only for output
    filenames and pairing keys; the caller builds ``env`` (this module's
    own callers still go through ``sanitize_environment``) and decides
    whether ``replay`` coverage validation applies.
    """
    stem = f"pair-{pair + 1:03d}-{side}"
    coverage = output / f"{stem}.coverage.json"

    started = time.time()
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, env=env)
    elapsed = time.time() - started
    (output / f"{stem}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / f"{stem}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    combined = completed.stdout + "\n" + completed.stderr
    run: dict[str, Any] = {
        "pair": pair + 1,
        "mode": side,
        "returncode": completed.returncode,
        "elapsed_s": elapsed,
        "stdout": f"{stem}.stdout.log",
        "stderr": f"{stem}.stderr.log",
    }
    if replay:
        run["coverage"] = coverage.name
    if completed.returncode == 0:
        try:
            if structured:
                run["repetitions"] = extract_structured_metrics(combined)
                run["metrics"] = {
                    name: statistics.mean(values)
                    for name, values in run["repetitions"].items()
                }
            else:
                run["metrics"] = extract_metrics(combined, patterns)
            if replay:
                run["replay_provenance"] = validate_replay_coverage(coverage)
        except ValueError as exc:
            run["metric_error"] = str(exc)
    return run


def _run_arm(
    *,
    command: list[str],
    cwd: Path | None,
    output: Path,
    pair: int,
    mode: str,
    cache: Path,
    patterns: dict[str, re.Pattern[str]],
    structured: bool = False,
) -> dict[str, Any]:
    """This module's own native/replay/stock CLI arm: builds the
    dispatch-mode env via sanitize_environment(), then delegates to the
    reusable run_arm_capture()."""
    coverage = output / f"pair-{pair + 1:03d}-{mode}.coverage.json"
    env = sanitize_environment(os.environ, mode, cache, coverage)
    return run_arm_capture(
        command=command, cwd=cwd, output=output, pair=pair, side=mode, env=env,
        patterns=patterns, structured=structured, replay=mode == "replay",
    )


def run_server_arm_capture(
    *, binary: Path, model: Path, extra_args: tuple[str, ...],
    output: Path, pair: int, side: str, position: int,
    env: dict[str, str], bench_configs: str, runner_root: Path,
    required_metrics: tuple[str, ...], repetitions: int = 1,
    shutdown_method: str = "http",
) -> dict[str, Any]:
    """Capture one server-bench cell using the maintained process lifecycle.

    This is measurement capture, NOT performance admission: source/binary
    identity, activation and work-equivalence must be checked by the caller.
    Full runner streams and teardown results survive failed cells too.
    """
    from dataclasses import asdict
    from bigcherry.campaign.bench_runner import run_bench_runner_server_bench
    from bigcherry.tuning.server_runner import ServerRunner

    if not required_metrics:
        raise ValueError("server cells require explicit expected metrics")
    if pair < 0 or position < 0 or not re.fullmatch(r"[A-Za-z0-9_-]+", side):
        raise ValueError("invalid cell identity")
    cell = output / f"pair-{pair + 1:03d}-{side}"
    cell.mkdir(parents=True, exist_ok=False)
    run: dict[str, Any] = {
        "pair": pair + 1, "mode": side, "position": position,
        "returncode": 1, "performance_admitted": False,
        "server_log": str(cell / "server.log"),
    }
    # Supply a complete, caller-sanitized environment. Merely overlaying it
    # would resurrect dispatch/tuning variables removed by the caller.
    server = ServerRunner(
        binary=binary, model=model, extra_args=extra_args,
        env_overrides=env, env_unset=tuple(os.environ),
        log_path=cell / "server.log", shutdown_method=shutdown_method,
    )
    started = time.monotonic()
    try:
        with server:
            run["metrics"] = run_bench_runner_server_bench(
                server_url=f"http://{server.host}:{server.port}",
                bench_configs=bench_configs, repetitions=repetitions,
                runner_root=runner_root, model_label=model.stem,
                evidence_dir=cell / "bench", required_metrics=required_metrics,
            )
        if server.last_shutdown is None or not server.last_shutdown.clean:
            raise ValueError("server teardown was not clean; cell rejected")
        run["returncode"] = 0
    except Exception as exc:
        run["metric_error"] = str(exc)
        # A failed cell must not accidentally contribute partially captured
        # throughput to callers that select rows by metric presence.
        run.pop("metrics", None)
    finally:
        run["elapsed_s"] = time.monotonic() - started
        run["shutdown"] = asdict(server.last_shutdown) if server.last_shutdown else None
        (cell / "cell.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return run


def run_server_comparison_capture(
    config_path: Path, output: Path, *, rounds: int, seed: int, settle_seconds: float,
) -> int:
    """Thin server lifecycle integration over the existing balanced A/B engine.

    Captures are explicitly unadmitted: build configuration observations and
    paired estimates do not establish execution, work equivalence or tuning
    activation. Keep those missing gates visible rather than implying that a
    successful subprocess constitutes a production performance result.
    """
    from bigcherry.build.builds import binary_hash, inspect_dispatch_build, resolve_runtime_artifacts
    from bigcherry.campaign.bench_runner import _resolve_runner_root

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("server capture configuration requires schema_version 1")
    arms = config.get("arms", [])
    if not isinstance(arms, list) or not 2 <= len(arms) <= 3:
        raise ValueError("server comparisons require two or three arms; use separate balanced contrasts for larger matrices")
    if rounds < 2 or rounds % math.factorial(len(arms)) or not math.isfinite(settle_seconds) or settle_seconds < 0:
        raise ValueError("server captures require complete permutation blocks and nonnegative finite settling")
    names = [arm["name"] for arm in arms]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
        raise ValueError("arm names must be unique simple identifiers")
    role = config.get("evidence_role")
    if role not in ("production", "diagnostic"):
        raise ValueError("explicit production or diagnostic evidence_role required")
    model = Path(os.path.expandvars(config["model"])).expanduser().resolve()
    extra_args = tuple(os.path.expandvars(value) for value in config["server_args"])
    if any(value.split("=", 1)[0] in ("-m", "--model", "--host", "--port") for value in extra_args):
        raise ValueError("server_args cannot override the managed model or endpoint")
    metrics = tuple(config["required_metrics"])
    if not metrics or any(not re.fullmatch(r"\w+_tps", name) for name in metrics):
        raise ValueError("explicit server-bench throughput metrics required")
    repetitions = config.get("repetitions", 1)
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    supplied_env = {key: os.path.expandvars(value) for key, value in config["environment"].items()}
    if any(not supplied_env.get(key) for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        raise ValueError("both explicit device selectors are required")
    runner_root = _resolve_runner_root(Path(os.path.expandvars(config["runner_root"])) if config.get("runner_root") else None)
    prepared = {}
    for arm in arms:
        binary = Path(os.path.expandvars(arm["binary"])).expanduser().resolve()
        mode = arm["mode"]
        if mode not in ("stock", "native", "record", "tune", "replay"):
            raise ValueError(f"unsupported dispatch mode: {mode}")
        build_dir = binary.parent.parent
        observation = inspect_dispatch_build(build_dir)
        if observation["issues"] or bool(observation["instrumented"]) != (role == "diagnostic"):
            raise ValueError(f"{arm['name']}: compiled instrumentation disagrees with evidence role")
        metadata_path = build_dir / f"bigcherry-build-metadata-{binary.name}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        runtime = {path.name: binary_hash(path) for path in resolve_runtime_artifacts(binary)}
        if metadata.get("binary_hash") != binary_hash(binary) or any(
                metadata.get("runtime_artifacts", {}).get(name) != digest for name, digest in runtime.items()):
            raise ValueError(f"{arm['name']}: runtime bytes disagree with campaign metadata")
        env = sanitize_environment({**os.environ, **supplied_env}, "stock" if mode == "stock" else "native")
        for key in tuple(env):
            if key.startswith(("GGML_HIP_DISPATCH_", "GGML_HIP_AUTOTUNE_", "GGML_HIP_TUNE_", "GGML_HIP_FORCE_")):
                env.pop(key)
        if mode != "stock":
            env["GGML_HIP_DISPATCH_MODE"] = mode
        # Only explicitly supplied arm controls survive ambient sanitization.
        controls = {key: os.path.expandvars(value) for key, value in arm.get("environment", {}).items()}
        if any(not key.startswith(("GGML_HIP_DISPATCH_", "GGML_HIP_TUNE_", "GGML_HIP_AUTOTUNE_")) for key in controls):
            raise ValueError("arm-specific environment is limited to dispatch/tuning controls; topology belongs to the shared environment")
        if "GGML_HIP_DISPATCH_MODE" in controls:
            raise ValueError("arm environment cannot override the declared dispatch mode")
        env.update(controls)
        if mode == "replay" and not Path(env.get("GGML_HIP_DISPATCH_CACHE", "")).is_file():
            raise ValueError("replay arm requires an existing explicit cache")
        prepared[arm["name"]] = {
            "binary": binary, "env": env, "shutdown_method": arm.get("shutdown_method", "sigint" if mode == "stock" else "http"),
            "provenance": {"campaign_metadata": metadata, "observed_runtime_artifacts": runtime,
                           "compiler_observation": observation},
        }
    if "stock" in names and "native" in names:
        validate_build_parity(prepared["stock"]["binary"].parent.parent / "CMakeCache.txt",
                              prepared["native"]["binary"].parent.parent / "CMakeCache.txt")
    output.mkdir(parents=True, exist_ok=False)
    run_schedule = schedule_named_arms(rounds, names, seed)
    summary = {
        "schema_version": 1, "capture_kind": "server-bench-balanced-v1", "evidence_role": role,
        "performance_admitted": False, "admission_blockers": [
            "execution identity and work equivalence require verification",
            "replay activation, if applicable, requires separate admission",
        ],
        "configuration": config, "model_sha256": binary_hash(model),
        "arm_provenance": {name: value["provenance"] for name, value in prepared.items()},
        "schedule": run_schedule, "schedule_seed": seed, "settle_seconds": settle_seconds,
        "runs": [],
    }
    def persist():
        (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    persist()
    for pair, order in enumerate(run_schedule):
        for position, name in enumerate(order):
            arm = prepared[name]
            print(f"[server-capture] round {pair + 1}/{rounds} position {position + 1}: {name}", flush=True)
            result = run_server_arm_capture(
                binary=arm["binary"], model=model, extra_args=extra_args, output=output,
                pair=pair, side=name, position=position, env=arm["env"],
                bench_configs=config["bench_configs"], runner_root=runner_root,
                required_metrics=metrics, repetitions=repetitions, shutdown_method=arm["shutdown_method"],
            )
            summary["runs"].append(result)
            persist()
            if result["returncode"]:
                return 1
            if (pair, position) != (rounds - 1, len(order) - 1) and settle_seconds:
                time.sleep(settle_seconds)
    summary["exploratory_comparisons"] = {
        f"{candidate}_vs_{names[0]}": {metric: block_bootstrap_effect(summary["runs"], candidate, names[0], metric)
                                      for metric in metrics}
        for candidate in names[1:]
    }
    persist()
    print("Capture complete; performance admission remains false.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bigcherry ab-benchmark",
        description="Run paired, interleaved native-versus-replay end-to-end benchmarks.",
    )
    parser.add_argument("--inspect-build", type=Path, help="read-only dispatch/diagnostic compiler inventory; no benchmark is launched")
    parser.add_argument("--server-config", type=Path, help="capture balanced server-bench arms from a local JSON run configuration; not performance admission")
    parser.add_argument("--cache", help="replay cache exported from this tune")
    parser.add_argument("--output", help="new artifacts/tuning-runs/<run> directory")
    parser.add_argument("--pairs", type=int, default=3, help="interleaved rounds per arm (default: 3; use power for a decision-grade count)")
    parser.add_argument("--schedule-seed", type=int, default=0)
    parser.add_argument("--structured", action="store_true", help="require current benchmark_result JSONL and retain all repetitions")
    parser.add_argument("--practical-threshold-pct", type=float, default=1.0)
    parser.add_argument("--decision-grade", action="store_true", help="require stock, complete six-order blocks, structured metrics and >=5 repetitions")
    parser.add_argument("--settle-seconds", type=float, default=20.0, help="idle time between arms (default: 20)")
    parser.add_argument("--cwd", default=None, help="working directory for the benchmark command")
    parser.add_argument(
        "--metric", action="append", default=[], metavar="NAME=REGEX",
        help="extract a metric from output; regex must have one numeric capture group; may repeat",
    )
    parser.add_argument(
        "--lower-is-better", action="append", default=[], metavar="NAME",
        help="treat this metric as latency rather than throughput; may repeat",
    )
    parser.add_argument("--stock-binary", default=None, help="unmodified llama.cpp executable from the same release")
    parser.add_argument("--stock-cmake-cache", default=None, help="stock build's CMakeCache.txt; required with --stock-binary")
    parser.add_argument("--patched-cmake-cache", default=None, help="patched build's CMakeCache.txt; required with --stock-binary")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="benchmark command, after --")
    args = parser.parse_args(argv)

    if args.server_config is not None:
        if (not args.output or args.command or args.cache or args.stock_binary
                or args.inspect_build or args.decision_grade or args.structured or args.metric):
            parser.error("--server-config requires --output and cannot use command-mode arms or decision-grade admission")
        try:
            return run_server_comparison_capture(
                args.server_config, Path(args.output), rounds=args.pairs,
                seed=args.schedule_seed, settle_seconds=args.settle_seconds,
            )
        except (ValueError, OSError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.inspect_build is not None:
        if args.command or args.cache or args.output or args.stock_binary:
            parser.error("--inspect-build cannot be combined with a benchmark command or arms")
        from bigcherry.build.builds import BuildIdentityError, inspect_dispatch_build
        try:
            observation = inspect_dispatch_build(args.inspect_build)
        except (BuildIdentityError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(observation, indent=2))
        return 1 if observation["issues"] else 0
    if not args.cache or not args.output:
        parser.error("benchmark execution requires --cache and --output")

    if args.pairs < 1 or args.settle_seconds < 0:
        print("error: --pairs must be >= 1 and --settle-seconds must be >= 0", file=sys.stderr)
        return 2
    if args.decision_grade and (args.pairs % 6 or not args.structured):
        print("error: decision-grade runs require --structured and a multiple of six rounds", file=sys.stderr)
        return 2
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("error: supply the unchanged benchmark command after --", file=sys.stderr)
        return 2
    cache = Path(args.cache)
    output = Path(args.output)
    if not cache.is_file():
        print(f"error: replay cache does not exist: {cache}", file=sys.stderr)
        return 2
    stock_values = (args.stock_binary, args.stock_cmake_cache, args.patched_cmake_cache)
    if any(stock_values) and not all(stock_values):
        print("error: --stock-binary requires both --stock-cmake-cache and --patched-cmake-cache", file=sys.stderr)
        return 2
    if args.decision_grade and not args.stock_binary:
        print("error: decision-grade runs require a stock arm", file=sys.stderr)
        return 2
    stock_command: list[str] | None = None
    parity: dict[str, str] | None = None
    if args.stock_binary:
        stock_binary = Path(args.stock_binary)
        if not stock_binary.is_file():
            print(f"error: stock binary does not exist: {stock_binary}", file=sys.stderr)
            return 2
        try:
            parity = validate_build_parity(Path(args.stock_cmake_cache), Path(args.patched_cmake_cache))
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        stock_command = [str(stock_binary), *command[1:]]
    if output.exists():
        print(f"error: output directory already exists: {output}", file=sys.stderr)
        return 2
    try:
        patterns = parse_metric_specs(args.metric)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    unknown_directions = set(args.lower_is_better) - set(patterns)
    if unknown_directions:
        print("error: --lower-is-better names a metric that was not supplied: " + ", ".join(sorted(unknown_directions)), file=sys.stderr)
        return 2

    output.mkdir(parents=True)
    cwd = Path(args.cwd) if args.cwd else None
    runs: list[dict[str, Any]] = []
    run_schedule = schedule(args.pairs, include_stock=stock_command is not None, seed=args.schedule_seed)
    for pair, modes in enumerate(run_schedule):
        for arm, mode in enumerate(modes):
            run = _run_arm(command=stock_command if mode == "stock" else command, cwd=cwd, output=output, pair=pair, mode=mode, cache=cache, patterns=patterns, structured=args.structured)
            runs.append(run)
            if run["returncode"] != 0 or "metric_error" in run:
                details = run.get("metric_error", f"command exit {run['returncode']}")
                (output / "run.json").write_text(json.dumps({"command": command, "runs": runs, "error": details}, indent=2) + "\n", encoding="utf-8")
                print(f"failed during pair {pair + 1} {mode}: {details}", file=sys.stderr)
                return 1
            if not (pair == args.pairs - 1 and arm == len(modes) - 1) and args.settle_seconds:
                time.sleep(args.settle_seconds)

    directions = {name: "lower" for name in args.lower_is_better}
    summary: dict[str, Any] = {
        "command": command,
        "cache": str(cache),
        "pairs": args.pairs,
        "settle_seconds": args.settle_seconds,
        "metrics": sorted(patterns),
        "directions": directions,
        "runs": runs,
        "schedule_seed": args.schedule_seed,
        "schedule": [list(order) for order in run_schedule],
        "decision_grade": args.decision_grade,
        "practical_threshold_pct": args.practical_threshold_pct,
        "comparisons": comparison_summaries(runs, directions, include_stock=stock_command is not None),
    }
    if args.structured:
        minimum_repetitions = min(
            len(values) for run in runs for values in run.get("repetitions", {}).values()
        )
        summary["minimum_internal_repetitions"] = minimum_repetitions
        if args.decision_grade and minimum_repetitions < 5:
            summary["error"] = "decision-grade runs require at least five internal repetitions per metric"
            (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print("error: " + summary["error"], file=sys.stderr)
            return 1
        decisions: dict[str, Any] = {}
        for comparison, (candidate, reference) in {
            "replay_vs_native": ("replay", "native"),
            **({"native_vs_stock": ("native", "stock"), "replay_vs_stock": ("replay", "stock")} if stock_command else {}),
        }.items():
            decisions[comparison] = {}
            metric_names = sorted({name for run in runs for name in run.get("repetitions", {})})
            for metric in metric_names:
                evidence = block_bootstrap_effect(
                    runs, candidate, reference, metric,
                    lower_is_better=directions.get(metric) == "lower",
                    seed=args.schedule_seed,
                )
                evidence["decision"] = (
                    "improved" if evidence["ci95_low_pct"] >= args.practical_threshold_pct else
                    "regressed" if evidence["ci95_high_pct"] <= -args.practical_threshold_pct else
                    "inconclusive"
                )
                decisions[comparison][metric] = evidence
        summary["decisions"] = decisions
    if stock_command is not None:
        summary["stock_command"] = stock_command
        summary["compilation_parity"] = parity
        summary["build_fingerprint"] = build_fingerprint(parity or {})
    (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(f"wrote {output / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
