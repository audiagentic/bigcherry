"""PVPS03: per-session reference ladder across the four validation arms.

Every standard campaign session measures the same llama-bench workloads on
stock llama.cpp, base BC, validated BC (control) and validated BC + patch
(subject), so the project keeps an ongoing record of where each layer
stands, not only the focal patch's delta. The ladder is REFERENCE evidence:
it never feeds the contract verdict (that stays the producer's paired
control/subject lanes).

Arms that resolve to the same binary (base == validated while the promoted
set is empty) are measured once and reported under every name. The arm
order rotates every round (rounds are a multiple of the distinct-arm count)
and per-position means are recorded, so drift cannot masquerade as an arm
effect.

Shared arms (stock / base / validated) do not depend on the patch: their
binaries live in content-addressed build dirs shared by every patch, so a
shared arm is measured once per (binary, model, device, workloads) and every
later session reuses those samples from ``cache_dir``; only arms without a
cached measurement (normally just the subject) are run.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bigcherry.patch.campaign.benchmark import (
    _PAIRED_BENCH_METRIC_NAME,
    _PAIRED_BENCH_METRIC_PATTERN,
    _PAIRED_BENCH_WORKLOAD_FLAGS,
)

LADDER_ARTIFACT = "reference-ladder.json"
LADDER_SCHEMA = 1


@dataclass(frozen=True)
class LadderRun:
    arm: str
    workload: str
    round: int
    position: int
    value: float | None
    returncode: int


Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _distinct_arms(arms: dict[str, Path]) -> dict[Path, list[str]]:
    by_binary: dict[Path, list[str]] = {}
    for name, binary in arms.items():
        by_binary.setdefault(Path(binary).resolve(), []).append(name)
    return by_binary


def _cache_path(
    cache_dir: Path, binary_dir: Path, model: Path, device_key: str,
    workloads: tuple[str, ...], rounds_per_arm: int,
) -> Path:
    stat = model.stat()
    key = json.dumps({
        "binary_dir": str(binary_dir), "model": str(model.resolve()), "model_size": stat.st_size,
        "model_mtime_ns": stat.st_mtime_ns, "device": device_key, "workloads": sorted(workloads),
        "rounds_per_arm": rounds_per_arm,
    }, sort_keys=True)
    return cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.json"


def run_reference_ladder(
    *,
    arms: dict[str, Path],
    model: Path,
    runner: Runner,
    cache_dir: Path,
    shared_arms: frozenset[str],
    device_key: str,
    workloads: tuple[str, ...] = ("decode", "prefill"),
    rounds_per_arm: int = 2,
    exe: str = "",
) -> dict[str, object]:
    """Measure every distinct arm binary without a cached measurement,
    rotated, and summarise per arm.

    One llama-bench invocation per (round, arm) measures every workload
    together (-p 512 -n 128 in one process), so the model loads once per
    invocation rather than once per workload (same samples, fewer loads).
    Arms named in ``shared_arms`` are read from / written to ``cache_dir``."""
    runs: list[LadderRun] = []
    cached: list[str] = []
    to_measure: list[tuple[Path, list[str]]] = []
    distinct = _distinct_arms(arms)
    # Every arm keeps the same sample count whether or not the others were
    # cached: one sample per rotation round, len(distinct) x rounds_per_arm.
    rounds = len(distinct) * rounds_per_arm
    for binary_dir, names in distinct.items():
        path = _cache_path(cache_dir, binary_dir, model, device_key, workloads, rounds_per_arm)
        entry = json.loads(path.read_text(encoding="utf-8")) if set(names) & shared_arms and path.is_file() else None
        if entry is not None and all(entry["samples"].get(w) for w in workloads):
            cached.extend(names)
            for workload in workloads:
                for i, value in enumerate(entry["samples"][workload]):
                    runs.append(LadderRun(arm=names[0], workload=workload, round=i, position=-1,
                                          value=value, returncode=0))
        else:
            to_measure.append((binary_dir, names))
    groups = to_measure
    n = len(groups)
    flags: list[str] = []
    for workload in ("prefill", "decode"):
        if workload in workloads:
            flags += list(_PAIRED_BENCH_WORKLOAD_FLAGS[workload])
    prompt = next((flags[i + 1] for i, f in enumerate(flags) if f == "-p" and flags[i + 1] != "0"), "0")
    gen = next((flags[i + 1] for i, f in enumerate(flags) if f == "-n" and flags[i + 1] != "0"), "0")
    for r in range(rounds if n else 0):
        order = groups[r % n:] + groups[: r % n]
        for position, (binary_dir, names) in enumerate(order):
            command = [str(binary_dir / f"llama-bench{exe}"), "-m", str(model), "-p", prompt, "-n", gen, "-ngl", "99"]
            completed = runner(command)
            for workload in workloads:
                pattern = _PAIRED_BENCH_METRIC_PATTERN[workload]
                match = pattern.search(completed.stdout or "") if completed.returncode == 0 else None
                runs.append(LadderRun(
                    arm=names[0], workload=workload, round=r, position=position,
                    value=float(match.group(1)) if match else None, returncode=completed.returncode,
                ))
    for binary_dir, names in groups:
        if not set(names) & shared_arms:
            continue
        samples = {
            w: [x.value for x in runs if x.arm == names[0] and x.workload == w and x.value is not None]
            for w in workloads
        }
        if all(len(samples[w]) == rounds for w in workloads):
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = _cache_path(cache_dir, binary_dir, model, device_key, workloads, rounds_per_arm)
            path.write_text(json.dumps({"binary_dir": str(binary_dir), "arms": names, "samples": samples},
                                       indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = summarise_ladder(arms=arms, runs=runs, workloads=workloads)
    payload["cached_arms"] = sorted(cached)
    return payload


def summarise_ladder(
    *, arms: dict[str, Path], runs: list[LadderRun], workloads: tuple[str, ...]
) -> dict[str, object]:
    groups = _distinct_arms(arms)
    alias = {name: names[0] for names in groups.values() for name in names}
    metrics: dict[str, dict[str, object]] = {}
    for workload in workloads:
        metric = _PAIRED_BENCH_METRIC_NAME[workload]
        per_arm: dict[str, list[float]] = {}
        per_position: dict[int, list[float]] = {}
        failed = 0
        for run in runs:
            if run.workload != workload:
                continue
            if run.value is None:
                failed += 1
                continue
            per_arm.setdefault(run.arm, []).append(run.value)
            if run.position >= 0:
                per_position.setdefault(run.position, []).append(run.value)
        means = {name: statistics.fmean(per_arm[alias[name]]) for name in arms if per_arm.get(alias[name])}
        stock = means.get("stock")
        metrics[metric] = {
            "mean": means,
            "samples": {name: per_arm.get(alias[name], []) for name in arms},
            "pct_vs_stock": {
                name: (value / stock - 1.0) * 100.0 for name, value in means.items()
            } if stock else {},
            "position_mean": {str(k): statistics.fmean(v) for k, v in sorted(per_position.items())},
            "failed_runs": failed,
        }
    return {
        "schema": LADDER_SCHEMA,
        "reference_only": True,
        "arms": {name: str(path) for name, path in arms.items()},
        "shared_binaries": [names for names in groups.values() if len(names) > 1],
        "metrics": metrics,
    }


def write_ladder(run_dir: Path, payload: dict[str, object]) -> Path:
    path = run_dir / LADDER_ARTIFACT
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
