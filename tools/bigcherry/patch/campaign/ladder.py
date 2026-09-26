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
"""

from __future__ import annotations

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


def run_reference_ladder(
    *,
    arms: dict[str, Path],
    model: Path,
    runner: Runner,
    workloads: tuple[str, ...] = ("decode", "prefill"),
    rounds_per_arm: int = 1,
    exe: str = "",
) -> dict[str, object]:
    """Measure every distinct arm binary, rotated, and summarise per arm.

    One llama-bench invocation per (round, arm) measures every workload
    together (-p 512 -n 128 in one process), so the model loads once per
    invocation rather than once per workload; the ladder is reference-only,
    so one rotation per arm is the default."""
    groups = list(_distinct_arms(arms).items())
    n = len(groups)
    rounds = n * rounds_per_arm
    flags: list[str] = []
    for workload in ("prefill", "decode"):
        if workload in workloads:
            flags += list(_PAIRED_BENCH_WORKLOAD_FLAGS[workload])
    prompt = next((flags[i + 1] for i, f in enumerate(flags) if f == "-p" and flags[i + 1] != "0"), "0")
    gen = next((flags[i + 1] for i, f in enumerate(flags) if f == "-n" and flags[i + 1] != "0"), "0")
    runs: list[LadderRun] = []
    for r in range(rounds):
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
    return summarise_ladder(arms=arms, runs=runs, workloads=workloads)


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
