"""Identity-aware balanced pairwise comparison planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ComparisonError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkArm:
    name: str
    binary_hash: str
    executable: Path
    environment: tuple[tuple[str, str], ...]
    cache_hash: str | None
    source_slice_id: str
    build_id: str
    workload_id: str


@dataclass(frozen=True)
class ComparisonPlan:
    left: BenchmarkArm
    right: BenchmarkArm
    allowed_differences: frozenset[str]
    label: str


def plan_pair(
    left: BenchmarkArm,
    right: BenchmarkArm,
    *,
    label: str,
    allowed_differences: frozenset[str] = frozenset(),
) -> ComparisonPlan:
    differences: set[str] = set()
    if left.binary_hash != right.binary_hash:
        differences.add("binary_hash")
    if left.environment != right.environment:
        differences.add("environment")
    if left.cache_hash != right.cache_hash:
        differences.add("cache_hash")
    if left.source_slice_id != right.source_slice_id:
        differences.add("source_slice_id")
    if left.build_id != right.build_id:
        differences.add("build_id")
    if left.workload_id != right.workload_id:
        differences.add("workload_id")
    undeclared = differences - set(allowed_differences)
    if undeclared:
        raise ComparisonError(
            f"comparison {label!r} has undeclared differences: {', '.join(sorted(undeclared))}"
        )
    return ComparisonPlan(left, right, allowed_differences, label)


def report_row(plan: ComparisonPlan, *, validity: str, metric: float | None,
               delta: float | None, confidence_interval: tuple[float, float] | None = None) -> dict[str, object]:
    return {
        "label": plan.label,
        "left": {"name": plan.left.name, "binary_hash": plan.left.binary_hash,
                  "source_slice_id": plan.left.source_slice_id, "build_id": plan.left.build_id,
                  "workload_id": plan.left.workload_id},
        "right": {"name": plan.right.name, "binary_hash": plan.right.binary_hash,
                   "source_slice_id": plan.right.source_slice_id, "build_id": plan.right.build_id,
                   "workload_id": plan.right.workload_id},
        "allowed_differences": sorted(plan.allowed_differences),
        "validity": validity,
        "metric": metric,
        "delta": delta,
        "confidence_interval": confidence_interval,
    }
