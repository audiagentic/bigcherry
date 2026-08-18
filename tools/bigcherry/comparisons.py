"""RE12-min (RV50-locked scope): identity-aware balanced pairwise
comparison planning AND real execution against descriptor-backed
runtime-bundle/binary/replay-cache artifacts.

BenchmarkArm identifies immutable ArtifactStore artifacts -- never a
caller-supplied executable Path as authority. The executable is obtained
only after rehydrating and verifying the arm's runtime-bundle artifact
(the same member-by-member re-hash lifecycle.py's own
_verify_runtime_bundle() already does). run_comparison() reuses
ab_benchmark's real subprocess/statistics primitives
(run_arm_capture/sanitize_environment/block_bootstrap_effect/
validate_replay_coverage) -- no second benchmark framework, per 6.3.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import ab_benchmark, provenance
from .artifacts import ArtifactRef, ArtifactStore


class ComparisonError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkArm:
    name: str
    runtime_bundle_artifact_id: str
    binary_artifact_id: str
    replay_cache_artifact_id: str | None
    source_slice_id: str
    build_plan_id: str
    effective_build_id: str
    workload_id: str
    environment: tuple[tuple[str, str], ...]
    device: str


@dataclass(frozen=True)
class ComparisonPlan:
    left: BenchmarkArm
    right: BenchmarkArm
    allowed_differences: frozenset[str]
    label: str


#: Every identity axis 6.2 requires preflight-checked before any GPU time.
_IDENTITY_FIELDS = (
    "runtime_bundle_artifact_id",
    "binary_artifact_id",
    "replay_cache_artifact_id",
    "source_slice_id",
    "build_plan_id",
    "effective_build_id",
    "workload_id",
    "environment",
    "device",
)


def plan_pair(
    left: BenchmarkArm,
    right: BenchmarkArm,
    *,
    label: str,
    allowed_differences: frozenset[str] = frozenset(),
) -> ComparisonPlan:
    """6.2 preflight: compare every identity axis between arms BEFORE any
    executable is even rehydrated, let alone run -- only differences named
    in ``allowed_differences`` may proceed."""
    differences = {field for field in _IDENTITY_FIELDS if getattr(left, field) != getattr(right, field)}
    undeclared = differences - set(allowed_differences)
    if undeclared:
        raise ComparisonError(
            f"comparison {label!r} has undeclared differences: {', '.join(sorted(undeclared))}"
        )
    return ComparisonPlan(left, right, allowed_differences, label)


def _rehydrate_arm(store: ArtifactStore, arm: BenchmarkArm) -> tuple[Path, ArtifactRef, ArtifactRef | None]:
    """6.1/6.2: rehydrate and re-verify every byte this arm claims to run
    -- a tampered runtime-bundle member is rejected here, before any
    subprocess is launched. Returns (entrypoint_path, binary_ref, cache_ref)."""
    bundle_ref = store.rehydrate(arm.runtime_bundle_artifact_id, expected_kind="runtime-bundle")
    binary_ref = store.rehydrate(arm.binary_artifact_id, expected_kind="binary")
    manifest = json.loads(bundle_ref.path.read_text(encoding="utf-8"))
    for member_name, member_hash in manifest["members"].items():
        member_relative = bundle_ref.path.parent.relative_to(store.root) / member_name
        if not store.verify(member_relative, member_hash):
            raise ComparisonError(
                f"arm {arm.name!r}: runtime-bundle member {member_name!r} failed store "
                f"verification -- refusing to run against a tampered or missing dependency"
            )
    entrypoint_path = bundle_ref.path.parent / manifest["entrypoint"]
    cache_ref = (
        store.rehydrate(arm.replay_cache_artifact_id, expected_kind="replay-cache")
        if arm.replay_cache_artifact_id else None
    )
    return entrypoint_path, binary_ref, cache_ref


def _publish_raw_evidence(
    store: ArtifactStore, output: Path, run: dict[str, Any], *, run_id: str, side: str, pair: int,
    doc: provenance.ProvenanceV2,
) -> dict[str, str]:
    """6.4: every arm's stdout/stderr/coverage becomes its own immutable
    ArtifactRef before the comparison report is assembled."""
    ids: dict[str, str] = {}
    for field in ("stdout", "stderr", "coverage"):
        name = run.get(field)
        if not name:
            continue
        source_path = output / name
        ref = store.publish_file_ref(
            f"runs/{run_id}/compare/{side}/pair-{pair}/{name}", source_path,
            kind="comparison-raw-evidence", provenance=doc)
        ids[field] = ref.artifact_id
    return ids


def run_comparison(
    plan: ComparisonPlan,
    *,
    store: ArtifactStore,
    run_id: str,
    model_args: list[str],
    output: Path,
    pairs: int = 3,
    metric_patterns: dict[str, re.Pattern[str]] | None = None,
    structured: bool = False,
    practical_threshold_pct: float = 1.0,
    resamples: int = 10_000,
    schedule_seed: int = 0,
    decision_grade: bool = False,
    campaign_plan_id: str | None = None,
    comparison_plan_id: str | None = None,
    project_revision: str = "",
    local_provenance_class: provenance.ProvenanceClass = "production",
) -> ArtifactRef:
    """6.3/6.4: run the balanced comparison for real (each arm identifies
    a real descriptor-backed runtime bundle/binary/optional replay cache),
    then publish ONE comparison-report artifact carrying every raw
    evidence artifact ID, effect estimates, and a validity/decision-grade
    verdict -- never a bare in-memory summary."""
    left_entrypoint, left_binary, left_cache = _rehydrate_arm(store, plan.left)
    right_entrypoint, right_binary, right_cache = _rehydrate_arm(store, plan.right)
    left_doc = provenance.ProvenanceV2.from_document(left_binary.provenance)
    right_doc = provenance.ProvenanceV2.from_document(right_binary.provenance)

    output.mkdir(parents=True, exist_ok=True)
    patterns = metric_patterns or {}
    runs: list[dict[str, Any]] = []
    raw_evidence: dict[str, dict[str, str]] = {"left": {}, "right": {}}
    issues: list[str] = []

    entrypoints = {"left": left_entrypoint, "right": right_entrypoint}
    caches = {"left": left_cache, "right": right_cache}
    arms = {"left": plan.left, "right": plan.right}
    binaries = {"left": left_binary, "right": right_binary}
    parent_docs = {"left": left_doc, "right": right_doc}

    def _run_doc(side: str) -> provenance.ProvenanceV2:
        parent = parent_docs[side]
        entries, parent_ids = provenance.lane_input_provenance({"binary": binaries[side]})
        return provenance.derive(
            parents=(parent,), parent_artifact_ids=parent_ids, project_revision=project_revision,
            source=parent.source,
            build=provenance.BuildProvenance(
                build_plan_id=parent.build.build_plan_id,
                effective_build_id=parent.build.effective_build_id, inputs=entries),
            workload=parent.workload, run_id=run_id, producer_stage="compare-run",
            campaign_plan_id=campaign_plan_id, comparison_plan_id=comparison_plan_id,
            local_class=local_provenance_class,
        )

    run_docs = {"left": _run_doc("left"), "right": _run_doc("right")}

    for pair in range(pairs):
        order = ("left", "right") if pair % 2 == 0 else ("right", "left")
        for side in order:
            arm = arms[side]
            cache_ref = caches[side]
            base_env = dict(os.environ)
            base_env.update(arm.environment)
            if arm.device:
                base_env["HIP_VISIBLE_DEVICES"] = arm.device
            coverage = output / f"pair-{pair + 1:03d}-{side}.coverage.json"
            env = ab_benchmark.sanitize_environment(
                base_env, "replay" if cache_ref else "native",
                cache=cache_ref.path if cache_ref else None, coverage=coverage)
            command = [str(entrypoints[side]), *model_args]
            run = ab_benchmark.run_arm_capture(
                command=command, cwd=None, output=output, pair=pair, side=side, env=env,
                patterns=patterns, structured=structured, replay=cache_ref is not None)
            runs.append(run)
            if run["returncode"] != 0:
                issues.append(f"pair {pair + 1} {side}: command exited {run['returncode']}")
            elif "metric_error" in run:
                issues.append(f"pair {pair + 1} {side}: {run['metric_error']}")
            ids = _publish_raw_evidence(
                store, output, run, run_id=run_id, side=side, pair=pair + 1, doc=run_docs[side])
            raw_evidence[side][f"pair-{pair + 1}"] = ids

    metric_names = sorted({name for run in runs for name in run.get("metrics", {})})
    effects: dict[str, dict[str, Any]] = {}
    for metric in metric_names:
        try:
            evidence = ab_benchmark.block_bootstrap_effect(
                runs, "right", "left", metric, seed=schedule_seed, resamples=resamples)
        except ValueError as exc:
            issues.append(f"metric {metric!r}: {exc}")
            continue
        evidence["decision"] = (
            "improved" if evidence["ci95_low_pct"] >= practical_threshold_pct else
            "regressed" if evidence["ci95_high_pct"] <= -practical_threshold_pct else
            "inconclusive"
        )
        effects[metric] = evidence

    valid = not issues
    resolved_decision_grade = decision_grade and valid

    entries, parent_ids = provenance.lane_input_provenance(
        {"left-binary": left_binary, "right-binary": right_binary})
    report_doc = provenance.derive(
        parents=(left_doc, right_doc), parent_artifact_ids=parent_ids,
        project_revision=project_revision,
        source=left_doc.source,
        build=provenance.BuildProvenance(
            build_plan_id=left_doc.build.build_plan_id,
            effective_build_id=left_doc.build.effective_build_id, inputs=entries),
        workload=left_doc.workload, run_id=run_id, producer_stage="compare",
        campaign_plan_id=campaign_plan_id, comparison_plan_id=comparison_plan_id,
        local_class=local_provenance_class,
    )

    def _arm_document(arm: BenchmarkArm) -> dict[str, object]:
        return {
            "name": arm.name,
            "runtime_bundle_artifact_id": arm.runtime_bundle_artifact_id,
            "binary_artifact_id": arm.binary_artifact_id,
            "replay_cache_artifact_id": arm.replay_cache_artifact_id,
            "source_slice_id": arm.source_slice_id,
            "build_plan_id": arm.build_plan_id,
            "effective_build_id": arm.effective_build_id,
            "workload_id": arm.workload_id,
            "environment": [list(item) for item in arm.environment],
            "device": arm.device,
        }

    report = {
        "label": plan.label,
        "comparison_plan_id": comparison_plan_id,
        "campaign_plan_id": campaign_plan_id,
        "campaign_run_id": run_id,
        "left_arm": _arm_document(plan.left),
        "right_arm": _arm_document(plan.right),
        "left_binary_artifact_id": left_binary.artifact_id,
        "right_binary_artifact_id": right_binary.artifact_id,
        "replay_arm": "right" if right_cache is not None else ("left" if left_cache is not None else None),
        "allowed_differences": sorted(plan.allowed_differences),
        "schedule_seed": schedule_seed,
        "pairs": pairs,
        "raw_evidence_artifact_ids": raw_evidence,
        "effects": effects,
        "issues": issues,
        "valid": valid,
        "decision_grade": resolved_decision_grade,
    }
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return store.publish_bytes_ref(
        f"runs/{run_id}/compare/report.json", report_bytes,
        kind="comparison-report", provenance=report_doc)
