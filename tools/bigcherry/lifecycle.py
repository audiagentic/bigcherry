"""RE10 (locked scope per RV50): the record -> inventory -> tune -> promote
-> replay-export -> replay-validate stage surface, on the same descriptor-
backed ArtifactStore/typed-provenance boundary RE25 built and RE09's
CampaignDatabaseIdentity schema-4 columns.

Deliberately ONE module, not the file-explosion the original design
proposed: PipelineService stays a small passive stage-boundary checker: no
graph planner, no second orchestrator. Each function here accepts a typed
spec plus ArtifactRefs/ArtifactLocators, rehydrates and verifies its
descriptor-backed inputs, invokes the EXISTING production implementation
(inventory.py, tune_promotion.py, replay_cache.py, ab_benchmark.py -- no
statistics/validation logic is reimplemented here), publishes outputs
through ArtifactStore descriptors, and returns a typed result carrying
ArtifactRefs.

Raw Path inputs are NOT specially handled here -- callers that want to feed
one through still go through the same lane-input resolution
campaign_lane.py's own _resolve_lane_inputs() uses (RE25.3); this module
only accepts already-resolved ArtifactRef/ArtifactLocator values, matching
the "descriptor IDs are the normal cross-run input" principle without
re-implementing raw-Path taint handling a second time.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import inventory as inventory_module
from . import paths, provenance, replay_cache, runtime_smoke, tune_promotion
from .ab_benchmark import validate_replay_coverage
from .artifacts import ArtifactLocator, ArtifactRef, ArtifactStore
from .context import ProjectContext
from .inventory import CampaignDatabaseIdentity


class LifecycleError(RuntimeError):
    pass


LaneInputValue = ArtifactRef | ArtifactLocator


def _rehydrate(
    store: ArtifactStore, value: LaneInputValue, *, expected_kind: str
) -> ArtifactRef:
    """The same descriptor-backed resolution campaign_lane.py's
    _resolve_lane_inputs() uses (RE25.3), narrowed to the two production
    input shapes this module accepts: an ArtifactLocator (rehydrate by
    ID) or an ArtifactRef that already carries a real artifact_id
    (re-verified against the store, never trusted as supplied). A caller
    holding only a raw ArtifactRef with no artifact_id (pre-descriptor
    legacy shape) is not a supported input here -- route it through
    campaign_lane.py's own raw-Path/legacy handling first.
    """
    if isinstance(value, ArtifactLocator):
        try:
            return store.rehydrate(value.artifact_id, expected_kind=expected_kind)
        except Exception as exc:
            raise LifecycleError(
                f"locator artifact_id {value.artifact_id!r} failed rehydration "
                f"(expected kind {expected_kind!r}): {exc}"
            ) from exc
    if isinstance(value, ArtifactRef) and value.artifact_id:
        rehydrated = _rehydrate(
            store, ArtifactLocator(value.artifact_id), expected_kind=expected_kind
        )
        if rehydrated.content_hash != value.content_hash:
            raise LifecycleError(
                f"supplied ArtifactRef content_hash {value.content_hash!r} disagrees "
                f"with the store's own record {rehydrated.content_hash!r} for "
                f"artifact_id {value.artifact_id!r}"
            )
        return rehydrated
    raise LifecycleError(
        f"lifecycle stage inputs must be descriptor-backed (ArtifactLocator or an "
        f"ArtifactRef with a real artifact_id), got {value!r}"
    )


def _verify_runtime_bundle(
    store: ArtifactStore, bundle_ref: ArtifactRef
) -> tuple[dict[str, Any], Path]:
    """Same verification make_smoke_worker() already does: every bundle
    member re-hashed against the store, not just the manifest JSON's own
    bytes. Returns (manifest, entrypoint_path)."""
    if not store.verify(
        bundle_ref.path.resolve().relative_to(store.root), bundle_ref.content_hash
    ):
        raise LifecycleError("runtime-bundle manifest failed store verification")
    manifest = json.loads(bundle_ref.path.read_text(encoding="utf-8"))
    for member_name, member_hash in manifest["members"].items():
        member_relative = bundle_ref.path.parent.relative_to(store.root) / member_name
        if not store.verify(member_relative, member_hash):
            raise LifecycleError(
                f"runtime-bundle member {member_name!r} failed store verification -- "
                f"refusing to run against a tampered or missing runtime dependency"
            )
    entrypoint_path = bundle_ref.path.parent / manifest["entrypoint"]
    return manifest, entrypoint_path


def _workload_spec_id(spec: runtime_smoke.RuntimeSmokeSpec) -> str:
    """A content-derived identity for the requested workload execution --
    what record-jsonl's provenance contract requires
    (workload.workload_spec_id), distinct from workload_id (the content
    hash of the resulting INVENTORY, established later, once one exists).
    """
    payload = {
        "model_hash": ArtifactStore.digest(spec.model_path.read_bytes()),
        "n_prompt": spec.n_prompt,
        "n_gen": spec.n_gen,
        "repetitions": spec.repetitions,
        "n_gpu_layers": spec.n_gpu_layers,
        "split_mode": spec.split_mode,
        "tensor_split": list(spec.tensor_split),
        "environment": list(spec.environment),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return ArtifactStore.digest(encoded)


def _parent_doc(ref: ArtifactRef) -> provenance.ProvenanceV2:
    try:
        return provenance.ProvenanceV2.from_document(ref.provenance)
    except provenance.ProvenanceError as exc:
        raise LifecycleError(
            f"parent artifact provenance is not a schema-v2 document: {exc}"
        ) from exc


def _identity_from_provenance(
    doc: provenance.ProvenanceV2, *, run_id: str
) -> CampaignDatabaseIdentity | None:
    """RE09/RE10: a real production caller derives its CampaignDatabaseIdentity
    from the ACTUAL runtime-bundle provenance it just verified, not from a
    separately caller-supplied value -- an imported-legacy/development
    runtime bundle has no complete real identity to assert, so this
    returns None (diagnostic scope) exactly when the source data doesn't
    support more.

    RE09 audit fix: imported-legacy is rejected FIRST. RE25.3's downgraded-
    Ref branch keeps an unverified doc's source/build fields, so an
    imported-legacy document CAN carry a complete triple -- trusting those
    fields would write a 'campaign'-scoped DB row from identity we explicitly
    do not trust (RE09: 'a partial identity is not campaign evidence' -- by
    the same rule, an UNVERIFIED one isn't either). Only imported-legacy is
    rejected: production/development/diagnostic documents were generated by
    BigCherry itself, so their fields are genuine and a complete triple on
    any of them IS campaign evidence (RE09's scope rule is completeness-
    based; the DB row carries no class, and release promotion stays gated
    separately by require_promotable)."""
    if doc.project.provenance_class == "imported-legacy":
        return None
    source_slice_id = doc.source.source_slice_id
    build_plan_id = doc.build.build_plan_id
    effective_build_id = doc.build.effective_build_id
    if not (source_slice_id and build_plan_id and effective_build_id):
        return None
    return CampaignDatabaseIdentity(
        source_slice_id=source_slice_id,
        build_plan_id=build_plan_id,
        effective_build_id=effective_build_id,
        campaign_run_id=run_id,
        workload_id=doc.workload.workload_id,
    )


# --------------------------------------------------------------------- record


@dataclass(frozen=True)
class RecordStageResult:
    record_ref: ArtifactRef


def execute_record_stage(
    *,
    context: ProjectContext,
    store: ArtifactStore,
    run_id: str,
    runtime_bundle: LaneInputValue,
    spec: runtime_smoke.RuntimeSmokeSpec,
    environment: dict[str, str] | None = None,
    project_revision: str = "",
    local_provenance_class: provenance.ProvenanceClass = "production",
) -> RecordStageResult:
    """Runs the ALREADY-BUILT record runtime bundle for real, with
    GGML_HIP_DISPATCH_DB pointed at a run-scoped path, and publishes the
    resulting JSONL. Never touches a build directory -- only the
    immutable, verified runtime bundle."""
    bundle_ref = _rehydrate(store, runtime_bundle, expected_kind="runtime-bundle")
    _manifest, entrypoint_path = _verify_runtime_bundle(store, bundle_ref)

    stage_root = context.work_root / "runs" / run_id / "record"
    stage_root.mkdir(parents=True, exist_ok=True)
    record_path = stage_root / "record.jsonl"

    env = dict(os.environ)
    env.update(environment or {})
    # RE15 real-hardware finding: GGML_HIP_DISPATCH_MODE defaults to
    # "native" (ggml_hip_parse_mode()) when unset -- without this, the
    # record build's dispatch layer never enters GGML_HIP_DISPATCH_MODE_
    # RECORD and silently records nothing, no matter what
    # GGML_HIP_DISPATCH_DB points at.
    env["GGML_HIP_DISPATCH_MODE"] = "record"
    env["GGML_HIP_DISPATCH_DB"] = str(record_path)
    argv = runtime_smoke.smoke_argv(entrypoint_path, spec)
    completed = subprocess.run(argv, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        raise LifecycleError(
            f"record run exited {completed.returncode}: {completed.stderr[-2000:]}"
        )

    try:
        inventory_module.read_jsonl(record_path)
    except inventory_module.RecordError as exc:
        raise LifecycleError(
            f"record run produced an invalid record file: {exc}"
        ) from exc

    bundle_doc = _parent_doc(bundle_ref)
    entries, parent_ids = provenance.lane_input_provenance(
        {"runtime-bundle": bundle_ref}
    )
    doc = provenance.derive(
        parents=(bundle_doc,),
        parent_artifact_ids=parent_ids,
        project_revision=project_revision,
        source=bundle_doc.source,
        build=provenance.BuildProvenance(
            build_plan_id=bundle_doc.build.build_plan_id,
            effective_build_id=bundle_doc.build.effective_build_id,
            inputs=entries,
        ),
        workload=provenance.WorkloadProvenance(
            workload_spec_id=_workload_spec_id(spec)
        ),
        run_id=run_id,
        producer_stage="record",
        local_class=local_provenance_class,
    )
    record_ref = store.publish_file_ref(
        f"runs/{run_id}/record/record.jsonl",
        record_path,
        kind="record-jsonl",
        provenance=doc,
    )
    return RecordStageResult(record_ref=record_ref)


# ------------------------------------------------------------------ inventory


@dataclass(frozen=True)
class InventoryStageResult:
    inventory_ref: ArtifactRef
    database_ref: ArtifactRef
    workload_id: str


def execute_inventory_stage(
    *,
    context: ProjectContext,
    store: ArtifactStore,
    run_id: str,
    record: LaneInputValue,
    project_revision: str = "",
    local_provenance_class: provenance.ProvenanceClass = "production",
) -> InventoryStageResult:
    record_ref = _rehydrate(store, record, expected_kind="record-jsonl")
    record_doc = _parent_doc(record_ref)

    rec = inventory_module.read_jsonl(record_ref.path)
    inventory_doc = inventory_module.build_inventory(rec)
    inventory_bytes = (json.dumps(inventory_doc, indent=2) + "\n").encode("utf-8")
    workload_id = ArtifactStore.digest(inventory_bytes)

    stage_root = context.work_root / "runs" / run_id / "inventory"
    stage_root.mkdir(parents=True, exist_ok=True)
    db_path = stage_root / "dispatch.sqlite"
    identity = _identity_from_provenance(record_doc, run_id=run_id)
    if identity is not None:
        identity = dataclasses.replace(identity, workload_id=workload_id)
    inventory_module.build_database(
        rec, db_path, paths.SQL / "dispatch-db.sql", identity=identity
    )

    entries, parent_ids = provenance.lane_input_provenance({"record-jsonl": record_ref})
    doc = provenance.derive(
        parents=(record_doc,),
        parent_artifact_ids=parent_ids,
        project_revision=project_revision,
        source=record_doc.source,
        build=provenance.BuildProvenance(
            build_plan_id=record_doc.build.build_plan_id,
            effective_build_id=record_doc.build.effective_build_id,
            inputs=entries,
        ),
        workload=provenance.WorkloadProvenance(workload_id=workload_id),
        run_id=run_id,
        producer_stage="inventory",
        local_class=local_provenance_class,
    )
    inventory_ref = store.publish_bytes_ref(
        f"runs/{run_id}/inventory/inventory.json",
        inventory_bytes,
        kind="inventory",
        provenance=doc,
    )
    database_ref = store.publish_file_ref(
        f"runs/{run_id}/inventory/dispatch.sqlite",
        db_path,
        kind="dispatch-db",
        provenance=doc,
    )
    return InventoryStageResult(
        inventory_ref=inventory_ref, database_ref=database_ref, workload_id=workload_id
    )


# ----------------------------------------------------------------------- tune


@dataclass(frozen=True)
class TuneStageResult:
    measurements_ref: ArtifactRef
    database_ref: ArtifactRef


def execute_tune_stage(
    *,
    context: ProjectContext,
    store: ArtifactStore,
    run_id: str,
    runtime_bundle: LaneInputValue,
    dispatch_db: LaneInputValue,
    spec: runtime_smoke.RuntimeSmokeSpec,
    environment: dict[str, str] | None = None,
    project_revision: str = "",
    local_provenance_class: provenance.ProvenanceClass = "production",
) -> TuneStageResult:
    """The tune build's candidate catalog is already baked in at compile
    time (GGML_HIP_AUTOTUNE_SIGNATURE_FILE is a cmake-configure option,
    not a runtime env var) -- this stage only RUNS the already-built tune
    runtime bundle and captures its measurements; it does not re-consume
    the inventory that selected that catalog."""
    bundle_ref = _rehydrate(store, runtime_bundle, expected_kind="runtime-bundle")
    _manifest, entrypoint_path = _verify_runtime_bundle(store, bundle_ref)
    db_ref = _rehydrate(store, dispatch_db, expected_kind="dispatch-db")

    stage_root = context.work_root / "runs" / run_id / "tune"
    stage_root.mkdir(parents=True, exist_ok=True)
    # RE09 4.5: never write into an ArtifactStore object directly -- copy
    # the immutable published DB into this run's own workspace first.
    working_db = stage_root / "dispatch.sqlite"
    working_db.write_bytes(db_ref.path.read_bytes())

    measurements_base = stage_root / "tune"
    env = dict(os.environ)
    env.update(environment or {})
    # RE15 real-hardware finding (same root cause as execute_record_stage):
    # GGML_HIP_DISPATCH_MODE defaults to "native" when unset, so the tune
    # build's dispatch layer never enters GGML_HIP_DISPATCH_MODE_TUNE
    # without this.
    env["GGML_HIP_DISPATCH_MODE"] = "tune"
    # RE15 real-hardware finding: HIP graph capture/replay is keyed on
    # tensor shapes/pointers, not on which candidate kernel the tune
    # dispatcher picked -- with graphs on, a captured launch from one
    # candidate gets replayed against a later call sized for a DIFFERENT
    # candidate at the same shape, which segfaulted (illegal memory
    # access) at small ubatch sizes. Confirmed required by
    # docs/reference/TEST.md's own tuning instructions. Tune-only: the
    # replay/production path never sets this, so shipped inference keeps
    # graph-launch performance.
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    env["GGML_HIP_DISPATCH_DB"] = str(measurements_base)
    argv = runtime_smoke.smoke_argv(entrypoint_path, spec)
    completed = subprocess.run(argv, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        raise LifecycleError(
            f"tune run exited {completed.returncode}: {completed.stderr[-2000:]}"
        )

    measurements_path = Path(str(measurements_base) + ".measurements.jsonl")
    if not measurements_path.is_file():
        raise LifecycleError(f"tune run did not produce {measurements_path}")

    bundle_doc = _parent_doc(bundle_ref)
    db_doc = _parent_doc(db_ref)
    identity = _identity_from_provenance(bundle_doc, run_id=run_id)
    inventory_module.load_measurements(
        measurements_path, working_db, paths.SQL / "dispatch-db.sql", identity=identity
    )

    entries, parent_ids = provenance.lane_input_provenance(
        {"runtime-bundle": bundle_ref, "dispatch-db": db_ref}
    )
    doc = provenance.derive(
        parents=(bundle_doc, db_doc),
        parent_artifact_ids=parent_ids,
        project_revision=project_revision,
        source=bundle_doc.source,
        build=provenance.BuildProvenance(
            build_plan_id=bundle_doc.build.build_plan_id,
            effective_build_id=bundle_doc.build.effective_build_id,
            inputs=entries,
        ),
        workload=db_doc.workload,
        run_id=run_id,
        producer_stage="tune",
        local_class=local_provenance_class,
    )
    measurements_ref = store.publish_file_ref(
        f"runs/{run_id}/tune/measurements.jsonl",
        measurements_path,
        kind="tuning-measurements",
        provenance=doc,
    )
    database_ref = store.publish_file_ref(
        f"runs/{run_id}/tune/dispatch.sqlite",
        working_db,
        kind="dispatch-db",
        provenance=doc,
    )
    return TuneStageResult(measurements_ref=measurements_ref, database_ref=database_ref)


# ------------------------------------------------------------------ promotion


@dataclass(frozen=True)
class PromotionStageResult:
    promoted_winners_ref: ArtifactRef


def execute_promotion_stage(
    *,
    context: ProjectContext,
    store: ArtifactStore,
    run_id: str,
    measurements: LaneInputValue,
    q: float = 0.05,
    threshold_pct: float = 1.0,
    resamples: int = 10_000,
    project_revision: str = "",
    local_provenance_class: provenance.ProvenanceClass = "production",
) -> PromotionStageResult:
    """Delegates to the EXISTING tune_promotion.promote() implementation
    -- it already owns evidence validation including the noise-canary
    protocol; nothing about promotion statistics is reimplemented here."""
    measurements_ref = _rehydrate(
        store, measurements, expected_kind="tuning-measurements"
    )
    measurements_doc = _parent_doc(measurements_ref)

    stage_root = context.work_root / "runs" / run_id / "promote"
    stage_root.mkdir(parents=True, exist_ok=True)
    output_path = stage_root / "promoted-winners.jsonl"
    try:
        tune_promotion.promote(
            measurements_ref.path,
            output_path,
            q=q,
            threshold_pct=threshold_pct,
            resamples=resamples,
        )
    except tune_promotion.PromotionError as exc:
        raise LifecycleError(
            f"promotion rejected this measurements evidence: {exc}"
        ) from exc

    entries, parent_ids = provenance.lane_input_provenance(
        {"tuning-measurements": measurements_ref}
    )
    doc = provenance.derive(
        parents=(measurements_doc,),
        parent_artifact_ids=parent_ids,
        project_revision=project_revision,
        source=measurements_doc.source,
        build=provenance.BuildProvenance(
            build_plan_id=measurements_doc.build.build_plan_id,
            effective_build_id=measurements_doc.build.effective_build_id,
            inputs=entries,
        ),
        workload=measurements_doc.workload,
        run_id=run_id,
        producer_stage="tune-promote",
        local_class=local_provenance_class,
    )
    promoted_winners_ref = store.publish_file_ref(
        f"runs/{run_id}/promote/promoted-winners.jsonl",
        output_path,
        kind="promoted-winners",
        provenance=doc,
    )
    return PromotionStageResult(promoted_winners_ref=promoted_winners_ref)


# -------------------------------------------------------------- replay export


@dataclass(frozen=True)
class ReplayExportStageResult:
    replay_cache_ref: ArtifactRef


def execute_replay_export_stage(
    *,
    context: ProjectContext,
    store: ArtifactStore,
    run_id: str,
    promoted_winners: LaneInputValue,
    manifest: LaneInputValue,
    source_root: Path,
    generation: int = 0,
    keep_generations: int = 3,
    project_revision: str = "",
    local_provenance_class: provenance.ProvenanceClass = "production",
) -> ReplayExportStageResult:
    """Delegates to replay_cache.build() -- keeps the portable replay
    dispatch identity independent of campaign DB identity (RE09 4.6):
    this stage's own provenance envelope carries campaign identity, but
    the exported bytes themselves never do."""
    winners_ref = _rehydrate(store, promoted_winners, expected_kind="promoted-winners")
    manifest_ref = _rehydrate(store, manifest, expected_kind="manifest")
    winners_doc = _parent_doc(winners_ref)
    manifest_doc = _parent_doc(manifest_ref)

    ggml_h = source_root / "ggml" / "include" / "ggml.h"
    if not ggml_h.is_file():
        raise LifecycleError(
            f"replay export needs a real source tree: {ggml_h} does not exist"
        )

    try:
        cache_bytes = replay_cache.build(
            winners_ref.path,
            manifest_ref.path,
            ggml_h,
            keep_generations=keep_generations,
            generation=generation,
        )
    except SystemExit as exc:
        raise LifecycleError(f"replay export refused: {exc}") from exc

    entries, parent_ids = provenance.lane_input_provenance(
        {"promoted-winners": winners_ref, "manifest": manifest_ref}
    )
    doc = provenance.derive(
        parents=(winners_doc, manifest_doc),
        parent_artifact_ids=parent_ids,
        project_revision=project_revision,
        source=winners_doc.source,
        build=provenance.BuildProvenance(
            build_plan_id=winners_doc.build.build_plan_id,
            effective_build_id=winners_doc.build.effective_build_id,
            inputs=entries,
        ),
        workload=winners_doc.workload,
        run_id=run_id,
        producer_stage="replay-export",
        local_class=local_provenance_class,
    )
    replay_cache_ref = store.publish_bytes_ref(
        f"runs/{run_id}/replay/replay.cache",
        cache_bytes,
        kind="replay-cache",
        provenance=doc,
    )
    return ReplayExportStageResult(replay_cache_ref=replay_cache_ref)


# ------------------------------------------------------------ replay validate


@dataclass(frozen=True)
class ReplayValidationStageResult:
    coverage_ref: ArtifactRef
    coverage: dict[str, Any]


def execute_replay_validation_stage(
    *,
    context: ProjectContext,
    store: ArtifactStore,
    run_id: str,
    runtime_bundle: LaneInputValue,
    replay_cache_artifact: LaneInputValue,
    spec: runtime_smoke.RuntimeSmokeSpec,
    environment: dict[str, str] | None = None,
    project_revision: str = "",
    local_provenance_class: provenance.ProvenanceClass = "production",
) -> ReplayValidationStageResult:
    """Runs the replay runtime bundle with GGML_HIP_DISPATCH_MODE=replay
    against the verified cache, then reuses ab_benchmark.validate_replay_coverage()
    -- no second coverage validator."""
    bundle_ref = _rehydrate(store, runtime_bundle, expected_kind="runtime-bundle")
    _manifest, entrypoint_path = _verify_runtime_bundle(store, bundle_ref)
    cache_ref = _rehydrate(store, replay_cache_artifact, expected_kind="replay-cache")

    stage_root = context.work_root / "runs" / run_id / "replay-validate"
    stage_root.mkdir(parents=True, exist_ok=True)
    coverage_path = stage_root / "coverage.json"

    env = dict(os.environ)
    env.update(environment or {})
    env["GGML_HIP_DISPATCH_MODE"] = "replay"
    env["GGML_HIP_DISPATCH_CACHE"] = str(cache_ref.path)
    env["GGML_HIP_DISPATCH_COVERAGE"] = str(coverage_path)
    env["GGML_HIP_DISPATCH_MISS"] = "native-record"
    argv = runtime_smoke.smoke_argv(entrypoint_path, spec)
    completed = subprocess.run(argv, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        raise LifecycleError(
            f"replay validation run exited {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )

    coverage = validate_replay_coverage(coverage_path)

    bundle_doc = _parent_doc(bundle_ref)
    cache_doc = _parent_doc(cache_ref)
    entries, parent_ids = provenance.lane_input_provenance(
        {"runtime-bundle": bundle_ref, "replay-cache": cache_ref}
    )
    doc = provenance.derive(
        parents=(bundle_doc, cache_doc),
        parent_artifact_ids=parent_ids,
        project_revision=project_revision,
        source=bundle_doc.source,
        build=provenance.BuildProvenance(
            build_plan_id=bundle_doc.build.build_plan_id,
            effective_build_id=bundle_doc.build.effective_build_id,
            inputs=entries,
        ),
        workload=bundle_doc.workload,
        run_id=run_id,
        producer_stage="replay-validate",
        local_class=local_provenance_class,
    )
    coverage_ref = store.publish_file_ref(
        f"runs/{run_id}/replay-validate/coverage.json",
        coverage_path,
        kind="replay-coverage",
        provenance=doc,
    )
    return ReplayValidationStageResult(coverage_ref=coverage_ref, coverage=coverage)
