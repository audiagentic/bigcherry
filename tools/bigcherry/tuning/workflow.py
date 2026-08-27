"""HI130: the full record -> tune -> correctness-evidence -> promote ->
replay pipeline as one repeatable, deterministic call, so it can be kicked
off as a standard bigcherry workflow instead of an operator/agent manually
chaining ~12 commands and working around gotchas live (validated by hand
on real Brutus GPU hardware this session -- dual-XTX + single-GPU R9700,
Qwen3.8-27B-Q8_0 MTP, 20/39 candidates genuinely promoted with real
correctness evidence).

Calls existing campaign/tuning APIs directly -- never shells out to the
``bigcherry`` CLI internally. Every stage that touches a GPU launches a
real server via server_runner.ServerRunner and drives it with a real
request; there is no mock/simulation mode.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import inventory as inv_mod
from . import replay as replay_mod
from . import tune_promotion
from .server_runner import ServerRunner
from .. import hi80_generate_correctness_evidence as hi80
from ..campaign import planner as campaign_planner
from ..campaign.lane import CampaignLaneResult
from ..core import config as campaign_config
from ..core import gpu as gpu_mod
from ..core import paths
from ..core.artifacts import ArtifactStore
from ..core.context import ProjectContext
from ..source.identity import atomic_write_json


class TuneCampaignError(RuntimeError):
    pass


@dataclass(frozen=True)
class StageIdentity:
    run_id: str
    source_slice_id: str
    build_plan_id: str
    manifest_path: str | None
    source_root: str


@dataclass(frozen=True)
class WorkflowReceipt:
    schema_version: int
    campaign_run_id: str
    model_path: str
    platform_name: str
    devices: str
    runtime_profile_name: str
    effective_tune_context: int
    record: StageIdentity
    tune: StageIdentity
    correctness: StageIdentity | None
    replay: StageIdentity | None
    promoted_before_evidence: int
    promoted_after_evidence: int
    dispatch_cache_path: str | None
    replay_coverage: dict | None
    started_at: str
    finished_at: str


def _stage_identity(result: CampaignLaneResult) -> StageIdentity:
    return StageIdentity(
        # gpt review (2026-08-27): run_campaign()'s per-lane run_id
        # (_lane_run_id(campaign_run_id, lane)) is what actually got used
        # for this lane's ArtifactStore paths/provenance -- reconstructing
        # a different string here would make the receipt's run_id
        # unusable for re-deriving those paths even though the build
        # itself succeeded.
        run_id=result.run_id,
        source_slice_id=result.source_slice_id,
        build_plan_id=result.build_plan_id,
        manifest_path=str(result.manifest_ref.path) if result.manifest_ref else None,
        source_root=str(result.source_root),
    )


def _devices_tuple(devices: str) -> tuple[int, ...]:
    return tuple(int(d) for d in devices.split(","))


def _plan_and_run_one_lane(
    *, context: ProjectContext, cfg: campaign_config.Config, store: ArtifactStore,
    source_name: str, build_name: str, platform_name: str, run_id: str,
    binary_relative_path: str,
    inputs_by_build: dict[str, tuple[tuple[str, object], ...]] | None = None,
    experiment: str | None = None,
) -> CampaignLaneResult:
    request = campaign_planner.CampaignRequest(
        selectors=(campaign_config.CampaignLaneSelector(
            source=source_name, build=build_name, platform=platform_name,
        ),),
        inputs_by_build=inputs_by_build or {},
        binary_relative_path=binary_relative_path,
        experiment=experiment,
    )
    lanes = campaign_planner.plan(request, cfg)
    results = campaign_planner.run_campaign(
        lanes, cfg=cfg, context=context, store=store, run_id=run_id,
    )
    (result,) = results.values()
    if isinstance(result, Exception):
        raise TuneCampaignError(
            f"{source_name}:{build_name}:{platform_name} build failed: {result}"
        ) from result
    return result


def _stage_record(
    *, context, cfg, store, run_id, platform_name, source_name,
    model_path: Path, devices: str, runtime_profile: campaign_config.RuntimeProfile,
    workdir: Path,
) -> tuple[CampaignLaneResult, Path]:
    lane_result = _plan_and_run_one_lane(
        context=context, cfg=cfg, store=store, source_name=source_name,
        build_name="record", platform_name=platform_name, run_id=run_id,
        binary_relative_path="bin/llama-server",
    )
    binary_path = lane_result.binary_ref.path
    record_db_path = workdir / "record"
    runner = ServerRunner(
        binary=binary_path, model=model_path,
        extra_args=("-ngl", "99", "-c", str(runtime_profile.production_context), *runtime_profile.server_args),
        env_overrides={
            "HIP_VISIBLE_DEVICES": devices,
            "GGML_HIP_DISPATCH_MODE": "record",
            "GGML_HIP_DISPATCH_DB": str(record_db_path),
            "GGML_CUDA_DISABLE_GRAPHS": "1",
        },
        log_path=workdir / "record-server.log",
    )
    with runner:
        runner.run_completion("Describe the water cycle in two sentences.", n_predict=96)
    actual_record_path = record_db_path  # the binary writes this exact path, no suffix
    if not actual_record_path.is_file():
        raise TuneCampaignError(f"record stage produced no output at {actual_record_path}")
    return lane_result, actual_record_path


def _stage_inventory_record(*, record_db_path: Path, workdir: Path) -> tuple[Path, Path]:
    record = inv_mod.read_jsonl(record_db_path)
    inventory = inv_mod.build_inventory(record)
    inventory_path = workdir / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8", newline="")
    inventory_db_path = workdir / "inventory.sqlite"
    inv_mod.build_database(record, inventory_db_path, paths.SQL / "dispatch-db.sql")
    return inventory_path, inventory_db_path


def _stage_tune(
    *, context, cfg, store, run_id, platform_name, source_name,
    inventory_path: Path, model_path: Path, devices: str,
    runtime_profile: campaign_config.RuntimeProfile,
    screen_samples: int, final_samples: int, workdir: Path,
) -> tuple[CampaignLaneResult, Path]:
    gpu_mod.preflight_context(profile=runtime_profile, devices=_devices_tuple(devices), stage="tune")
    lane_result = _plan_and_run_one_lane(
        context=context, cfg=cfg, store=store, source_name=source_name,
        build_name="tune", platform_name=platform_name, run_id=run_id,
        binary_relative_path="bin/llama-server",
        inputs_by_build={"tune": (("inventory", inventory_path),)},
    )
    binary_path = lane_result.binary_ref.path
    tune_db_path = workdir / "tune"
    runner = ServerRunner(
        binary=binary_path, model=model_path,
        extra_args=("-ngl", "99", "-c", str(runtime_profile.tune_context), *runtime_profile.server_args),
        env_overrides={
            "HIP_VISIBLE_DEVICES": devices,
            "GGML_HIP_DISPATCH_MODE": "tune",
            "GGML_HIP_DISPATCH_DB": str(tune_db_path),
            "GGML_HIP_TUNE_SCREEN_SAMPLES": str(screen_samples),
            "GGML_HIP_TUNE_FINAL_SAMPLES": str(final_samples),
            "GGML_CUDA_DISABLE_GRAPHS": "1",
        },
        log_path=workdir / "tune-server.log",
    )
    with runner:
        runner.run_completion("Write a short paragraph about the ocean.", n_predict=96)
    measurements_path = Path(f"{tune_db_path}.measurements.jsonl")
    if not measurements_path.is_file():
        raise TuneCampaignError(f"tune stage produced no measurements at {measurements_path}")
    return lane_result, measurements_path


def _count_missing_correctness_evidence(promoted_path: Path) -> int:
    """How many rows are stuck on `rejected_no_correctness_evidence` --
    gpt review (2026-08-27): `promote()`'s own return dict has no such
    count, and `promoted == 0` is the wrong trigger for re-running the
    correctness-evidence stage -- a dispatch_db that already had SOME
    reusable evidence from a prior run can promote a few candidates
    immediately while leaving others stuck, and that partial case must
    still trigger evidence generation for the rest, not be silently
    skipped because promoted > 0."""
    count = 0
    with promoted_path.open(encoding="utf-8") as f:
        next(f, None)  # header line
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("promotion_status") == "rejected_no_correctness_evidence":
                count += 1
    return count


def _stage_load_and_promote(
    *, tune_measurements: Path, tune_manifest_path: Path | None, workdir: Path,
    q: float, threshold_pct: float, resamples: int,
) -> tuple[Path, dict, int, int]:
    dispatch_db = workdir / "tune.sqlite"
    inv_mod.load_measurements(
        tune_measurements, dispatch_db, paths.SQL / "dispatch-db.sql",
        manifest_path=tune_manifest_path,
    )
    promoted_path = workdir / "promoted.jsonl"
    result = tune_promotion.promote(
        tune_measurements, promoted_path,
        dispatch_db=dispatch_db, q=q, threshold_pct=threshold_pct, resamples=resamples,
    )
    missing_evidence = _count_missing_correctness_evidence(promoted_path)
    return dispatch_db, result, int(result.get("promoted", 0)), missing_evidence


def _stage_correctness_evidence(
    *, context, cfg, store, run_id, platform_name, source_name,
    inventory_path: Path, tune_measurements: Path, dispatch_db: Path,
    seeds: tuple[int, ...],
) -> CampaignLaneResult:
    lane_result = _plan_and_run_one_lane(
        context=context, cfg=cfg, store=store, source_name=source_name,
        build_name="tune", platform_name=platform_name, run_id=run_id,
        binary_relative_path="bin/test-backend-ops",
        inputs_by_build={"tune": (("inventory", inventory_path),)},
        experiment="hi105-correctness",
    )
    _header, results = hi80._read_measurements(tune_measurements)
    rows = hi80.find_candidate_rows(results)
    import sqlite3
    conn = sqlite3.connect(str(dispatch_db))
    try:
        for row in rows:
            try:
                hi80.generate_for_row(
                    conn, row,
                    binary=lane_result.binary_ref.path,
                    vendor_root=lane_result.source_root,
                    seeds=seeds,
                    headroom_fraction=hi80.ce.DEFAULT_HEADROOM_FRACTION,
                    contract_version=hi80.ce.CONTRACT_VERSION,
                    tool_version="hi130-tune-campaign-v1",
                )
            except hi80.scm.SignatureMappingError:
                continue  # unsupported signature domain -- same as the CLI's own honest skip
    finally:
        conn.close()
    return lane_result


def _stage_replay_export(
    *, promoted_path: Path, tune_manifest_path: Path, tune_source_root: Path,
    dispatch_db: Path, workdir: Path,
) -> Path:
    ggml_h = tune_source_root / "ggml" / "include" / "ggml.h"
    cache_bytes = replay_mod.build(
        promoted_path, tune_manifest_path, ggml_h, dispatch_db=dispatch_db,
    )
    cache_path = workdir / "dispatch.cache"
    cache_path.write_bytes(cache_bytes)
    return cache_path


def _stage_replay_build_and_verify(
    *, context, cfg, store, run_id, platform_name, source_name,
    inventory_path: Path, winners_path: Path, model_path: Path, devices: str,
    runtime_profile: campaign_config.RuntimeProfile, dispatch_cache: Path, workdir: Path,
) -> tuple[CampaignLaneResult, dict]:
    lane_result = _plan_and_run_one_lane(
        context=context, cfg=cfg, store=store, source_name=source_name,
        build_name="replay", platform_name=platform_name, run_id=run_id,
        binary_relative_path="bin/llama-server",
        inputs_by_build={
            "replay": (("inventory", inventory_path), ("promoted-winners", winners_path)),
        },
    )
    binary_path = lane_result.binary_ref.path
    coverage_path = workdir / "coverage.json"
    runner = ServerRunner(
        binary=binary_path, model=model_path,
        extra_args=("-ngl", "99", "-c", str(runtime_profile.production_context), *runtime_profile.server_args),
        env_overrides={
            "HIP_VISIBLE_DEVICES": devices,
            "GGML_HIP_DISPATCH_MODE": "replay",
            "GGML_HIP_DISPATCH_CACHE": str(dispatch_cache),
            "GGML_HIP_DISPATCH_COVERAGE": str(coverage_path),
        },
        log_path=workdir / "replay-server.log",
    )
    with runner:
        runner.run_completion("Explain how a compass works.", n_predict=96)
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.is_file() else {}
    return lane_result, coverage


def run_tune_campaign(
    *,
    context: ProjectContext,
    cfg: campaign_config.Config,
    store: ArtifactStore,
    model_path: Path,
    platform_name: str,
    devices: str,
    runtime_profile_name: str,
    source_name: str = "bigcherry",
    run_id: str | None = None,
    workdir: Path | None = None,
    tune_screen_samples: int = 3,
    tune_final_samples: int = 15,
    correctness_seeds: tuple[int, ...] = (1, 2, 3),
    promotion_q: float = 0.05,
    promotion_threshold_pct: float = 1.0,
    promotion_resamples: int = 10_000,
) -> WorkflowReceipt:
    import uuid

    profile = cfg.runtime_profiles.get(runtime_profile_name)
    if profile is None:
        raise TuneCampaignError(
            f"no runtime-profile named {runtime_profile_name!r} -- known: "
            f"{sorted(cfg.runtime_profiles)}"
        )
    campaign_run_id = run_id or uuid.uuid4().hex[:12]
    workdir = workdir or (context.work_root / "tune-campaigns" / campaign_run_id)
    workdir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    devices_tuple = _devices_tuple(devices)
    gpu_mod.preflight_context(profile=profile, devices=devices_tuple, stage="record")

    record_result, record_db_path = _stage_record(
        context=context, cfg=cfg, store=store, run_id=f"{campaign_run_id}-record",
        platform_name=platform_name, source_name=source_name, model_path=model_path,
        devices=devices, runtime_profile=profile, workdir=workdir,
    )
    inventory_path, _inventory_db_path = _stage_inventory_record(
        record_db_path=record_db_path, workdir=workdir,
    )
    tune_result, tune_measurements = _stage_tune(
        context=context, cfg=cfg, store=store, run_id=f"{campaign_run_id}-tune",
        platform_name=platform_name, source_name=source_name,
        inventory_path=inventory_path, model_path=model_path, devices=devices,
        runtime_profile=profile, screen_samples=tune_screen_samples,
        final_samples=tune_final_samples, workdir=workdir,
    )
    tune_manifest_path = (
        Path(tune_result.manifest_ref.path) if tune_result.manifest_ref else None
    )
    dispatch_db, _first_promote_result, promoted_before, missing_evidence = (
        _stage_load_and_promote(
            tune_measurements=tune_measurements, tune_manifest_path=tune_manifest_path,
            workdir=workdir, q=promotion_q, threshold_pct=promotion_threshold_pct,
            resamples=promotion_resamples,
        )
    )

    correctness_result: CampaignLaneResult | None = None
    promoted_after = promoted_before
    if missing_evidence > 0:
        # gpt review (2026-08-27): triggering only on promoted_before == 0
        # missed the partial-evidence case -- a dispatch_db that already
        # had SOME reusable evidence from a prior run promotes those
        # candidates immediately while others stay
        # rejected_no_correctness_evidence; that case must still generate
        # evidence for the REMAINING candidates, not be skipped just
        # because promoted_before > 0. Re-running is cheap and always
        # safe (idempotent over already-decided rows).
        correctness_result = _stage_correctness_evidence(
            context=context, cfg=cfg, store=store, run_id=f"{campaign_run_id}-correctness",
            platform_name=platform_name, source_name=source_name,
            inventory_path=inventory_path, tune_measurements=tune_measurements,
            dispatch_db=dispatch_db, seeds=correctness_seeds,
        )
        _dispatch_db2, _promote_result2, promoted_after, _missing_after = (
            _stage_load_and_promote(
                tune_measurements=tune_measurements, tune_manifest_path=tune_manifest_path,
                workdir=workdir, q=promotion_q, threshold_pct=promotion_threshold_pct,
                resamples=promotion_resamples,
            )
        )

    replay_result: CampaignLaneResult | None = None
    replay_coverage: dict | None = None
    dispatch_cache_path: Path | None = None
    if promoted_after > 0:
        if tune_manifest_path is None:
            raise TuneCampaignError("tune build produced no manifest_ref -- cannot export replay")
        dispatch_cache_path = _stage_replay_export(
            promoted_path=workdir / "promoted.jsonl", tune_manifest_path=tune_manifest_path,
            tune_source_root=tune_result.source_root, dispatch_db=dispatch_db, workdir=workdir,
        )
        replay_result, replay_coverage = _stage_replay_build_and_verify(
            context=context, cfg=cfg, store=store, run_id=f"{campaign_run_id}-replay",
            platform_name=platform_name, source_name=source_name,
            inventory_path=inventory_path, winners_path=workdir / "promoted.jsonl",
            model_path=model_path, devices=devices, runtime_profile=profile,
            dispatch_cache=dispatch_cache_path, workdir=workdir,
        )

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    receipt = WorkflowReceipt(
        schema_version=1,
        campaign_run_id=campaign_run_id,
        model_path=str(model_path),
        platform_name=platform_name,
        devices=devices,
        runtime_profile_name=runtime_profile_name,
        effective_tune_context=profile.tune_context,
        record=_stage_identity(record_result),
        tune=_stage_identity(tune_result),
        correctness=(
            _stage_identity(correctness_result)
            if correctness_result is not None else None
        ),
        replay=(
            _stage_identity(replay_result)
            if replay_result is not None else None
        ),
        promoted_before_evidence=promoted_before,
        promoted_after_evidence=promoted_after,
        dispatch_cache_path=str(dispatch_cache_path) if dispatch_cache_path else None,
        replay_coverage=replay_coverage,
        started_at=started_at,
        finished_at=finished_at,
    )
    receipt_path = workdir / "tune-campaign-receipt.json"
    atomic_write_json(receipt_path, asdict(receipt))
    store.publish_json(f"tune-campaigns/{campaign_run_id}/receipt.json", asdict(receipt))
    return receipt
