"""RE15: one continuous real-hardware run of the full evidence chain
(source -> record -> inventory -> tune -> promotion -> replay-export ->
runtime validation -> balanced comparison -> release pointer), locked per
RV50 steps A-I.

Not a permanent CLI command -- a harness script, same status as
re14_real_run.py, which this borrows its CLI-wrapper shape from. The
record/tune/replay lane BUILDS go through campaign_lane.execute_campaign_lane()
(the real materialize/generate/build/smoke orchestration); everything from
"run the record binary and capture its dispatch signatures" onward goes
through lifecycle.py's stage functions (RE10), comparisons.py (RE12), and
promotion.py -- no orchestration logic is reimplemented here.

Usage (device 2 = gfx1201 on Brutus; NEVER 0/1, those carry production
traffic -- check `rocm-smi --showuse` first):
    python -m bigcherry.re15_acceptance_run \
        --upstream-repo /mnt/vault/development/bc-branch/vendor/llama.cpp \
        --arch gfx1201 \
        --model /mnt/vault/llm-models/qwen3.5-0.8B/gguf/Qwen3.5-0.8B-UD-Q5_K_XL.gguf \
        --hip-visible-devices 2
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import uuid
from pathlib import Path

from . import comparisons, config, promotion, provenance
from .artifacts import ArtifactStore
from .campaign_lane import (
    CampaignLaneError,
    CampaignLaneExecutionSpec,
    CampaignLaneResult,
    execute_campaign_lane,
    smoke_environment_for_hip_devices,
)
from .context import ProjectContext
from . import lifecycle
from .lifecycle import LifecycleError
from .releases import ReleaseRecord
from .runtime_smoke import RuntimeSmokeSpec


def _source_root(context: ProjectContext, lane: CampaignLaneResult) -> Path:
    """The EXACT materialized source directory this lane's build actually
    compiled from -- not a directory-scanning guess.

    Pre-RE26 fix: this used to glob context.work_root/sources for "the"
    tree carrying ggml/include/ggml.h, on the assumption exactly one would
    ever exist. That broke the moment more than one materialized tree
    accumulates in the shared cache (any session that has run more than
    one build against this work_root, which is now the common case, not a
    rare one) -- SystemExit("ambiguous materialized source trees ..."),
    real-hardware-confirmed 2026-08-19. materialize_source() already
    writes materialization_plan_id into the source-metadata record it
    publishes as source_metadata_ref (the SAME identity
    campaign_build.materialize_source() used to name the directory), so
    read it back instead of re-deriving or guessing.
    """
    metadata = json.loads(lane.source_metadata_ref.path.read_text(encoding="utf-8"))
    plan_id = metadata.get("materialization_plan_id")
    if not isinstance(plan_id, str) or not plan_id:
        raise SystemExit(
            f"lane source-metadata artifact {lane.source_metadata_ref.artifact_id!r} "
            "has no materialization_plan_id -- cannot locate its materialized source tree"
        )
    source_root = context.work_root / "sources" / plan_id
    if not (source_root / "ggml" / "include" / "ggml.h").is_file():
        raise SystemExit(
            f"materialized source tree {source_root} (from lane source_slice_id "
            f"{lane.source_slice_id!r}) is missing or incomplete"
        )
    return source_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", required=True, type=Path)
    parser.add_argument("--arch", required=True, help="e.g. gfx1201")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", default="bigcherry")
    parser.add_argument("--platform", default="linux-multi")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--hip-visible-devices", required=True,
                         help="MUST be 2 or 3 on Brutus -- never 0/1 (production traffic)")
    parser.add_argument("--split-mode", default="none")
    parser.add_argument(
        "--direct-op-corpus-filter",
        default="m=127,n=128,k=256|m=32,.*,k=256",
        help="RE26: test-backend-ops -p filter selecting HI70's direct-op "
             "correctness corpus (patches/1100_hi70_direct_op_evidence.py) "
             "-- covers candidates (MMQ fb1, MMF narrow-batch) no real "
             "workload can reach. Pass '' to skip this stage entirely.",
    )
    args = parser.parse_args(argv)

    if args.hip_visible_devices in ("0", "1"):
        print("error: --hip-visible-devices must not be 0 or 1 (production traffic)", file=sys.stderr)
        return 2

    run_id = args.run_id or uuid.uuid4().hex[:12]
    print(f"=== RE15 acceptance run {run_id} (device {args.hip_visible_devices}) ===")

    context = ProjectContext.resolve(work_root=args.work_root, upstream_repo=args.upstream_repo)
    cfg = config.load(context.config_path)
    store = ArtifactStore(context.work_root / "artifacts-store")
    manifest: dict[str, object] = {"run_id": run_id, "device": args.hip_visible_devices}

    environment = smoke_environment_for_hip_devices(args.hip_visible_devices)
    architectures = (args.arch,)

    def _spec(build_name: str, inputs: tuple[tuple[str, object], ...] = (),
              extra_cmake_targets: tuple[str, ...] = ()) -> CampaignLaneExecutionSpec:
        return CampaignLaneExecutionSpec(
            source_name=args.source, build_name=build_name, platform_name=args.platform,
            architectures=architectures, inputs=inputs,
            validation=RuntimeSmokeSpec(model_path=args.model, split_mode=args.split_mode),
            smoke_environment=environment,
            extra_cmake_targets=extra_cmake_targets,
        )

    try:
        # A: source/build (record)
        print("--- A: materialize + build record lane ---")
        record_lane = execute_campaign_lane(
            _spec("record"), cfg=cfg, context=context, store=store, run_id=f"{run_id}-record-build")
        manifest["record_build"] = {
            "source_slice_id": record_lane.source_slice_id,
            "build_plan_id": record_lane.build_plan_id,
            "effective_build_id": record_lane.effective_build_id,
            "runtime_bundle_artifact_id": record_lane.runtime_bundle_ref.artifact_id,
            "binary_artifact_id": record_lane.binary_ref.artifact_id,
        }
        print(f"    record build: {manifest['record_build']}")

        # B: record
        print("--- B: run record capture ---")
        record_result = lifecycle.execute_record_stage(
            context=context, store=store, run_id=run_id,
            runtime_bundle=record_lane.runtime_bundle_ref,
            spec=RuntimeSmokeSpec(model_path=args.model, split_mode=args.split_mode),
            environment=dict(environment), local_provenance_class="production",
        )
        manifest["record_artifact_id"] = record_result.record_ref.artifact_id
        print(f"    record-jsonl: {record_result.record_ref.artifact_id}")

        # C: inventory
        print("--- C: build inventory + schema-4 dispatch-db ---")
        inventory_result = lifecycle.execute_inventory_stage(
            context=context, store=store, run_id=run_id, record=record_result.record_ref,
            local_provenance_class="production",
        )
        manifest["inventory_artifact_id"] = inventory_result.inventory_ref.artifact_id
        manifest["dispatch_db_artifact_id_c"] = inventory_result.database_ref.artifact_id
        manifest["workload_id"] = inventory_result.workload_id
        print(f"    inventory: {inventory_result.inventory_ref.artifact_id} "
              f"workload_id={inventory_result.workload_id}")

        conn = sqlite3.connect(str(inventory_result.database_ref.path))
        try:
            rows = conn.execute("SELECT identity_scope, source_slice_id, build_plan_id, "
                                 "effective_build_id FROM build").fetchall()
        finally:
            conn.close()
        print(f"    build rows: {rows}")
        non_null = [r for r in rows if r[0] == "campaign" and all(r[1:])]
        if not non_null:
            raise SystemExit("ACCEPTANCE FAILURE: no campaign-scoped build row with non-null identity")

        # D: tune
        print("--- D: build + run tune lane ---")
        tune_lane = execute_campaign_lane(
            _spec("tune", inputs=(("inventory", inventory_result.inventory_ref),),
                  extra_cmake_targets=("test-backend-ops",) if args.direct_op_corpus_filter else ()),
            cfg=cfg, context=context, store=store, run_id=f"{run_id}-tune-build")
        manifest["tune_build"] = {
            "build_plan_id": tune_lane.build_plan_id,
            "effective_build_id": tune_lane.effective_build_id,
            "runtime_bundle_artifact_id": tune_lane.runtime_bundle_ref.artifact_id,
            "binary_artifact_id": tune_lane.binary_ref.artifact_id,
        }
        print(f"    tune build: {manifest['tune_build']}")

        tune_result = lifecycle.execute_tune_stage(
            context=context, store=store, run_id=run_id,
            runtime_bundle=tune_lane.runtime_bundle_ref, dispatch_db=inventory_result.database_ref,
            spec=RuntimeSmokeSpec(model_path=args.model, split_mode=args.split_mode),
            environment=dict(environment), local_provenance_class="production",
        )
        manifest["measurements_artifact_id"] = tune_result.measurements_ref.artifact_id
        manifest["dispatch_db_artifact_id_d"] = tune_result.database_ref.artifact_id
        print(f"    measurements: {tune_result.measurements_ref.artifact_id}")

        # D2: HI70 direct-op evidence corpus (RE26) -- candidates no real
        # workload can ever reach (MMQ fb1, MMF narrow-batch). Same
        # continuous run, same compiled catalog as the tune pass above.
        if args.direct_op_corpus_filter:
            print("--- D2: run HI70 direct-op evidence corpus ---")
            tune_result = lifecycle.execute_direct_op_evidence_stage(
                context=context, store=store, run_id=run_id,
                runtime_bundle=tune_lane.runtime_bundle_ref, prior_tune=tune_result,
                corpus_op_filter=args.direct_op_corpus_filter,
                environment=dict(environment), local_provenance_class="production",
            )
            manifest["direct_op_measurements_artifact_id"] = tune_result.measurements_ref.artifact_id
            manifest["direct_op_dispatch_db_artifact_id"] = tune_result.database_ref.artifact_id
            print(f"    direct-op-evidence measurements: {tune_result.measurements_ref.artifact_id}")

        # E: promotion
        print("--- E: promote winners ---")
        promotion_result = lifecycle.execute_promotion_stage(
            context=context, store=store, run_id=run_id, measurements=tune_result.measurements_ref,
            local_provenance_class="production",
        )
        manifest["promoted_winners_artifact_id"] = promotion_result.promoted_winners_ref.artifact_id
        provenance.require_promotable(
            promotion_result.promoted_winners_ref.provenance, kind="promoted-winners")
        print(f"    promoted-winners: {promotion_result.promoted_winners_ref.artifact_id} (promotable)")

        # F: replay export
        print("--- F: build replay-full lane + export replay cache ---")
        replay_lane = execute_campaign_lane(
            _spec("replay", inputs=(
                ("inventory", inventory_result.inventory_ref),
                ("promoted-winners", promotion_result.promoted_winners_ref),
            )), cfg=cfg, context=context, store=store, run_id=f"{run_id}-replay-build")
        manifest["replay_build"] = {
            "build_plan_id": replay_lane.build_plan_id,
            "effective_build_id": replay_lane.effective_build_id,
            "runtime_bundle_artifact_id": replay_lane.runtime_bundle_ref.artifact_id,
            "binary_artifact_id": replay_lane.binary_ref.artifact_id,
            "manifest_artifact_id": replay_lane.manifest_ref.artifact_id if replay_lane.manifest_ref else None,
        }
        print(f"    replay build: {manifest['replay_build']}")
        if replay_lane.manifest_ref is None:
            raise SystemExit("ACCEPTANCE FAILURE: replay-full build produced no manifest")

        provenance.require_promotable(replay_lane.runtime_bundle_ref.provenance, kind="runtime-bundle")

        source_root = _source_root(context, replay_lane)
        replay_export_result = lifecycle.execute_replay_export_stage(
            context=context, store=store, run_id=run_id,
            promoted_winners=promotion_result.promoted_winners_ref, manifest=replay_lane.manifest_ref,
            source_root=source_root, local_provenance_class="production",
        )
        manifest["replay_cache_artifact_id"] = replay_export_result.replay_cache_ref.artifact_id
        print(f"    replay-cache: {replay_export_result.replay_cache_ref.artifact_id}")

        # G: runtime validation
        print("--- G: run replay validation, require exact coverage ---")
        validation_result = lifecycle.execute_replay_validation_stage(
            context=context, store=store, run_id=run_id,
            runtime_bundle=replay_lane.runtime_bundle_ref,
            replay_cache_artifact=replay_export_result.replay_cache_ref,
            spec=RuntimeSmokeSpec(model_path=args.model, split_mode=args.split_mode),
            environment=dict(environment), local_provenance_class="production",
        )
        manifest["replay_coverage_artifact_id"] = validation_result.coverage_ref.artifact_id
        manifest["replay_coverage"] = validation_result.coverage
        print(f"    coverage: {validation_result.coverage}")
        if validation_result.coverage.get("exact") != validation_result.coverage.get("total_dispatched"):
            raise SystemExit(f"ACCEPTANCE FAILURE: replay coverage not exact: {validation_result.coverage}")

        # H: balanced comparison (tune build vs replay build)
        print("--- H: balanced comparison (tune vs replay) ---")
        left_arm = comparisons.BenchmarkArm(
            name="tune", runtime_bundle_artifact_id=tune_lane.runtime_bundle_ref.artifact_id,
            binary_artifact_id=tune_lane.binary_ref.artifact_id, replay_cache_artifact_id=None,
            source_slice_id=tune_lane.source_slice_id, build_plan_id=tune_lane.build_plan_id,
            effective_build_id=tune_lane.effective_build_id or "", workload_id=inventory_result.workload_id,
            environment=environment, device=args.hip_visible_devices,
        )
        right_arm = comparisons.BenchmarkArm(
            name="replay", runtime_bundle_artifact_id=replay_lane.runtime_bundle_ref.artifact_id,
            binary_artifact_id=replay_lane.binary_ref.artifact_id,
            replay_cache_artifact_id=replay_export_result.replay_cache_ref.artifact_id,
            source_slice_id=replay_lane.source_slice_id, build_plan_id=replay_lane.build_plan_id,
            effective_build_id=replay_lane.effective_build_id or "", workload_id=inventory_result.workload_id,
            environment=environment, device=args.hip_visible_devices,
        )
        plan = comparisons.plan_pair(
            left_arm, right_arm, label="tune-vs-replay",
            allowed_differences=frozenset({
                "build_plan_id", "effective_build_id", "runtime_bundle_artifact_id",
                "binary_artifact_id", "replay_cache_artifact_id"}))
        metric_patterns = {"pp": re.compile(r'"avg_ts"\s*:\s*([0-9.]+)')}
        report_ref = comparisons.run_comparison(
            plan, store=store, run_id=run_id, model_args=["-p", "512", "-n", "128", "-r", "3", "-o", "json"],
            output=context.work_root / "runs" / run_id / "compare-out", pairs=6,
            metric_patterns=metric_patterns, structured=False, practical_threshold_pct=0.5,
            resamples=5000, decision_grade=True, campaign_plan_id="re15-acceptance",
            comparison_plan_id="tune-vs-replay", local_provenance_class="production",
        )
        manifest["comparison_report_artifact_id"] = report_ref.artifact_id
        report = json.loads(report_ref.path.read_text(encoding="utf-8"))
        print(f"    report: valid={report['valid']} decision_grade={report['decision_grade']} "
              f"effects={report.get('effects')}")
        if not report["valid"] or not report["decision_grade"]:
            raise SystemExit(f"ACCEPTANCE FAILURE: comparison report not valid/decision-grade: {report}")

        # I: release pointer
        print("--- I: build release pointer from the stored report ---")
        pointer = promotion.pointer_from_comparison_report(
            store=store, report_artifact_id=report_ref.artifact_id,
            release_tag=f"re15-acceptance-{run_id}",
            replay_coverage_artifact_id=validation_result.coverage_ref.artifact_id,
            required_architectures=architectures,
        )
        manifest["promotion_pointer"] = pointer.document()
        print(f"    pointer: {json.dumps(pointer.document(), indent=2)}")

        record = ReleaseRecord(revision=record_lane.resolved_revision,
                                release_tag=f"re15-acceptance-{run_id}")
        record.stage = "patched"
        from .releases import promote as releases_promote
        releases_promote(record, pointer)
        record.validate()
        assert record.stage == "validated"
        # Independent reload proof: re-parse the persisted document shape
        # (never write this test revision into the real releases/ ledger).
        reloaded = ReleaseRecord(**{
            f: getattr(record, f) for f in ReleaseRecord.__dataclass_fields__
        })
        reloaded.validate()
        print("    release record reaches 'validated' and independently re-validates (in-memory only, "
              "not persisted to releases/)")

    except (CampaignLaneError, LifecycleError, comparisons.ComparisonError,
            promotion.PromotionError) as exc:
        print(f"RE15 ACCEPTANCE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("RE15_MANIFEST_JSON: " + json.dumps(manifest, indent=2, default=str))
        return 1

    print("RE15_MANIFEST_JSON: " + json.dumps(manifest, indent=2, default=str))
    print(f"=== RE15 acceptance run {run_id}: STAGES A-I PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
