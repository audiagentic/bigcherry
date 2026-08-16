"""RE14: a real, standalone run of the new campaign path for one lane.

Not a permanent CLI command -- this is the first real end-to-end proof
that materialize -> generate -> build -> runtime-smoke works against real
git, a real compiler, and a real GPU, via the primitives built across this
session (campaign_source, campaign_plan, campaign_build,
campaign_execution, campaign_workers). The legacy ``cmd_build`` path is
completely untouched; this writes only into ``context.work_root`` and a
dedicated ArtifactStore directory.

Usage:
    python -m bigcherry.re14_real_run \
        --upstream-repo /mnt/vault/development/bc-branch/vendor/llama.cpp \
        --inventory artifacts/campaign-gfx1100-inventory.json \
        --arch gfx1100 \
        --model /mnt/vault/llm-models/qwen3.5-0.8B/gguf/Qwen3.5-0.8B-UD-Q5_K_XL.gguf \
        --hip-visible-devices 0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import campaign_build, campaign_execution, campaign_plan, campaign_source, \
    campaign_workers, config
from . import runtime_smoke as smoke_module
from .artifacts import ArtifactStore
from .builds import BuildPlan
from .campaign import CampaignRun
from .campaign_execution import CampaignExecutionError, make_artifact_reuse_checker, \
    require_campaign_success
from .context import ProjectContext
from .pipeline import ArtifactRef
from .provenance import make as make_provenance
from .workspace import UpstreamRepository


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--arch", required=True, help="comma-separated, e.g. gfx1100")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", default="bigcherry")
    parser.add_argument("--build", default="tune")
    parser.add_argument("--platform", default="linux-multi")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--hip-visible-devices", default="0")
    parser.add_argument("--split-mode", default="none")
    parser.add_argument("--binary-relative-path", default="bin/llama-bench")
    parser.add_argument(
        "--c-compiler", default=None,
        help="override platform.c_compiler -- e.g. to build against a "
             "specific ROCm install for a version comparison")
    parser.add_argument("--cxx-compiler", default=None)
    args = parser.parse_args(argv)

    import uuid
    run_id = args.run_id or uuid.uuid4().hex[:12]
    print(f"=== RE14 real run {run_id} ===")

    cfg = config.load(Path("recipes.toml"))
    context = ProjectContext.resolve(work_root=args.work_root)
    context = ProjectContext(
        project_root=context.project_root, config_path=context.config_path,
        artifacts_root=context.artifacts_root, work_root=context.work_root,
        upstream_repo=args.upstream_repo.resolve(),
        overlay_root=context.overlay_root, patches_root=context.patches_root,
    )
    store = ArtifactStore(context.work_root / "artifacts-store")

    source_plan = campaign_source.source_plan_for(cfg, args.source)
    resolved_revision = UpstreamRepository(context.upstream_repo).resolve_ref(
        source_plan.upstream_revision
    )
    source_plan = replace(source_plan, upstream_revision=resolved_revision)
    print(f"source plan: overlay={source_plan.overlay_enabled} "
          f"patches={len(source_plan.patch_ids)} revision={resolved_revision}")

    resource_root = context.work_root / "resource-locks"
    campaign_root = context.work_root / "campaign-runs" / run_id

    materialize_graph = campaign_plan.materialize_stage_graph(
        source_name=args.source, build_name=args.build,
        upstream_repo_path=str(context.upstream_repo))
    source_slice_id_holder: list[str | None] = [None]
    source_root = context.work_root / "sources" / campaign_source.source_plan_id(source_plan)

    materialize_executor = campaign_execution.CampaignStageExecutor(
        graph=materialize_graph, store=store, run_id=run_id,
        materialize=lambda: campaign_workers_materialize(context, source_plan),
        generate=lambda inputs: {},
        source_slice_id_holder=source_slice_id_holder,
    )
    # Through CampaignRun.execute(), not a direct executor(stage_id) call:
    # CampaignStageExecutor.__call__(stage_id) -> tuple[str, ...] is exactly
    # the Callable[[str], tuple[str, ...]] shape CampaignRun requires, so it
    # composes directly with no adapter -- confirmed against both signatures
    # before relying on it. This also means the materialize stage's
    # upstream-worktree ResourceClaim (from campaign_plan.materialize_stage_graph)
    # is now actually exercised via a real ResourceLock, not bypassed.
    materialize_run = CampaignRun(materialize_graph, root=campaign_root / "materialize", run_id=run_id)
    materialize_reuse = make_artifact_reuse_checker(executor=materialize_executor, store=store)
    try:
        materialize_records = materialize_run.execute(
            materialize_executor, resource_root=resource_root, reuse=materialize_reuse)
        require_campaign_success(materialize_records, label="materialize")
    except CampaignExecutionError as exc:
        print(f"MATERIALIZE FAILED: {exc}", file=sys.stderr)
        return 1
    source_slice_id = source_slice_id_holder[0]
    if not source_slice_id:
        print("MATERIALIZE FAILED: succeeded without resolving source_slice_id", file=sys.stderr)
        return 1
    print(f"materialized: source_slice_id={source_slice_id}")

    platform_cfg = cfg.platforms[args.platform]
    if args.c_compiler or args.cxx_compiler:
        # A single override point: everything downstream (toolchain_request_
        # for_platform, make_build_worker's cmake_configure_args call) reads
        # platform_cfg.c_compiler/cxx_compiler, so replacing it here means
        # neither has to grow its own override plumbing -- the override is
        # indistinguishable from the recipe having declared it that way.
        platform_cfg = replace(
            platform_cfg,
            c_compiler=args.c_compiler or platform_cfg.c_compiler,
            cxx_compiler=args.cxx_compiler or platform_cfg.cxx_compiler,
        )
    build_cfg = cfg.builds[args.build]
    # cmake_configure_args always compiles for platform.targets (AMDGPU_TARGETS),
    # never build_plan.targets -- there is exactly one authority for what
    # architectures actually get compiled, and it is the platform, not this
    # CLI's --arch. --arch only selects which architectures the CATALOG
    # (autotune_catalog.build_manifest) generates candidates for. Passing
    # platform_cfg.targets into BuildPlan too keeps BuildPlan.targets
    # truthful about what will actually be compiled, instead of silently
    # diverging from it as an earlier version of this script did.
    architectures = args.arch.split(",")
    merged_options = dict(platform_cfg.options)
    merged_options.update(dict(build_cfg.options))

    inventory_bytes = args.inventory.read_bytes()
    inventory_digest = store.publish_bytes("inputs/inventory.json", inventory_bytes)
    inventory_doc = make_provenance(
        project={}, source={"source_slice_id": source_slice_id}, build={},
        workload={}, campaign={"run_id": run_id})
    inventory_ref = ArtifactRef(kind="inventory", path=store.resolve("inputs/inventory.json"),
                                content_hash=inventory_digest, provenance=inventory_doc)

    # workload_id is deterministically the inventory's own content_hash
    # (see campaign_workers.make_generate_worker) -- so it is knowable here,
    # before generate ever runs, once the inventory is published. Computing
    # it now and building the graph with the real value (not a placeholder)
    # eliminates the split-brain identity a StageNode built before its real
    # identity is known would otherwise create.
    workload_id = inventory_digest

    build_plan = BuildPlan(
        source_slice_id=source_slice_id, phase=args.build, platform=platform_cfg.name,
        targets=platform_cfg.targets, cmake_options=tuple(sorted(merged_options.items())),
        variant_set=build_cfg.variant_set, inventory_hash=inventory_digest,
        toolchain_request=campaign_build.toolchain_request_for_platform(platform_cfg),
    )
    print(f"build plan: id={build_plan.build_plan_id} variant_set={build_plan.variant_set}")

    build_graph = campaign_plan.build_stage_graph(
        source_name=args.source, build_name=args.build,
        source_slice_id=source_slice_id, build_plan_id=build_plan.build_plan_id,
        workload_id=workload_id,
    )

    generate_worker = campaign_workers.make_generate_worker(
        context=context, source_root=source_root, run_id=run_id,
        variant_set=build_plan.variant_set or "inventory", architectures=architectures,
        upstream_revision=resolved_revision,
    )

    build_worker = campaign_workers.make_build_worker(
        context=context, source_root=source_root, run_id=run_id,
        build_plan=build_plan, platform=platform_cfg, build=build_cfg,
        store=store, binary_relative_path=args.binary_relative_path,
        source_slice_id=source_slice_id, workload_id=workload_id,
        cmake_targets=(Path(args.binary_relative_path).name,),
        # The verified, published copy -- not the caller's original file.
        # Passing args.inventory directly would let CMake compile against
        # whatever that path currently contains, even if it was mutated
        # after inventory_ref's bytes were hashed and published; the
        # artifact's own provenance would then describe content that is no
        # longer what the build actually consumed.
        inventory_path=inventory_ref.path,
    )
    smoke_spec = smoke_module.RuntimeSmokeSpec(
        model_path=args.model, split_mode=args.split_mode,
    )
    smoke_worker = campaign_workers.make_smoke_worker(
        run_id=run_id, store=store, source_slice_id=source_slice_id,
        build_plan_id=build_plan.build_plan_id, workload_id=workload_id,
        spec=smoke_spec,
        environment={
            "HIP_VISIBLE_DEVICES": args.hip_visible_devices,
            "ROCR_VISIBLE_DEVICES": args.hip_visible_devices,
            "PATH": __import__("os").environ.get("PATH", ""),
        },
    )

    # All three workers configured before CampaignRun.execute() ever runs
    # -- it iterates the whole graph in one pass, so (unlike the old
    # manual stage-by-stage calls) there is no opportunity to construct a
    # later stage's worker only after an earlier one has already run.
    executor = campaign_execution.CampaignStageExecutor(
        graph=build_graph, store=store, run_id=run_id,
        materialize=lambda: {}, generate=generate_worker,
        source_slice_id_holder=source_slice_id_holder,
        build_plan_id=build_plan.build_plan_id, inventory_ref=inventory_ref,
        workload_id=workload_id,
        build=build_worker, smoke=smoke_worker,
    )
    build_run = CampaignRun(build_graph, root=campaign_root / "build", run_id=run_id)
    build_reuse = make_artifact_reuse_checker(executor=executor, store=store)

    try:
        first_pass = build_run.execute(executor, resource_root=resource_root, reuse=build_reuse)
        require_campaign_success(first_pass, label="generate/build/runtime-smoke")
    except CampaignExecutionError as exc:
        print(f"BUILD CAMPAIGN FAILED: {exc}", file=sys.stderr)
        return 1

    generate_stage_id = f"{args.source}:{args.build}:generate"
    build_stage_id = f"{args.source}:{args.build}:build"
    smoke_stage_id = f"{args.source}:{args.build}:runtime-smoke"

    generated_workload_id = executor.outputs[generate_stage_id][0].provenance["workload"]["workload_id"]
    if generated_workload_id != workload_id:
        print(
            f"BUILD CAMPAIGN FAILED: generate established workload_id "
            f"{generated_workload_id!r}, but the graph was built with the "
            f"precomputed {workload_id!r} -- these must agree",
            file=sys.stderr,
        )
        return 1
    print(f"generated: workload_id={workload_id}")
    print(f"built: {len(executor.outputs[build_stage_id])} artifact(s) published")
    smoke_ref = executor.outputs[smoke_stage_id][0]
    result = json.loads(smoke_ref.path.read_text(encoding="utf-8"))
    print(f"smoke: {json.dumps(result, indent=2)}")

    # The actual CampaignRun -> reuse callback -> ArtifactStore.verify()
    # proof: run the SAME CampaignRun object again over the same graph.
    # Every stage must come back "reused", not re-execute for real (no
    # second git worktree materialize, no second compile, no second GPU
    # smoke run).
    try:
        second_pass = build_run.execute(executor, resource_root=resource_root, reuse=build_reuse)
        require_campaign_success(second_pass, label="generate/build/runtime-smoke reuse pass")
    except CampaignExecutionError as exc:
        print(f"BUILD CAMPAIGN REUSE PASS FAILED: {exc}", file=sys.stderr)
        return 1
    not_reused = {stage_id: record.state for stage_id, record in second_pass.items()
                  if record.state != "reused"}
    if not_reused:
        print(f"BUILD CAMPAIGN REUSE PASS FAILED: expected every stage to "
              f"reuse, got {not_reused}", file=sys.stderr)
        return 1
    print("reuse pass: every stage reused, none re-executed")

    # A machine-readable summary for tooling that wants to load this run's
    # artifacts back out (RE14's parity harness, re14_parity_run.py) without
    # re-deriving the store-relative paths itself -- store.root plus the
    # published ArtifactRefs already published are the only source of truth
    # for those paths; recomputing them independently would risk drifting
    # from what CampaignStageExecutor/campaign_workers actually publish to.
    manifest_ref = executor.outputs[generate_stage_id][0]
    binary_ref = executor.outputs[build_stage_id][0]
    print("RE14_PARITY_RESULT_JSON: " + json.dumps({
        "run_id": run_id,
        "source_slice_id": source_slice_id,
        "build_plan_id": build_plan.build_plan_id,
        "workload_id": workload_id,
        "store_root": str(store.root),
        "manifest_relative": str(manifest_ref.path.relative_to(store.root).as_posix()),
        "manifest_content_hash": manifest_ref.content_hash,
        "binary_relative": str(binary_ref.path.relative_to(store.root).as_posix()),
    }))

    print(f"=== RE14 real run {run_id}: ALL STAGES PASSED ===")
    return 0


def campaign_workers_materialize(context: ProjectContext, source_plan) -> dict:
    # allow_dirty_bigcherry=True: this repo has uncommitted RE14 work in
    # progress (the very files this script is part of). That is a real,
    # deliberate override of workspace.materialize's dirty-tree guard, not
    # a default -- see workspace.py's docstring on that parameter.
    return campaign_build.materialize_source(context, source_plan, allow_dirty_bigcherry=True)


if __name__ == "__main__":
    raise SystemExit(main())
