"""RE14/RE16: the reusable, in-process production API for executing one
campaign lane (materialize -> generate -> build -> runtime-smoke).

This is the extraction gpt-auto-agent's round-19 design called for: the
orchestration that used to live entirely inside re14_real_run.py's main()
is now a real library function, ``execute_campaign_lane()``, with no
argparse/print/stdout/process concerns of its own. re14_real_run.py becomes
a thin CLI adapter over it (parse args -> build a spec -> call this ->
render the result). A future multi-lane runner (RE18) calls it directly,
once per lane, reusing one ``cfg``/``context``/``store`` across all lanes
rather than each lane reloading config or reopening a store.

Only ``execute_campaign_lane()`` is public. The two real phases
(materialize, build) are deliberately private: campaign_plan.py documents
materialize as an identity-resolution boundary -- source_slice_id does not
exist until materialize actually runs -- and a public
``execute_build_lane(source_slice_id=...)`` would let a future caller skip
materialization and supply a source_slice_id it never actually proved,
defeating that boundary.

The same-run reuse proof previous rounds ran automatically (execute the
build CampaignRun twice, confirm every stage comes back "reused") is
deliberately NOT part of this API's normal execution. It proves
CampaignStageExecutor + make_artifact_reuse_checker's same-process/
same-executor contract (see that function's own docstring: "only valid
within the same process/executor instance") -- a test/harness concern, not
something a production multi-lane runner should pay a second
generate+build+GPU-smoke pass for on every lane it executes.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from . import campaign_build, campaign_plan, campaign_source, \
    campaign_workers, config as campaign_config
from . import runtime_smoke as smoke_module
from .artifacts import ArtifactStore
from .builds import BuildPlan
from .campaign import CampaignRun
from .campaign_execution import CampaignExecutionError, CampaignStageExecutor, \
    make_artifact_reuse_checker, require_campaign_success
from .context import ProjectContext
from .pipeline import ArtifactRef
from .provenance import make as make_provenance
from .workspace import UpstreamRepository


class CampaignLaneError(RuntimeError):
    pass


@dataclass(frozen=True)
class CampaignLaneExecutionSpec:
    """What one lane execution needs. Deliberately not named ``CampaignLane``
    -- RE18 is where the canonical planning object of that name belongs;
    this is today's execution-time shape (inventory required, smoke
    required), which RE17 will generalize.
    """

    source_name: str
    build_name: str
    platform_name: str
    architectures: tuple[str, ...]
    inventory_path: Path
    validation: smoke_module.RuntimeSmokeSpec
    binary_relative_path: str = "bin/llama-bench"
    c_compiler: str | None = None
    cxx_compiler: str | None = None
    smoke_environment: tuple[tuple[str, str], ...] | None = None
    #: Threaded through to campaign_plan.build_stage_graph() now (it already
    #: accepts this), even though nothing supplies real scheduler resource
    #: identities until RE18 -- avoids a second API change just to expose
    #: an argument that already exists one layer down.
    gpu_resource_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CampaignLaneResult:
    run_id: str
    resolved_revision: str
    source_slice_id: str
    build_plan: BuildPlan
    workload_id: str
    source_metadata_ref: ArtifactRef
    inventory_ref: ArtifactRef
    manifest_ref: ArtifactRef
    generated_tree_ref: ArtifactRef
    binary_ref: ArtifactRef
    smoke_ref: ArtifactRef

    @property
    def build_plan_id(self) -> str:
        return self.build_plan.build_plan_id


@dataclass(frozen=True)
class _MaterializedLaneSource:
    resolved_revision: str
    source_slice_id: str
    source_root: Path
    source_metadata_ref: ArtifactRef


def _require_one_artifact(
    refs: tuple[ArtifactRef, ...], *, kind: str, stage_id: str,
) -> ArtifactRef:
    """Replaces the previous ``executor.outputs[stage_id][0]`` positional
    indexing -- correct only because today's stages happen to return
    exactly one relevant kind first. Selecting by kind is robust to a
    stage's output ordering or count changing (e.g. RE17's optional stages).
    """
    matches = tuple(ref for ref in refs if ref.kind == kind)
    if len(matches) != 1:
        raise CampaignLaneError(
            f"stage {stage_id!r} expected exactly one {kind!r} artifact, "
            f"found {len(matches)}"
        )
    return matches[0]


def _materialize_worker(context: ProjectContext, source_plan) -> dict:
    # allow_dirty_bigcherry=True: real production callers run against a
    # clean checkout, but this proof/harness path has always run against a
    # repo with in-progress RE14 work -- see workspace.py's docstring on
    # this parameter for why it is a real, deliberate override rather than
    # a default.
    return campaign_build.materialize_source(context, source_plan, allow_dirty_bigcherry=True)


def _execute_materialize_phase(
    spec: CampaignLaneExecutionSpec, *,
    cfg: campaign_config.Config, context: ProjectContext, store: ArtifactStore,
    run_id: str, campaign_root: Path, resource_root: Path,
) -> _MaterializedLaneSource:
    source_plan = campaign_source.source_plan_for(cfg, spec.source_name)
    resolved_revision = UpstreamRepository(context.upstream_repo).resolve_ref(
        source_plan.upstream_revision)
    source_plan = replace(source_plan, upstream_revision=resolved_revision)
    source_root = context.work_root / "sources" / campaign_source.source_plan_id(source_plan)

    graph = campaign_plan.materialize_stage_graph(
        source_name=spec.source_name, build_name=spec.build_name,
        upstream_repo_path=str(context.upstream_repo))
    source_slice_id_holder: list[str | None] = [None]

    executor = CampaignStageExecutor(
        graph=graph, store=store, run_id=run_id,
        materialize=lambda: _materialize_worker(context, source_plan),
        generate=lambda inputs: {},
        source_slice_id_holder=source_slice_id_holder,
    )
    campaign_run = CampaignRun(graph, root=campaign_root / "materialize", run_id=run_id)
    reuse = make_artifact_reuse_checker(executor=executor, store=store)

    try:
        records = campaign_run.execute(executor, resource_root=resource_root, reuse=reuse)
        require_campaign_success(records, label="materialize")
    except CampaignExecutionError as exc:
        raise CampaignLaneError(f"materialize failed: {exc}") from exc

    source_slice_id = source_slice_id_holder[0]
    if not source_slice_id:
        raise CampaignLaneError("materialize succeeded without resolving source_slice_id")

    stage_id = f"{spec.source_name}:{spec.build_name}:materialize"
    source_metadata_ref = _require_one_artifact(
        executor.outputs[stage_id], kind="source-metadata", stage_id=stage_id)

    return _MaterializedLaneSource(
        resolved_revision=resolved_revision, source_slice_id=source_slice_id,
        source_root=source_root, source_metadata_ref=source_metadata_ref,
    )


def _publish_inventory(
    spec: CampaignLaneExecutionSpec, *, store: ArtifactStore,
    source_slice_id: str, run_id: str,
) -> ArtifactRef:
    # Content-addressed, not the previous fixed "inputs/inventory.json" --
    # ArtifactStore is immutable (publishing different bytes to the same
    # relative path raises ArtifactError), so a fixed path cannot survive a
    # multi-lane caller where two lanes use different inventories. This is
    # the one behavioral change bundled into this otherwise-pure extraction:
    # the API cannot safely serve its stated next consumer (a multi-lane
    # runner) without it.
    inventory_bytes = spec.inventory_path.read_bytes()
    inventory_digest = ArtifactStore.digest(inventory_bytes)
    inventory_relative = f"inputs/inventory/{inventory_digest}.json"
    published_digest = store.publish_bytes(inventory_relative, inventory_bytes)
    if published_digest != inventory_digest:
        raise CampaignLaneError("inventory digest changed during publication")

    doc = make_provenance(
        project={}, source={"source_slice_id": source_slice_id}, build={},
        workload={}, campaign={"run_id": run_id})
    return ArtifactRef(kind="inventory", path=store.resolve(inventory_relative),
                       content_hash=inventory_digest, provenance=doc)


def _execute_build_phase(
    spec: CampaignLaneExecutionSpec, *,
    cfg: campaign_config.Config, context: ProjectContext, store: ArtifactStore,
    run_id: str, campaign_root: Path, resource_root: Path,
    materialized: _MaterializedLaneSource,
) -> CampaignLaneResult:
    platform_cfg = cfg.platforms[spec.platform_name]
    if spec.c_compiler or spec.cxx_compiler:
        # A single override point: everything downstream (toolchain_request_
        # for_platform, make_build_worker's cmake_configure_args call) reads
        # platform_cfg.c_compiler/cxx_compiler, so replacing it here means
        # neither has to grow its own override plumbing.
        platform_cfg = replace(
            platform_cfg,
            c_compiler=spec.c_compiler or platform_cfg.c_compiler,
            cxx_compiler=spec.cxx_compiler or platform_cfg.cxx_compiler,
        )
    build_cfg = cfg.builds[spec.build_name]
    merged_options = dict(platform_cfg.options)
    merged_options.update(dict(build_cfg.options))

    inventory_ref = _publish_inventory(
        spec, store=store, source_slice_id=materialized.source_slice_id, run_id=run_id)
    # workload_id is deterministically the inventory's own content_hash --
    # knowable before generate ever runs, once the inventory is published.
    workload_id = inventory_ref.content_hash

    build_plan = BuildPlan(
        source_slice_id=materialized.source_slice_id, phase=spec.build_name,
        platform=platform_cfg.name, targets=platform_cfg.targets,
        cmake_options=tuple(sorted(merged_options.items())),
        variant_set=build_cfg.variant_set, inventory_hash=workload_id,
        toolchain_request=campaign_build.toolchain_request_for_platform(platform_cfg),
        environment=campaign_build.resolve_build_environment(),
    )

    build_graph = campaign_plan.build_stage_graph(
        source_name=spec.source_name, build_name=spec.build_name,
        source_slice_id=materialized.source_slice_id, build_plan_id=build_plan.build_plan_id,
        workload_id=workload_id, gpu_resource_ids=spec.gpu_resource_ids,
    )

    generate_worker = campaign_workers.make_generate_worker(
        context=context, source_root=materialized.source_root, run_id=run_id,
        variant_set=build_plan.variant_set or "inventory", architectures=list(spec.architectures),
        upstream_revision=materialized.resolved_revision,
    )
    build_worker = campaign_workers.make_build_worker(
        context=context, source_root=materialized.source_root, run_id=run_id,
        build_plan=build_plan, platform=platform_cfg, build=build_cfg,
        store=store, binary_relative_path=spec.binary_relative_path,
        source_slice_id=materialized.source_slice_id, workload_id=workload_id,
        cmake_targets=(Path(spec.binary_relative_path).name,),
        inventory_path=inventory_ref.path,
    )
    smoke_worker = campaign_workers.make_smoke_worker(
        run_id=run_id, store=store, source_slice_id=materialized.source_slice_id,
        build_plan_id=build_plan.build_plan_id, workload_id=workload_id,
        spec=spec.validation,
        environment=None if spec.smoke_environment is None else dict(spec.smoke_environment),
    )

    executor = CampaignStageExecutor(
        graph=build_graph, store=store, run_id=run_id,
        materialize=lambda: {}, generate=generate_worker,
        source_slice_id_holder=[materialized.source_slice_id],
        build_plan_id=build_plan.build_plan_id, inventory_ref=inventory_ref,
        workload_id=workload_id, build=build_worker, smoke=smoke_worker,
    )
    campaign_run = CampaignRun(build_graph, root=campaign_root / "build", run_id=run_id)
    reuse = make_artifact_reuse_checker(executor=executor, store=store)

    try:
        records = campaign_run.execute(executor, resource_root=resource_root, reuse=reuse)
        require_campaign_success(records, label="generate/build/runtime-smoke")
    except CampaignExecutionError as exc:
        raise CampaignLaneError(f"generate/build/runtime-smoke failed: {exc}") from exc

    generate_stage_id = f"{spec.source_name}:{spec.build_name}:generate"
    build_stage_id = f"{spec.source_name}:{spec.build_name}:build"
    smoke_stage_id = f"{spec.source_name}:{spec.build_name}:runtime-smoke"

    manifest_ref = _require_one_artifact(
        executor.outputs[generate_stage_id], kind="manifest", stage_id=generate_stage_id)
    generated_tree_ref = _require_one_artifact(
        executor.outputs[generate_stage_id], kind="generated-tree", stage_id=generate_stage_id)
    binary_ref = _require_one_artifact(
        executor.outputs[build_stage_id], kind="binary", stage_id=build_stage_id)
    smoke_ref = _require_one_artifact(
        executor.outputs[smoke_stage_id], kind="smoke-result", stage_id=smoke_stage_id)

    generated_workload_id = manifest_ref.provenance["workload"]["workload_id"]
    if generated_workload_id != workload_id:
        raise CampaignLaneError(
            f"generate established workload_id {generated_workload_id!r}, but the "
            f"graph was built with the precomputed {workload_id!r} -- these must agree"
        )

    return CampaignLaneResult(
        run_id=run_id, resolved_revision=materialized.resolved_revision,
        source_slice_id=materialized.source_slice_id, build_plan=build_plan,
        workload_id=workload_id, source_metadata_ref=materialized.source_metadata_ref,
        inventory_ref=inventory_ref, manifest_ref=manifest_ref,
        generated_tree_ref=generated_tree_ref, binary_ref=binary_ref, smoke_ref=smoke_ref,
    )


def execute_campaign_lane(
    spec: CampaignLaneExecutionSpec, *,
    cfg: campaign_config.Config, context: ProjectContext, store: ArtifactStore,
    run_id: str | None = None,
) -> CampaignLaneResult:
    """Execute one campaign lane: materialize, then generate/build/smoke.
    Each graph runs exactly once -- no automatic reuse-proof second pass
    (see module docstring). Raises CampaignLaneError on any failure.
    """
    effective_run_id = run_id or uuid.uuid4().hex[:12]
    campaign_root = context.work_root / "campaign-runs" / effective_run_id
    resource_root = context.work_root / "resource-locks"

    materialized = _execute_materialize_phase(
        spec, cfg=cfg, context=context, store=store, run_id=effective_run_id,
        campaign_root=campaign_root, resource_root=resource_root,
    )
    return _execute_build_phase(
        spec, cfg=cfg, context=context, store=store, run_id=effective_run_id,
        campaign_root=campaign_root, resource_root=resource_root, materialized=materialized,
    )


def smoke_environment_for_hip_devices(hip_visible_devices: str) -> tuple[tuple[str, str], ...]:
    """The CLI-convenience translation from a single --hip-visible-devices
    flag to the environment dict the smoke worker actually accepts --
    kept out of the reusable API itself (make_smoke_worker's own environment
    parameter is HIP-agnostic; this is a CLI-specific convenience)."""
    return tuple(sorted({
        "HIP_VISIBLE_DEVICES": hip_visible_devices,
        "ROCR_VISIBLE_DEVICES": hip_visible_devices,
        "PATH": os.environ.get("PATH", ""),
    }.items()))
