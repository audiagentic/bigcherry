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
    campaign_workers, config as campaign_config, patchset
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


#: A raw path (the CLI importing a file, e.g. --inventory) or an
#: already-published ArtifactRef (a future canonical-lane caller, RE18).
#: Both normalize to a verified ArtifactRef in _resolve_lane_inputs()
#: before any stage runs.
LaneInputValue = Path | ArtifactRef


@dataclass(frozen=True)
class CampaignLaneExecutionSpec:
    """What one lane execution needs. Deliberately not named ``CampaignLane``
    -- RE18 is where the canonical planning object of that name belongs;
    this is today's execution-time shape.

    ``inputs`` keys must exactly equal ``cfg.builds[build_name].needs`` --
    config.Build.needs (a real frozenset[str]) is the single authority for
    what a lane requires, not a second typed inventory/winners schema
    living beside it. A tuple of pairs, not a dict, because the spec is
    frozen/hashable; ``_spec_inputs()`` below normalizes it.
    """

    source_name: str
    build_name: str
    platform_name: str
    architectures: tuple[str, ...]
    inputs: tuple[tuple[str, LaneInputValue], ...] = ()
    validation: smoke_module.RuntimeSmokeSpec | None = None
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
    #: None for a build with no workload at all (needs=[] --
    #: stock/control/record/audit), not a placeholder value.
    workload_id: str | None
    source_metadata_ref: ArtifactRef
    input_refs: tuple[tuple[str, ArtifactRef], ...]
    #: None when the build has no generate stage (build_cfg.variant_set is
    #: None -- stock).
    manifest_ref: ArtifactRef | None
    generated_tree_ref: ArtifactRef | None
    binary_ref: ArtifactRef
    #: RE07/RV48: the full runtime .so closure + effective-build identity,
    #: published as its own immutable artifact -- downstream consumers
    #: (smoke/benchmark/release) should read THIS, not binary_ref alone,
    #: since binary_ref names only the launcher and says nothing about
    #: whether the dependent libraries it actually loads are the ones
    #: this build produced.
    runtime_bundle_ref: ArtifactRef
    #: None when spec.validation is None -- no runtime-smoke stage ran.
    smoke_ref: ArtifactRef | None

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
    stage's output ordering or count changing.
    """
    matches = tuple(ref for ref in refs if ref.kind == kind)
    if len(matches) != 1:
        raise CampaignLaneError(
            f"stage {stage_id!r} expected exactly one {kind!r} artifact, "
            f"found {len(matches)}"
        )
    return matches[0]


def _optional_one_artifact(
    refs: tuple[ArtifactRef, ...] | None, *, kind: str, stage_id: str,
) -> ArtifactRef | None:
    """RE17: a stage that may not exist in the graph at all (generate for
    stock, runtime-smoke when validation is None) has no ``refs`` to look
    in -- distinct from a stage that ran but produced the wrong thing,
    which still fails via _require_one_artifact.
    """
    if refs is None:
        return None
    return _require_one_artifact(refs, kind=kind, stage_id=stage_id)


def _materialize_worker(
    context: ProjectContext, source_plan, *, allow_dirty_bigcherry: bool,
) -> dict:
    return campaign_build.materialize_source(
        context, source_plan, allow_dirty_bigcherry=allow_dirty_bigcherry)


def _execute_materialize_phase(
    spec: CampaignLaneExecutionSpec, *,
    cfg: campaign_config.Config, context: ProjectContext, store: ArtifactStore,
    run_id: str, campaign_root: Path, resource_root: Path,
    allow_dirty_bigcherry: bool = False,
) -> _MaterializedLaneSource:
    # GPT-auto-agent review (RE03/RE05 follow-up, 2026-08-17): without an
    # explicit catalog, source_plan_for()/resolve_lane() fall back to the
    # real project's default patch directory (paths.PATCHES), NOT this
    # context's patches_root -- two authorities in one execution. A
    # context rooted at a non-default checkout (isolated tests today; any
    # real non-default patches_root deployment) would have its logical
    # patch-set resolution silently validated against the wrong catalog.
    source_plan = campaign_source.source_plan_for(
        cfg, spec.source_name, catalog=patchset.catalog(directory=context.patches_root))
    resolved_revision = UpstreamRepository(context.upstream_repo).resolve_ref(
        source_plan.upstream_revision)
    source_plan = replace(source_plan, upstream_revision=resolved_revision)
    source_root = context.work_root / "sources" / campaign_source.materialization_plan_id(
        campaign_source.resolve_materialization_identity(context, source_plan))

    graph = campaign_plan.materialize_stage_graph(
        source_name=spec.source_name, build_name=spec.build_name,
        upstream_repo_path=str(context.upstream_repo))
    source_slice_id_holder: list[str | None] = [None]

    executor = CampaignStageExecutor(
        graph=graph, store=store, run_id=run_id,
        materialize=lambda: _materialize_worker(
            context, source_plan, allow_dirty_bigcherry=allow_dirty_bigcherry),
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


def _spec_inputs(spec: CampaignLaneExecutionSpec) -> dict[str, LaneInputValue]:
    result: dict[str, LaneInputValue] = {}
    for name, value in spec.inputs:
        if name in result:
            raise CampaignLaneError(f"duplicate lane input {name!r}")
        result[name] = value
    return result


def _sniff_embedded_provenance(data: bytes) -> dict[str, object] | None:
    """Deliberately always returns ``None`` -- kept as a named seam (not
    deleted outright) documenting why a raw-Path input's bytes are NEVER
    trusted to self-report their own provenance.

    An earlier version of this function accepted any payload that merely
    happened to parse as JSON with ``schema_version == 2`` and five dict
    namespaces (``provenance.validate()``'s own shape check) as "real
    embedded provenance". GPT-auto-agent review (RV48 follow-up,
    2026-08-17) correctly identified this as still a provenance-laundering
    route, just a smaller one: ``provenance.validate()`` is structural
    only -- it has no way to prove a document was genuinely produced by
    THIS project rather than hand-crafted by whoever supplied the raw
    file. A crafted JSON blob asserting any ``source_slice_id`` it likes
    would have been accepted as "real" here. Verifying a claimed embedded
    identity for real needs a chain-of-custody primitive this project
    does not have yet (RE25b's ArtifactDescriptor persistence/
    rehydration -- checking the claimed content against what THIS
    project's own ArtifactStore actually recorded for it, not just
    trusting the shape of the bytes). Until that lands, every raw-Path
    input is unconditionally imported-legacy -- see _resolve_lane_inputs.
    """
    return None


def _resolve_lane_inputs(
    spec: CampaignLaneExecutionSpec, *,
    build: campaign_config.Build, store: ArtifactStore,
    source_slice_id: str, run_id: str,
) -> dict[str, ArtifactRef]:
    """Generalizes the previous single-purpose _publish_inventory(): every
    lane input, keyed by name, resolved to a verified ArtifactRef.

    Exact set equality against build.needs, not a subset check -- a
    caller that supplies an extra, unrequested input is as much a bug as
    one that's missing a required one. Content-addressed publication
    (inputs/<name>/<sha256>), not a fixed path: ArtifactStore is immutable,
    so a fixed path cannot survive two lanes in one store using different
    bytes for the same need name.
    """
    provided = _spec_inputs(spec)
    actual = frozenset(provided)
    required = build.needs
    if actual != required:
        missing = sorted(required - actual)
        unexpected = sorted(actual - required)
        raise CampaignLaneError(
            f"build {build.name!r} input mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    if unsupported := (actual - campaign_workers.SUPPORTED_BUILD_NEEDS):
        raise CampaignLaneError(
            f"build {build.name!r} declares need(s) this lane executor "
            f"cannot yet represent in BuildPlan identity: {sorted(unsupported)}"
        )

    resolved: dict[str, ArtifactRef] = {}
    for name, value in provided.items():
        if isinstance(value, ArtifactRef):
            if value.kind != name:
                raise CampaignLaneError(
                    f"lane input {name!r} has artifact kind {value.kind!r}"
                )
            try:
                relative = value.path.resolve().relative_to(store.root)
            except ValueError as exc:
                raise CampaignLaneError(
                    f"lane input {name!r} is not owned by this campaign's "
                    f"ArtifactStore ({store.root})"
                ) from exc
            if not store.verify(relative, value.content_hash):
                raise CampaignLaneError(
                    f"lane input {name!r} bytes do not match its own content_hash"
                )
            resolved[name] = value
            continue

        data = value.read_bytes()
        digest = ArtifactStore.digest(data)
        relative = f"inputs/{name}/{digest}"
        published_digest = store.publish_bytes(relative, data)
        if published_digest != digest:
            raise CampaignLaneError(f"lane input {name!r} digest changed during publication")
        # RE08/RV48 audit fix: a raw Path has no producer identity of its
        # own to trust -- the caller merely handed us a file while building
        # THIS source, which is not proof the file was ever produced BY
        # this source (an inventory from source A, supplied while building
        # source B, must not silently acquire source-B provenance). Prefer
        # the file's own embedded provenance if it happens to carry one
        # (a real producer, just passed as a path instead of an
        # ArtifactRef); otherwise stamp it visibly as imported-legacy
        # evidence rather than inventing an unearned claim.
        embedded = _sniff_embedded_provenance(data)
        if embedded is not None:
            doc = embedded
        else:
            doc = make_provenance(
                project={"provenance_class": "imported-legacy"},
                source={"source_slice_id": source_slice_id}, build={},
                workload={}, campaign={"run_id": run_id})
        resolved[name] = ArtifactRef(
            kind=name, path=store.resolve(relative), content_hash=digest, provenance=doc)

    return resolved


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

    input_refs = _resolve_lane_inputs(
        spec, build=build_cfg, store=store,
        source_slice_id=materialized.source_slice_id, run_id=run_id)
    inventory_ref = input_refs.get("inventory")
    # workload_id is deterministically the inventory's own content_hash --
    # knowable before generate ever runs, once the inventory is published.
    # NEVER a hash of inventory+winners combined: winners already has its
    # own BuildPlan.winners_hash slot, which independently feeds
    # build_plan_id -- a changed promoted-winners artifact changes
    # build_plan_id, not workload_id.
    workload_id = inventory_ref.content_hash if inventory_ref is not None else None

    has_generate = build_cfg.variant_set is not None
    has_smoke = spec.validation is not None

    build_plan = BuildPlan(
        source_slice_id=materialized.source_slice_id, phase=spec.build_name,
        platform=platform_cfg.name, targets=platform_cfg.targets,
        cmake_options=tuple(sorted(merged_options.items())),
        variant_set=build_cfg.variant_set,
        # RE07/RV48: empty for a build with no generate stage -- there is
        # no catalog to disambiguate. Otherwise the exact architecture set
        # generation was actually requested for (not platform_cfg.targets,
        # which is a different axis: compiled AMDGPU targets).
        catalog_architectures=tuple(sorted(spec.architectures)) if has_generate else (),
        # Generic over every declared need this lane resolved -- see
        # BuildPlan's own docstring for why this replaced the old
        # inventory_hash/winners_hash pair.
        input_hashes=tuple(sorted(
            (name, ref.content_hash) for name, ref in input_refs.items())),
        toolchain_request=campaign_build.toolchain_request_for_platform(platform_cfg),
        environment=campaign_build.resolve_build_environment(),
    )

    build_graph = campaign_plan.build_stage_graph(
        source_name=spec.source_name, build_name=spec.build_name,
        source_slice_id=materialized.source_slice_id, build_plan_id=build_plan.build_plan_id,
        workload_id=workload_id, include_generate=has_generate,
        include_runtime_smoke=has_smoke, gpu_resource_ids=spec.gpu_resource_ids,
    )

    generate_worker = None
    if has_generate:
        generate_worker = campaign_workers.make_generate_worker(
            context=context, source_root=materialized.source_root, run_id=run_id,
            variant_set=build_cfg.variant_set, architectures=list(spec.architectures),
            upstream_revision=materialized.resolved_revision,
            required_needs=build_cfg.needs,
        )
    build_worker = campaign_workers.make_build_worker(
        context=context, source_root=materialized.source_root, run_id=run_id,
        build_plan=build_plan, platform=platform_cfg, build=build_cfg,
        store=store, binary_relative_path=spec.binary_relative_path,
        source_slice_id=materialized.source_slice_id, workload_id=workload_id,
        lane_inputs=input_refs, has_generate_stage=has_generate,
        cmake_targets=(Path(spec.binary_relative_path).name,),
    )
    smoke_worker = None
    if has_smoke:
        smoke_worker = campaign_workers.make_smoke_worker(
            run_id=run_id, store=store, source_slice_id=materialized.source_slice_id,
            build_plan_id=build_plan.build_plan_id, workload_id=workload_id,
            spec=spec.validation,
            environment=None if spec.smoke_environment is None else dict(spec.smoke_environment),
        )

    executor = CampaignStageExecutor(
        graph=build_graph, store=store, run_id=run_id,
        materialize=lambda: {}, generate=generate_worker,
        generate_inputs=input_refs if has_generate else None,
        generate_needs=build_cfg.needs if has_generate else frozenset(),
        source_slice_id_holder=[materialized.source_slice_id],
        build_plan_id=build_plan.build_plan_id,
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

    generate_outputs = executor.outputs.get(generate_stage_id)
    manifest_ref = _optional_one_artifact(
        generate_outputs, kind="manifest", stage_id=generate_stage_id)
    generated_tree_ref = _optional_one_artifact(
        generate_outputs, kind="generated-tree", stage_id=generate_stage_id)
    binary_ref = _require_one_artifact(
        executor.outputs[build_stage_id], kind="binary", stage_id=build_stage_id)
    runtime_bundle_ref = _require_one_artifact(
        executor.outputs[build_stage_id], kind="runtime-bundle", stage_id=build_stage_id)
    smoke_ref = _optional_one_artifact(
        executor.outputs.get(smoke_stage_id), kind="smoke-result", stage_id=smoke_stage_id)

    return CampaignLaneResult(
        run_id=run_id, resolved_revision=materialized.resolved_revision,
        source_slice_id=materialized.source_slice_id, build_plan=build_plan,
        workload_id=workload_id, source_metadata_ref=materialized.source_metadata_ref,
        input_refs=tuple(sorted(input_refs.items())),
        manifest_ref=manifest_ref, generated_tree_ref=generated_tree_ref,
        binary_ref=binary_ref, runtime_bundle_ref=runtime_bundle_ref, smoke_ref=smoke_ref,
    )


def execute_campaign_lane(
    spec: CampaignLaneExecutionSpec, *,
    cfg: campaign_config.Config, context: ProjectContext, store: ArtifactStore,
    run_id: str | None = None,
    allow_dirty_bigcherry: bool = False,
) -> CampaignLaneResult:
    """Execute one campaign lane: materialize, then generate/build/smoke.
    Each graph runs exactly once -- no automatic reuse-proof second pass
    (see module docstring). Raises CampaignLaneError on any failure.

    ``allow_dirty_bigcherry`` (RE04/RV48 audit fix) defaults to False: a
    production caller of this public API runs against a clean BigCherry
    checkout. Only a development/harness caller (re14_real_run.py, RE14's
    own proof/parity tooling) has a real reason to pass True -- it must
    never be an implicit default baked into the library function itself.
    """
    effective_run_id = run_id or uuid.uuid4().hex[:12]
    campaign_root = context.work_root / "campaign-runs" / effective_run_id
    resource_root = context.work_root / "resource-locks"

    materialized = _execute_materialize_phase(
        spec, cfg=cfg, context=context, store=store, run_id=effective_run_id,
        campaign_root=campaign_root, resource_root=resource_root,
        allow_dirty_bigcherry=allow_dirty_bigcherry,
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
