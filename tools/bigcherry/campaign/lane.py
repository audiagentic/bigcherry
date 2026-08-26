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

import json
import os
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from . import build as campaign_build
from . import plan as campaign_plan
from . import source as campaign_source
from . import workers as campaign_workers
from ..core import config as campaign_config
from ..patch import catalog as patch_catalog
from ..patch import patchset
from . import smoke as smoke_module
from ..core.artifacts import ArtifactDescriptor, ArtifactLocator, ArtifactStore
from ..build.builds import BuildPlan
from .campaign import CampaignRun
from .execution import (
    CampaignExecutionError,
    CampaignStageExecutor,
    make_artifact_reuse_checker,
    require_campaign_success,
    source_provenance_from_metadata,
)
from ..core.context import ProjectContext
from ..core.pipeline import ArtifactRef
from ..core.provenance import (
    BuildProvenance,
    CampaignProvenance,
    ProvenanceClass,
    ProvenanceError,
    ProvenanceV2,
    ProjectProvenance,
    SCHEMA_VERSION,
    SourceProvenance,
    WorkloadProvenance,
)
from ..source.workspace import UpstreamRepository, WorkspaceError, bigcherry_revision


class CampaignLaneError(RuntimeError):
    pass


#: A raw path (the CLI importing a file, e.g. --inventory), an
#: already-published ArtifactRef (a future canonical-lane caller, RE18), or
#: an ArtifactLocator -- the small cross-process handle naming an artifact
#: by its descriptor identity alone (RE25.3). All three normalize to a
#: verified, descriptor-backed ArtifactRef in _resolve_lane_inputs()
#: before any stage runs.
LaneInputValue = Path | ArtifactRef | ArtifactLocator


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
    #: RE26: additional executables to build from the same configure and
    #: publish into the same runtime bundle as binary_relative_path (e.g.
    #: test-backend-ops alongside the tune lane's llama-bench) -- names
    #: only, resolved relative to binary_relative_path's own directory.
    extra_cmake_targets: tuple[str, ...] = ()
    #: RE26 prep: names a [experiment.<name>] entry in recipes.toml, per
    #: campaign_resolution.resolve_lane's own experiment= parameter -- for
    #: benching one experimental patch in isolation. None means "just the
    #: source's normal patch-set", identical to today's behavior.
    experiment: str | None = None
    #: EC04: Experiment Contract lane metadata (tools/bigcherry/
    #: experiment_contract.py). Deliberately provenance-only -- these five
    #: fields must NEVER be read by anything that computes build_plan_id,
    #: effective_build_id, source_slice_id, or a runtime signature/dispatch
    #: digest (see the Experiment Contract guide's non-negotiable
    #: architecture: "keep runtime candidate identity separate from source
    #: provenance and from experiment identity"). _to_spec()/materialize/
    #: generate/build/smoke never read them; they exist purely so EC05's
    #: evidence binding and EC10's reporting know which part of a contract
    #: a given lane's result is proving. contract_id/optimization_id name
    #: the ExperimentContract.id and .source.atomic_part; role is one of
    #: "positive"/"control"/"boundary"/"holdout"; workload_tag is one of
    #: experiment_contract.WORKLOAD_TAGS; model_ref is a free-form
    #: recipe/model reference string, not a runtime dispatch identity.
    contract_id: str | None = None
    optimization_id: str | None = None
    role: str | None = None
    workload_tag: str | None = None
    model_ref: str | None = None
    #: Set only when role == "boundary": which swept dimension (e.g.
    #: "physical_m", from the contract's boundary.dimensions) and which
    #: single value along it this lane probes. Stored as a string
    #: regardless of the dimension's native type (int/float/str in the
    #: contract) -- this is metadata for reporting, never parsed back into
    #: a typed value anything dispatch-relevant reads.
    boundary_dimension: str | None = None
    boundary_value: str | None = None


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

    @property
    def effective_build_id(self) -> str | None:
        """RE25.2: the runtime bundle's recorded effective-build identity,
        read from its own provenance rather than re-parsing the bundle
        manifest JSON -- RE09's v4 campaign-identity boundary needs exactly
        this and previously had no way to get it without a second parse.
        None when the build worker did not record one (e.g. a test fake)."""
        build = self.runtime_bundle_ref.provenance.get("build")
        if not isinstance(build, dict):
            return None
        value = build.get("effective_build_id")
        return value if isinstance(value, str) else None


@dataclass(frozen=True)
class _MaterializedLaneSource:
    resolved_revision: str
    source_slice_id: str
    source_root: Path
    source_metadata_ref: ArtifactRef
    #: RE25.2: the typed SourceProvenance re-derived from the published
    #: source-metadata artifact's own (verified) bytes -- the same record
    #: materialize_source() returned, so every stage in this lane carries
    #: the full real source lineage downstream.
    source_provenance: SourceProvenance
    #: RE30 P0-3 (external review, 2026-08-20): the exact composed patch
    #: selection resolve_lane() already approved for this source -- carried
    #: forward so the build phase can validate it against the REAL backend
    #: and cmake options a lane is about to compile with, not just the
    #: patch-set-name policy resolve_lane() already checked at materialize
    #: time (before backend-injected options like GGML_VULKAN=ON exist).
    patch_ids: tuple[str, ...] = ()


def _require_one_artifact(
    refs: tuple[ArtifactRef, ...],
    *,
    kind: str,
    stage_id: str,
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
    refs: tuple[ArtifactRef, ...] | None,
    *,
    kind: str,
    stage_id: str,
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
    context: ProjectContext,
    source_plan,
    *,
    allow_dirty_bigcherry: bool,
) -> dict:
    return campaign_build.materialize_source(
        context, source_plan, allow_dirty_bigcherry=allow_dirty_bigcherry
    )


def _execute_materialize_phase(
    spec: CampaignLaneExecutionSpec,
    *,
    cfg: campaign_config.Config,
    context: ProjectContext,
    store: ArtifactStore,
    run_id: str,
    campaign_root: Path,
    resource_root: Path,
    allow_dirty_bigcherry: bool = False,
    project_revision: str,
    local_provenance_class: ProvenanceClass,
) -> _MaterializedLaneSource:
    # GPT-auto-agent review (RE03/RE05 follow-up, 2026-08-17): without an
    # explicit catalog, source_plan_for()/resolve_lane() fall back to the
    # real project's default patch directory (paths.PATCHES), NOT this
    # context's patches_root -- two authorities in one execution. A
    # context rooted at a non-default checkout (isolated tests today; any
    # real non-default patches_root deployment) would have its logical
    # patch-set resolution silently validated against the wrong catalog.
    source_plan = campaign_source.source_plan_for(
        cfg,
        spec.source_name,
        catalog=patchset.catalog(directory=context.patches_root),
        # Explicit, not just inferred from the catalog's own first entry:
        # a genuinely empty context.patches_root (zero patch files) would
        # otherwise lose this directory entirely and silently fall back to
        # the wrong global default (GPT-auto-agent review, 2026-08-17).
        catalog_directory=context.patches_root,
        experiment=spec.experiment,
    )
    resolved_revision = UpstreamRepository(context.upstream_repo).resolve_ref(
        source_plan.upstream_revision
    )
    source_plan = replace(source_plan, upstream_revision=resolved_revision)
    source_root = (
        context.work_root
        / "sources"
        / campaign_source.materialization_plan_id(
            campaign_source.resolve_materialization_identity(context, source_plan)
        )
    )

    graph = campaign_plan.materialize_stage_graph(
        source_name=spec.source_name,
        build_name=spec.build_name,
        upstream_repo_path=str(context.upstream_repo),
    )
    source_slice_id_holder: list[str | None] = [None]

    executor = CampaignStageExecutor(
        graph=graph,
        store=store,
        run_id=run_id,
        materialize=lambda: _materialize_worker(
            context, source_plan, allow_dirty_bigcherry=allow_dirty_bigcherry
        ),
        generate=lambda inputs: {},
        source_slice_id_holder=source_slice_id_holder,
        project_revision=project_revision,
        local_provenance_class=local_provenance_class,
    )
    campaign_run = CampaignRun(graph, root=campaign_root / "materialize", run_id=run_id)
    reuse = make_artifact_reuse_checker(executor=executor, store=store)

    try:
        records = campaign_run.execute(
            executor, resource_root=resource_root, reuse=reuse
        )
        require_campaign_success(records, label="materialize")
    except CampaignExecutionError as exc:
        raise CampaignLaneError(f"materialize failed: {exc}") from exc

    source_slice_id = source_slice_id_holder[0]
    if not source_slice_id:
        raise CampaignLaneError(
            "materialize succeeded without resolving source_slice_id"
        )

    stage_id = f"{spec.source_name}:{spec.build_name}:materialize"
    source_metadata_ref = _require_one_artifact(
        executor.outputs[stage_id], kind="source-metadata", stage_id=stage_id
    )
    # RE25.2: build the lane's typed SourceProvenance from the PUBLISHED
    # artifact's own bytes (verified by ArtifactRef at publish time), not
    # from a second in-memory copy -- one source of truth, and it is the
    # exact record materialize_source() re-verified per RE04/RE05.
    metadata_doc = json.loads(source_metadata_ref.path.read_text(encoding="utf-8"))
    source_provenance = source_provenance_from_metadata(metadata_doc)

    return _MaterializedLaneSource(
        resolved_revision=resolved_revision,
        source_slice_id=source_slice_id,
        source_root=source_root,
        source_metadata_ref=source_metadata_ref,
        source_provenance=source_provenance,
        patch_ids=source_plan.patch_ids,
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
    would have been accepted as "real" here. The chain-of-custody
    primitive that closes the gap for good has since landed (RE25.1
    ArtifactDescriptor persistence + rehydration): a caller who KNOWS an
    artifact's descriptor identity passes an ArtifactLocator and
    _resolve_lane_inputs() rehydrates it against the store -- checking
    the claimed content against what THIS project actually recorded. A
    raw Path, however, still carries no claimed identity to verify, so
    every raw-Path input remains unconditionally imported-legacy.
    """
    return None


def _imported_legacy_document(
    run_id: str, producer_stage: str = "lane-import"
) -> ProvenanceV2:
    """RE25.3: the honest identity for evidence whose origin we cannot
    verify -- imported-legacy class with NO source/build/workload claims.

    The previous version stamped THIS lane's source_slice_id onto imported
    evidence, which is not proof of where the file was produced (an
    inventory from source A supplied while building source B must not
    acquire a source-B claim). The empty source namespace is deliberate:
    PipelineService._check_inputs() exempts imported-legacy inputs from the
    stage-envelope field check for exactly this reason, and the sticky
    class guarantees nothing derived from it can ever be promotable.
    """
    return ProvenanceV2(
        schema_version=SCHEMA_VERSION,
        project=ProjectProvenance(provenance_class="imported-legacy"),
        source=SourceProvenance(),
        build=BuildProvenance(),
        workload=WorkloadProvenance(),
        campaign=CampaignProvenance(run_id=run_id, producer_stage=producer_stage),
    )


def _resolve_lane_inputs(
    spec: CampaignLaneExecutionSpec,
    *,
    build: campaign_config.Build,
    store: ArtifactStore,
    run_id: str,
) -> dict[str, ArtifactRef]:
    """Generalizes the previous single-purpose _publish_inventory(): every
    lane input, keyed by name, resolved to a verified, descriptor-backed
    ArtifactRef (RE25.3).

    Exact set equality against build.needs, not a subset check -- a
    caller that supplies an extra, unrequested input is as much a bug as
    one that's missing a required one.

    Resolution by input kind:
    - ArtifactLocator: rehydrate by artifact_id (kind + provenance
      contract enforced), use the returned ref unchanged.
    - ArtifactRef WITH artifact_id: rehydrate by its ID and verify kind /
      content hash agree with the supplied ref -- a claimed descriptor
      identity is re-proven, never trusted (closes the RE25.2-review
      follow-on). Use the rehydrated ref.
    - ArtifactRef WITHOUT artifact_id: legacy evidence. Verify store
      ownership + bytes, then downgrade any non-imported provenance class
      to imported-legacy -- an unverified ref cannot be first-party
      production evidence (no laundering). Parseable docs keep their fields
      (so a wrong-source claim still fails the envelope); unparseable docs
      are replaced with a minimal imported-legacy identity.
    - raw Path: read/hash bytes, stamp a typed imported-legacy document
      with NO source claims, publish through the descriptor API so the
      input gets a real rehydratable artifact_id.
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
        if isinstance(value, ArtifactLocator):
            # The trusted cross-process path: rehydrate proves bytes +
            # provenance contract + kind against the store's own records.
            try:
                resolved[name] = store.rehydrate(value.artifact_id, expected_kind=name)
            except Exception as exc:
                raise CampaignLaneError(
                    f"lane input {name!r}: locator artifact_id "
                    f"{value.artifact_id!r} failed rehydration: {exc}"
                ) from exc
            continue

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
            if value.artifact_id:
                # Descriptor-backed: re-prove the claimed identity against
                # the store's persisted descriptor, then use the
                # rehydrated ref (its provenance is what the store
                # recorded, not what the in-memory ref happened to carry).
                try:
                    rehydrated = store.rehydrate(value.artifact_id, expected_kind=name)
                except Exception as exc:
                    raise CampaignLaneError(
                        f"lane input {name!r}: claimed artifact_id "
                        f"{value.artifact_id!r} failed rehydration: {exc}"
                    ) from exc
                if rehydrated.content_hash != value.content_hash:
                    raise CampaignLaneError(
                        f"lane input {name!r} content hash "
                        f"{value.content_hash!r} disagrees with its own "
                        f"descriptor's {rehydrated.content_hash!r}"
                    )
                resolved[name] = rehydrated
            else:
                # Legacy (no descriptor identity): legacy/imported evidence
                # only -- downgrade any first-party class claim, never let
                # an unverified ref launder itself into production.
                try:
                    parsed = ProvenanceV2.from_document(value.provenance)
                except ProvenanceError:
                    # Unparseable/malformed doc: nothing verifiable in it.
                    parsed = _imported_legacy_document(run_id)
                if parsed.project.provenance_class != "imported-legacy":
                    parsed = ProvenanceV2(
                        schema_version=parsed.schema_version,
                        project=ProjectProvenance(
                            provenance_class="imported-legacy",
                            bigcherry_revision=parsed.project.bigcherry_revision,
                        ),
                        source=parsed.source,
                        build=parsed.build,
                        workload=parsed.workload,
                        campaign=parsed.campaign,
                    )
                # GPT audit fix (2026-08-18): the locked RE25.3 contract is
                # that EVERY lane input normalizes to a descriptor-backed
                # ref -- this branch used to downgrade the class but leave
                # artifact_id="", keeping the content_hash-as-identity
                # fallback alive exactly where it was supposed to die.
                # Persist a descriptor for the ALREADY-VERIFIED bytes at
                # value.path (persist_descriptor re-verifies them, so no new
                # trust is introduced) and return the ref with its real id.
                # The kind contract needs no separate check here: the doc
                # is imported-legacy by construction, and
                # validate_for_kind imposes no production fields on that
                # class (same as the raw-Path branch's _publish_ref gate).
                try:
                    descriptor = ArtifactDescriptor.create(
                        kind=name,
                        relative_path=str(Path(relative).as_posix()),
                        content_hash=value.content_hash,
                        provenance=parsed,
                    )
                    store.persist_descriptor(descriptor)
                except Exception as exc:
                    raise CampaignLaneError(
                        f"lane input {name!r} failed descriptor persistence: {exc}"
                    ) from exc
                resolved[name] = replace(
                    value,
                    provenance=parsed.document(),
                    artifact_id=descriptor.artifact_id,
                )
            continue

        # raw Path: imported-legacy, descriptor-backed, no source claims.
        data = value.read_bytes()
        digest = ArtifactStore.digest(data)
        relative = f"inputs/{name}/{digest}"
        embedded = _sniff_embedded_provenance(data)
        if embedded is not None:
            # Seam is always-None today (see its docstring); kept so a
            # future verified path flows through the same contract.
            try:
                parsed = ProvenanceV2.from_document(embedded)
            except ProvenanceError as exc:
                raise CampaignLaneError(
                    f"lane input {name!r}: embedded provenance is not a "
                    f"valid schema-v2 document: {exc}"
                ) from exc
        else:
            parsed = _imported_legacy_document(run_id)
        try:
            resolved[name] = store.publish_bytes_ref(
                relative, data, kind=name, provenance=parsed
            )
        except Exception as exc:
            raise CampaignLaneError(
                f"lane input {name!r} failed descriptor publication: {exc}"
            ) from exc

    return resolved


def _execute_build_phase(
    spec: CampaignLaneExecutionSpec,
    *,
    cfg: campaign_config.Config,
    context: ProjectContext,
    store: ArtifactStore,
    run_id: str,
    campaign_root: Path,
    resource_root: Path,
    materialized: _MaterializedLaneSource,
    project_revision: str,
    local_provenance_class: ProvenanceClass,
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
    # RE-backend-identity (external review, 2026-08-20): backend must be
    # part of the SAME options map that becomes both BuildPlan.cmake_options
    # (requested identity) and the real cmake_configure_args() call below
    # (what actually gets compiled) -- previously these were two
    # independently-computed maps, so BuildPlan.cmake_options never saw the
    # backend-injected options (GGML_VULKAN=ON / AMDGPU_TARGETS=...) unless
    # a source's static recipes.toml options happened to already declare
    # them. A HIP and a Vulkan request sharing the same nominal source/
    # build/platform names but relying on backend injection (rather than
    # static build.options) for GGML_VULKAN/AMDGPU_TARGETS could otherwise
    # produce the SAME build_plan_id while compiling different commands.
    backend = cfg.sources[spec.source_name].backend
    merged_options = dict(platform_cfg.options)
    merged_options.update(dict(build_cfg.options))
    merged_options.update(campaign_build._backend_configure_options(backend, platform_cfg))

    # RE30 P0-3 (external review, 2026-08-20): resolve_lane() already
    # enforced patch-set-name/state policy at materialize time, but never
    # checked an individual patch's declared backend/requires-options/
    # forbids-options (patch_catalog.CatalogEntry) against what this build
    # is ACTUALLY about to compile with -- the real backend and the real
    # merged cmake options, only known here. A selected-but-inapplicable
    # patch (e.g. a hip-only patch on a vulkan source, or one requiring an
    # option this lane never sets) is a hard error, not a silent skip.
    # Resolved against THIS context's own catalog.toml (matching the
    # patches_root already used for resolve_lane() at materialize time,
    # not paths.PATCH_CATALOG's real-project default) -- an isolated test
    # context with its own synthetic patches directory and no descriptive
    # catalog layer at all has nothing to check applicability against, and
    # must not be validated against the unrelated real project's catalog.
    lane_catalog_path = context.patches_root / "catalog.toml"
    if materialized.patch_ids and lane_catalog_path.is_file():
        try:
            patch_catalog.resolve_for_context(
                materialized.patch_ids,
                patch_catalog.PatchContext(
                    backend=backend,
                    source=spec.source_name,
                    build=spec.build_name,
                    platform=platform_cfg.name,
                    options=tuple(sorted(merged_options)),
                ),
                catalog_path=lane_catalog_path,
                # HI102 (adversarial-review follow-up): the only place this
                # call chain actually knows the LIVE resolved base revision
                # -- resolve_for_context() itself has no source-root access,
                # so a caller who materialized the source must supply it.
                resolved_base_revision=materialized.resolved_revision,
            )
        except ValueError as exc:
            raise CampaignLaneError(f"patch applicability check failed: {exc}") from exc

    input_refs = _resolve_lane_inputs(spec, build=build_cfg, store=store, run_id=run_id)
    inventory_ref = input_refs.get("inventory")
    # workload_id is deterministically the inventory's own content_hash --
    # knowable before generate ever runs, once the inventory is published.
    # NEVER a hash of inventory+winners combined: winners already has its
    # own BuildPlan.winners_hash slot, which independently feeds
    # build_plan_id -- a changed promoted-winners artifact changes
    # build_plan_id, not workload_id.
    workload_id = inventory_ref.content_hash if inventory_ref is not None else None

    # Walrus-narrowed, not just a separate boolean: pyright cannot carry a
    # `build_cfg.variant_set is not None` fact through a distinct
    # ``has_generate`` variable into the guarded call sites below, so the
    # non-None proof has to live in the value itself.
    variant_set = build_cfg.variant_set
    has_smoke = spec.validation is not None

    build_plan = BuildPlan(
        source_slice_id=materialized.source_slice_id,
        phase=spec.build_name,
        platform=platform_cfg.name,
        targets=platform_cfg.targets,
        cmake_options=tuple(sorted(merged_options.items())),
        backend=backend,
        variant_set=build_cfg.variant_set,
        # RE07/RV48: empty for a build with no generate stage -- there is
        # no catalog to disambiguate. Otherwise the exact architecture set
        # generation was actually requested for (not platform_cfg.targets,
        # which is a different axis: compiled AMDGPU targets).
        catalog_architectures=tuple(sorted(spec.architectures))
        if variant_set is not None
        else (),
        # HI110: the real requested output set -- see BuildPlan's own
        # docstring for why this must participate in build_plan_id.
        requested_targets=tuple(sorted(
            {Path(spec.binary_relative_path).name, *spec.extra_cmake_targets}
        )),
        # Generic over every declared need this lane resolved -- see
        # BuildPlan's own docstring for why this replaced the old
        # inventory_hash/winners_hash pair.
        input_hashes=tuple(
            sorted((name, ref.content_hash) for name, ref in input_refs.items())
        ),
        toolchain_request=campaign_build.toolchain_request_for_platform(platform_cfg),
        environment=campaign_build.resolve_build_environment(),
    )

    build_graph = campaign_plan.build_stage_graph(
        source_name=spec.source_name,
        build_name=spec.build_name,
        source_slice_id=materialized.source_slice_id,
        build_plan_id=build_plan.build_plan_id,
        workload_id=workload_id,
        include_generate=variant_set is not None,
        include_runtime_smoke=has_smoke,
        gpu_resource_ids=spec.gpu_resource_ids,
    )

    generate_worker = None
    if variant_set is not None:
        generate_worker = campaign_workers.make_generate_worker(
            context=context,
            source_root=materialized.source_root,
            run_id=run_id,
            variant_set=variant_set,
            architectures=list(spec.architectures),
            upstream_revision=materialized.resolved_revision,
            required_needs=build_cfg.needs,
        )
    build_worker = campaign_workers.make_build_worker(
        context=context,
        source_root=materialized.source_root,
        run_id=run_id,
        build_plan=build_plan,
        platform=platform_cfg,
        build=build_cfg,
        store=store,
        binary_relative_path=spec.binary_relative_path,
        source_slice_id=materialized.source_slice_id,
        workload_id=workload_id,
        lane_inputs=input_refs,
        has_generate_stage=variant_set is not None,
        cmake_targets=(Path(spec.binary_relative_path).name, *spec.extra_cmake_targets),
        extra_binary_names=spec.extra_cmake_targets,
        # RE25.2: full typed lineage for the artifacts this worker publishes
        # -- the same source provenance and project identity every other
        # stage in this lane carries.
        source_provenance=materialized.source_provenance,
        project_revision=project_revision,
        local_provenance_class=local_provenance_class,
        backend=cfg.sources[spec.source_name].backend,
    )
    smoke_worker = None
    validation_spec = spec.validation
    if has_smoke and validation_spec is not None:
        smoke_worker = campaign_workers.make_smoke_worker(
            run_id=run_id,
            store=store,
            source_slice_id=materialized.source_slice_id,
            build_plan_id=build_plan.build_plan_id,
            workload_id=workload_id,
            spec=validation_spec,
            environment=None
            if spec.smoke_environment is None
            else dict(spec.smoke_environment),
            source_provenance=materialized.source_provenance,
            project_revision=project_revision,
            local_provenance_class=local_provenance_class,
        )

    executor = CampaignStageExecutor(
        graph=build_graph,
        store=store,
        run_id=run_id,
        materialize=lambda: {},
        generate=generate_worker,
        generate_inputs=input_refs if variant_set is not None else None,
        generate_needs=build_cfg.needs if variant_set is not None else frozenset(),
        source_slice_id_holder=[materialized.source_slice_id],
        build_plan_id=build_plan.build_plan_id,
        workload_id=workload_id,
        build=build_worker,
        smoke=smoke_worker,
        project_revision=project_revision,
        local_provenance_class=local_provenance_class,
        initial_source_provenance=materialized.source_provenance,
    )
    campaign_run = CampaignRun(build_graph, root=campaign_root / "build", run_id=run_id)
    reuse = make_artifact_reuse_checker(executor=executor, store=store)

    try:
        records = campaign_run.execute(
            executor, resource_root=resource_root, reuse=reuse
        )
        require_campaign_success(records, label="generate/build/runtime-smoke")
    except CampaignExecutionError as exc:
        raise CampaignLaneError(f"generate/build/runtime-smoke failed: {exc}") from exc

    generate_stage_id = f"{spec.source_name}:{spec.build_name}:generate"
    build_stage_id = f"{spec.source_name}:{spec.build_name}:build"
    smoke_stage_id = f"{spec.source_name}:{spec.build_name}:runtime-smoke"

    generate_outputs = executor.outputs.get(generate_stage_id)
    manifest_ref = _optional_one_artifact(
        generate_outputs, kind="manifest", stage_id=generate_stage_id
    )
    generated_tree_ref = _optional_one_artifact(
        generate_outputs, kind="generated-tree", stage_id=generate_stage_id
    )
    binary_ref = _require_one_artifact(
        executor.outputs[build_stage_id], kind="binary", stage_id=build_stage_id
    )
    runtime_bundle_ref = _require_one_artifact(
        executor.outputs[build_stage_id], kind="runtime-bundle", stage_id=build_stage_id
    )
    smoke_ref = _optional_one_artifact(
        executor.outputs.get(smoke_stage_id),
        kind="smoke-result",
        stage_id=smoke_stage_id,
    )

    return CampaignLaneResult(
        run_id=run_id,
        resolved_revision=materialized.resolved_revision,
        source_slice_id=materialized.source_slice_id,
        build_plan=build_plan,
        workload_id=workload_id,
        source_metadata_ref=materialized.source_metadata_ref,
        input_refs=tuple(sorted(input_refs.items())),
        manifest_ref=manifest_ref,
        generated_tree_ref=generated_tree_ref,
        binary_ref=binary_ref,
        runtime_bundle_ref=runtime_bundle_ref,
        smoke_ref=smoke_ref,
    )


def execute_campaign_lane(
    spec: CampaignLaneExecutionSpec,
    *,
    cfg: campaign_config.Config,
    context: ProjectContext,
    store: ArtifactStore,
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

    # RE25.2: the executing checkout's own revision, and the honest
    # provenance class of this execution -- a dirty-tree development run is
    # 'development' end to end (never 'production'), not something callers
    # remember to stamp on individual artifacts by hand.
    local_provenance_class: ProvenanceClass = (
        "development" if allow_dirty_bigcherry else "production"
    )
    try:
        project_revision = bigcherry_revision(context)
    except WorkspaceError as exc:
        # Production mode cannot get here with a non-git project_root:
        # require_clean_bigcherry (inside materialize_source) already ran
        # `git status` there and would have failed first. So a failure HERE
        # in production is a real contradiction -- fail closed on it.
        if local_provenance_class == "production":
            raise CampaignLaneError(
                f"cannot resolve the BigCherry project revision at "
                f"{context.project_root}: {exc}"
            ) from exc
        # Development/harness mode with a synthetic (non-git) project root:
        # no revision exists to record -- bigcherry_revision stays None in
        # provenance rather than being invented. The 'development' class
        # already marks the whole run as non-promotable.
        project_revision = ""

    materialized = _execute_materialize_phase(
        spec,
        cfg=cfg,
        context=context,
        store=store,
        run_id=effective_run_id,
        campaign_root=campaign_root,
        resource_root=resource_root,
        allow_dirty_bigcherry=allow_dirty_bigcherry,
        project_revision=project_revision,
        local_provenance_class=local_provenance_class,
    )
    return _execute_build_phase(
        spec,
        cfg=cfg,
        context=context,
        store=store,
        run_id=effective_run_id,
        campaign_root=campaign_root,
        resource_root=resource_root,
        materialized=materialized,
        project_revision=project_revision,
        local_provenance_class=local_provenance_class,
    )


def smoke_environment_for_hip_devices(
    hip_visible_devices: str,
) -> tuple[tuple[str, str], ...]:
    """The CLI-convenience translation from a single --hip-visible-devices
    flag to the environment dict the smoke worker actually accepts --
    kept out of the reusable API itself (make_smoke_worker's own environment
    parameter is HIP-agnostic; this is a CLI-specific convenience).

    RE15 real-hardware finding (2026-08-18): HIP_VISIBLE_DEVICES and
    ROCR_VISIBLE_DEVICES are NOT aliases -- ROCR_VISIBLE_DEVICES filters the
    ROCm agent list first, and HIP_VISIBLE_DEVICES then indexes into
    whatever that filtered list left. Setting both to the SAME raw device
    index double-filters: on a real 4-GPU Brutus box, HIP_VISIBLE_DEVICES=2
    ROCR_VISIBLE_DEVICES=2 selects agent 2 (gfx1201), which becomes the
    sole entry in the ROCR-filtered list, and then HIP_VISIBLE_DEVICES=2
    tries to index into that single-entry list and fails with "no
    ROCm-capable device is detected" -- verified this only breaks for a
    nonzero index (index 0 degenerately still resolves), which is why
    every prior real run here (always device 0) never surfaced it. Only
    HIP_VISIBLE_DEVICES is set now."""
    return tuple(
        sorted(
            {
                "HIP_VISIBLE_DEVICES": hip_visible_devices,
                "PATH": os.environ.get("PATH", ""),
            }.items()
        )
    )
