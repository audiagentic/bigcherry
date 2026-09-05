"""RE18: canonical CampaignLane planner and sequential multi-lane runner.

The small outer layer this round adds on top of execute_campaign_lane()
(RE16/RE17): expand a campaign request into concrete lanes, then run each
one sequentially. Deliberately NOT a unification of multiple lanes into one
CampaignGraph -- campaign_plan.py's own docstring documents materialize as
an identity-resolution boundary (source_slice_id does not exist until
materialize actually runs), and RE14's own step ordering only requires
sequential execution to be correct, not parallel or graph-unified. And
deliberately built over canonical v2 identities (config.Source/Build/
Platform names, config.CampaignLaneSelector/CampaignProfile from RE19) --
not legacy recipes/groups/states, which recipes.py's own _load_v2_compat
adapter documents as transitional compatibility concepts only.

The run_id collision trap flagged during RE17's design review: every
run-scoped filesystem/ArtifactStore path inside campaign_execution.py and
campaign_workers.py (e.g. ``runs/<run_id>/generate/generated-tree.json``,
``context.work_root/runs/<run_id>/generate/generated``) is keyed on run_id
alone, not on lane identity -- because execute_campaign_lane() was designed
for one lane per run_id (RE16/RE17 callers each get a fresh uuid). A naive
multi-lane runner sharing one campaign-level run_id across all its lanes
would collide two different lanes' generate/build/smoke artifacts onto the
same paths. The fix here is at this seam, not lower down: each lane gets
its own distinct execution run_id, derived from the shared campaign run_id
plus the lane's own stable identity, so paths never collide -- while the
campaign run_id itself is still recoverable from every lane's result for
grouping/reporting.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Mapping

from ..core import config as campaign_config
from ..experiment import contract as experiment_contract
from ..core.artifacts import ArtifactStore
from .lane import (CampaignLaneExecutionSpec, CampaignLaneResult,
                            LaneInputValue, execute_campaign_lane)
from ..core.context import ProjectContext
from .smoke import RuntimeSmokeSpec


class CampaignPlannerError(RuntimeError):
    pass


@dataclass(frozen=True)
class CampaignLane:
    """The canonical unit a planner produces: everything one lane execution
    needs, keyed by real config identities (not a legacy recipe name).
    """

    source_name: str
    build_name: str
    platform_name: str
    architectures: tuple[str, ...]
    inputs: tuple[tuple[str, LaneInputValue], ...] = ()
    validation: RuntimeSmokeSpec | None = None
    binary_relative_path: str = "bin/llama-bench"
    c_compiler: str | None = None
    cxx_compiler: str | None = None
    smoke_environment: tuple[tuple[str, str], ...] | None = None
    #: RE26 prep: names a [experiment.<name>] entry in recipes.toml (an
    #: exact extra patch list, per campaign_resolution.resolve_lane's own
    #: experiment= parameter) -- for benching ONE experimental patch in
    #: isolation against a source's normal framework patch-set, without
    #: needing a dedicated recipe/group per patch.
    experiment: str | None = None
    #: EC04: mirrors CampaignLaneExecutionSpec's own contract fields --
    #: see that dataclass's docstring for the identity-separation
    #: invariant these five fields must never violate.
    contract_id: str | None = None
    optimization_id: str | None = None
    role: str | None = None
    workload_tag: str | None = None
    model_ref: str | None = None
    boundary_dimension: str | None = None
    boundary_value: str | None = None


@dataclass(frozen=True)
class CampaignRequest:
    """What a caller supplies to plan(): either an explicit list of
    (source, build, platform) selections, or the name of a canonical
    profile from ``config.campaigns`` (RE19) -- exactly one of the two.

    ``inputs_by_build``/``validation_by_build`` are keyed by build name,
    not per-lane, because what a lane needs is a property of its build
    (``config.Build.needs``) -- a tune lane needs an inventory regardless
    of which source/platform it runs against, a stock lane needs nothing.
    """

    selectors: tuple[campaign_config.CampaignLaneSelector, ...] = ()
    profile_name: str | None = None
    architectures: tuple[str, ...] = ()
    inputs_by_build: Mapping[str, tuple[tuple[str, LaneInputValue], ...]] = field(
        default_factory=dict)
    validation_by_build: Mapping[str, RuntimeSmokeSpec | None] = field(
        default_factory=dict)
    binary_relative_path: str = "bin/llama-bench"
    c_compiler: str | None = None
    cxx_compiler: str | None = None
    smoke_environment: tuple[tuple[str, str], ...] | None = None
    #: RE26 prep: applies to every lane this request plans -- a request is
    #: already a single coherent build ask (one profile or one set of
    #: explicit lanes), so one experiment name per request, not per lane,
    #: matches how --experiment is meant to be used (isolate one patch
    #: across the standard lane set, not mix-and-match per lane).
    experiment: str | None = None


def lane_id(lane: CampaignLane) -> str:
    base = f"{lane.source_name}:{lane.build_name}:{lane.platform_name}"
    if lane.experiment is not None:
        # A patch-qualification profile deliberately holds the SAME
        # source/build/platform twice -- once plain as the baseline, once
        # carrying the patch under test. Without the experiment in the id
        # those two collide, and run_campaign()'s duplicate-lane check would
        # silently drop the arm that gives the comparison its meaning.
        base = f"{base}+{lane.experiment}"
    if lane.contract_id is None:
        return base
    # Contract-expanded lanes (EC03) legitimately share one source/build/
    # platform tuple across many roles/workloads/models/boundary points --
    # the whole point of expansion is many lanes over one build shape.
    # Fold in every EC04 field that distinguishes them so run_campaign()'s
    # duplicate-lane check and results dict stay meaningful instead of
    # silently colliding every contract lane onto one key.
    return ":".join((
        base, lane.contract_id, lane.role or "", lane.workload_tag or "",
        lane.model_ref or "", lane.boundary_dimension or "", lane.boundary_value or "",
    ))


def _resolve_architectures(
    request_architectures: tuple[str, ...], platform: campaign_config.Platform,
) -> tuple[str, ...]:
    """GPT review (round 20 check-in): empty/unspecified
    CampaignRequest.architectures means "use this lane's Platform.targets",
    not "generate for nothing". architectures flows straight into
    make_generate_worker's candidate-universe enumeration independently of
    platform.targets (which only drives AMDGPU_TARGETS at compile time) --
    an empty tuple here is not a harmless default, it is the exact
    pre-existing legacy _generate_for() bug (omits --arch, silently falls
    back to the separate generate CLI's own --arch all) that RE14's parity
    harness already had to route around. An explicit override must be a
    non-empty subset of the platform's declared targets -- never a
    superset, and never empty (that would just be the default spelled out
    redundantly, or a typo).
    """
    if not request_architectures:
        return platform.targets
    unknown = sorted(set(request_architectures) - set(platform.targets))
    if unknown:
        raise CampaignPlannerError(
            f"architecture(s) {unknown} are not in platform {platform.name!r}'s "
            f"targets {list(platform.targets)}"
        )
    return request_architectures


def plan(
    request: CampaignRequest, cfg: campaign_config.Config,
) -> tuple[CampaignLane, ...]:
    """Expand a request into concrete, config-validated lanes.

    Validates every selector's source/build/platform actually exist in
    ``cfg`` up front -- a lane that fails this never reaches
    execute_campaign_lane(), rather than failing deep inside materialize.
    """
    if bool(request.selectors) == bool(request.profile_name):
        raise CampaignPlannerError(
            "exactly one of request.selectors or request.profile_name must be set"
        )
    if request.profile_name is not None:
        profile = cfg.campaigns.get(request.profile_name)
        if profile is None:
            raise CampaignPlannerError(
                f"no campaign profile named {request.profile_name!r}"
            )
        selectors = profile.lanes
    else:
        selectors = request.selectors

    lanes: list[CampaignLane] = []
    seen_lane_ids: set[str] = set()
    for selector in selectors:
        if selector.source not in cfg.sources:
            raise CampaignPlannerError(f"unknown source {selector.source!r}")
        if selector.build not in cfg.builds:
            raise CampaignPlannerError(f"unknown build {selector.build!r}")
        if selector.platform not in cfg.platforms:
            raise CampaignPlannerError(f"unknown platform {selector.platform!r}")
        # GPT review (round 20 check-in): _lane_run_id()/run_campaign()'s
        # result dict are both keyed on source:build:platform alone -- a
        # duplicate selector (a hand-built request, or a campaign profile
        # that happens to repeat one) would silently overwrite an earlier
        # lane's result rather than erroring. Fail closed here, once, for
        # every caller rather than leaving it a fail-open edge in the
        # production planner.
        selector_id = f"{selector.source}:{selector.build}:{selector.platform}"
        if selector.experiment is not None:
            # Must match lane_id()'s composition exactly, or a legitimate
            # patched/unpatched pair on one source is rejected as a duplicate.
            selector_id = f"{selector_id}+{selector.experiment}"
        if selector_id in seen_lane_ids:
            raise CampaignPlannerError(f"duplicate lane {selector_id!r} in request")
        seen_lane_ids.add(selector_id)
        platform_cfg = cfg.platforms[selector.platform]
        lanes.append(CampaignLane(
            source_name=selector.source, build_name=selector.build,
            platform_name=selector.platform,
            architectures=_resolve_architectures(request.architectures, platform_cfg),
            inputs=request.inputs_by_build.get(selector.build, ()),
            validation=request.validation_by_build.get(selector.build),
            # A lane's own binary wins: it decides which cmake target is built,
            # so a profile that needs llama-server must be able to say so
            # without the caller remembering --binary-relative-path.
            binary_relative_path=selector.binary or request.binary_relative_path,
            c_compiler=request.c_compiler, cxx_compiler=request.cxx_compiler,
            smoke_environment=request.smoke_environment,
            # A lane's OWN experiment wins over the request-level one. A
            # patch-qualification profile needs arms that carry the patch and
            # arms that deliberately do not (the baselines they are measured
            # against), which a single request-level --experiment applying to
            # every lane cannot express. Request-level remains the default for
            # "isolate one patch across the standard lane set".
            experiment=selector.experiment or request.experiment,
        ))
    return tuple(lanes)


def expand_contract(
    contract: experiment_contract.ExperimentContract,
    *,
    cfg: campaign_config.Config,
    source_name: str,
    build_name: str,
    platform_name: str,
    architectures: tuple[str, ...] = (),
    inputs: tuple[tuple[str, LaneInputValue], ...] = (),
    validation: RuntimeSmokeSpec | None = None,
    binary_relative_path: str = "bin/llama-bench",
    c_compiler: str | None = None,
    cxx_compiler: str | None = None,
    smoke_environment: tuple[tuple[str, str], ...] | None = None,
    experiment: str | None = None,
) -> tuple[CampaignLane, ...]:
    """EC03: expand one Experiment Contract into positive, control and
    boundary lanes -- one CampaignLane per (model, workload) pair in
    ``contract.positive``, one per pair in ``contract.controls``, and one
    per value in each of ``contract.boundary.dimensions``.

    All expanded lanes share the SAME source/build/platform/architectures
    (and therefore the same build_plan_id/effective_build_id) -- a contract
    plans many EVALUATION ROLES over one build shape, it does not plan
    separate builds. This is exactly the "keep runtime candidate identity
    separate from experiment identity" non-negotiable from the Experiment
    Contract guide: contract_id/role/workload_tag/model_ref/boundary_* are
    the only things that vary between these lanes' specs (see
    CampaignLaneExecutionSpec's own docstring for the identity-separation
    invariant).

    Does not itself execute anything -- pass the result to run_campaign()
    exactly like plan()'s output, or execute individual lanes via
    execute_campaign_lane(_to_spec(lane), ...).
    """
    if source_name not in cfg.sources:
        raise CampaignPlannerError(f"unknown source {source_name!r}")
    if build_name not in cfg.builds:
        raise CampaignPlannerError(f"unknown build {build_name!r}")
    if platform_name not in cfg.platforms:
        raise CampaignPlannerError(f"unknown platform {platform_name!r}")
    platform_cfg = cfg.platforms[platform_name]
    resolved_architectures = _resolve_architectures(architectures, platform_cfg)

    def _make(
        *, role: str, workload_tag: str | None = None, model_ref: str | None = None,
        boundary_dimension: str | None = None, boundary_value: object | None = None,
    ) -> CampaignLane:
        return CampaignLane(
            source_name=source_name, build_name=build_name, platform_name=platform_name,
            architectures=resolved_architectures, inputs=inputs, validation=validation,
            binary_relative_path=binary_relative_path, c_compiler=c_compiler,
            cxx_compiler=cxx_compiler, smoke_environment=smoke_environment,
            experiment=experiment,
            contract_id=contract.id, optimization_id=contract.source.atomic_part,
            role=role, workload_tag=workload_tag, model_ref=model_ref,
            boundary_dimension=boundary_dimension,
            boundary_value=None if boundary_value is None else str(boundary_value),
        )

    lanes: list[CampaignLane] = []
    for model in contract.positive.models:
        for workload in contract.positive.workloads:
            lanes.append(_make(role="positive", workload_tag=workload, model_ref=model))
    for model in contract.controls.models:
        for workload in contract.controls.workloads:
            lanes.append(_make(role="control", workload_tag=workload, model_ref=model))
    for dimension_name, values in contract.boundary.dimensions:
        for value in values:
            lanes.append(_make(
                role="boundary", boundary_dimension=dimension_name, boundary_value=value,
            ))

    seen_ids: set[str] = set()
    for lane in lanes:
        lid = lane_id(lane)
        if lid in seen_ids:
            raise CampaignPlannerError(
                f"duplicate contract lane {lid!r} -- contract "
                f"{contract.id!r}'s positive/controls/boundary sections "
                f"produced two identical lanes"
            )
        seen_ids.add(lid)
    return tuple(lanes)


def _to_spec(lane: CampaignLane) -> CampaignLaneExecutionSpec:
    return CampaignLaneExecutionSpec(
        source_name=lane.source_name, build_name=lane.build_name,
        platform_name=lane.platform_name, architectures=lane.architectures,
        inputs=lane.inputs, validation=lane.validation,
        binary_relative_path=lane.binary_relative_path,
        c_compiler=lane.c_compiler, cxx_compiler=lane.cxx_compiler,
        smoke_environment=lane.smoke_environment,
        experiment=lane.experiment,
        contract_id=lane.contract_id, optimization_id=lane.optimization_id,
        role=lane.role, workload_tag=lane.workload_tag, model_ref=lane.model_ref,
        boundary_dimension=lane.boundary_dimension, boundary_value=lane.boundary_value,
    )


def _lane_run_id(campaign_run_id: str, lane: CampaignLane) -> str:
    """A distinct execution run_id per lane, derived from the shared
    campaign run_id plus the lane's own stable identity -- see module
    docstring for why sharing one run_id across lanes is unsafe.
    """
    digest = hashlib.blake2b(lane_id(lane).encode("utf-8"), digest_size=6).hexdigest()
    return f"{campaign_run_id}-{digest}"


def run_campaign(
    lanes: tuple[CampaignLane, ...],
    *,
    cfg: campaign_config.Config,
    context: ProjectContext,
    store: ArtifactStore,
    run_id: str | None = None,
    allow_dirty_bigcherry: bool = False,
) -> dict[str, CampaignLaneResult | Exception]:
    """Execute every lane sequentially, isolating faults per lane: one
    lane raising must not affect any other lane's own result, matching
    legacy cmd_build's own per-recipe failure aggregation behavior.

    Defensively rejects duplicate lane identities even though plan()
    already does (GPT review, round 20 check-in): callers can construct
    CampaignLanes directly without going through plan(), and a duplicate
    here would silently overwrite an earlier lane's result in ``results``
    (both keyed on the same lane_id) rather than erroring.

    ``allow_dirty_bigcherry`` (RE04/RV48 audit fix) defaults to False and
    is forwarded to every lane's ``execute_campaign_lane()`` call -- the
    production `build` CLI never sets it; only a development/harness
    caller has a real reason to.
    """
    duplicate_ids = sorted({
        lid for lid in (lane_id(lane) for lane in lanes)
        if sum(1 for other in lanes if lane_id(other) == lid) > 1
    })
    if duplicate_ids:
        raise CampaignPlannerError(f"duplicate lane(s) in run_campaign(): {duplicate_ids}")
    campaign_run_id = run_id or uuid.uuid4().hex[:12]
    results: dict[str, CampaignLaneResult | Exception] = {}
    for lane in lanes:
        try:
            results[lane_id(lane)] = execute_campaign_lane(
                _to_spec(lane), cfg=cfg, context=context, store=store,
                run_id=_lane_run_id(campaign_run_id, lane),
                allow_dirty_bigcherry=allow_dirty_bigcherry,
            )
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: one
            # lane's failure (of any kind) must not abort sibling lanes.
            results[lane_id(lane)] = exc
    return results
