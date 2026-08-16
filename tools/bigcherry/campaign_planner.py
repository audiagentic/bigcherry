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

from . import config as campaign_config
from .artifacts import ArtifactStore
from .campaign_lane import (CampaignLaneExecutionSpec, CampaignLaneResult,
                            LaneInputValue, execute_campaign_lane)
from .context import ProjectContext
from .runtime_smoke import RuntimeSmokeSpec


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


def lane_id(lane: CampaignLane) -> str:
    return f"{lane.source_name}:{lane.build_name}:{lane.platform_name}"


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
    for selector in selectors:
        if selector.source not in cfg.sources:
            raise CampaignPlannerError(f"unknown source {selector.source!r}")
        if selector.build not in cfg.builds:
            raise CampaignPlannerError(f"unknown build {selector.build!r}")
        if selector.platform not in cfg.platforms:
            raise CampaignPlannerError(f"unknown platform {selector.platform!r}")
        lanes.append(CampaignLane(
            source_name=selector.source, build_name=selector.build,
            platform_name=selector.platform, architectures=request.architectures,
            inputs=request.inputs_by_build.get(selector.build, ()),
            validation=request.validation_by_build.get(selector.build),
            binary_relative_path=request.binary_relative_path,
            c_compiler=request.c_compiler, cxx_compiler=request.cxx_compiler,
            smoke_environment=request.smoke_environment,
        ))
    return tuple(lanes)


def _to_spec(lane: CampaignLane) -> CampaignLaneExecutionSpec:
    return CampaignLaneExecutionSpec(
        source_name=lane.source_name, build_name=lane.build_name,
        platform_name=lane.platform_name, architectures=lane.architectures,
        inputs=lane.inputs, validation=lane.validation,
        binary_relative_path=lane.binary_relative_path,
        c_compiler=lane.c_compiler, cxx_compiler=lane.cxx_compiler,
        smoke_environment=lane.smoke_environment,
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
) -> dict[str, CampaignLaneResult | Exception]:
    """Execute every lane sequentially, isolating faults per lane: one
    lane raising must not affect any other lane's own result, matching
    legacy cmd_build's own per-recipe failure aggregation behavior.
    """
    campaign_run_id = run_id or uuid.uuid4().hex[:12]
    results: dict[str, CampaignLaneResult | Exception] = {}
    for lane in lanes:
        try:
            results[lane_id(lane)] = execute_campaign_lane(
                _to_spec(lane), cfg=cfg, context=context, store=store,
                run_id=_lane_run_id(campaign_run_id, lane),
            )
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: one
            # lane's failure (of any kind) must not abort sibling lanes.
            results[lane_id(lane)] = exc
    return results
