"""CLI presentation handler for the build workflow."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

from ..release import pin_status


def cmd_build_new(args: Namespace) -> int:
    """RE21: `build`'s new-engine implementation -- parse -> plan -> run_campaign
    -> render. No legacy tree-state/checkout mutation of any kind; this only
    ever touches ``context.work_root`` and a dedicated ArtifactStore, exactly
    like ``campaign-build``/``re14_real_run`` before it.

    Exit codes (RE21/RE22): 2 for invalid/unsupported request syntax
    (argparse itself raises this for any legacy-only flag, since this parser
    simply does not define them); 1 if one or more planned lanes execute and
    fail; 0 only if every planned lane succeeds.
    """
    from ..core import config as campaign_config
    from ..core.artifacts import ArtifactStore
    from ..campaign.lane import smoke_environment_for_hip_devices
    from ..campaign.planner import (
        CampaignPlannerError,
        CampaignRequest,
        plan,
        run_campaign,
    )
    from ..core.context import ProjectContext
    from ..campaign.smoke import RuntimeSmokeSpec

    if sum(bool(x) for x in (args.lane, args.all, args.profile)) != 1:
        print(
            "build: pass exactly one of --profile, --all, or --lane (repeatable)",
            file=sys.stderr,
        )
        return 2

    context = ProjectContext.resolve(
        work_root=None, upstream_repo=Path(args.llama_root) if args.llama_root else None
    )
    try:
        cfg = campaign_config.load(context.config_path)
    except campaign_config.ConfigError as exc:
        print(f"build: {exc}", file=sys.stderr)
        return 2

    store = ArtifactStore(context.work_root / "artifacts-store")

    selectors: tuple[campaign_config.CampaignLaneSelector, ...] = ()
    profile_name = None
    if args.all:
        profile_name = "standard"
    elif args.profile:
        profile_name = args.profile
    else:
        parsed: list[campaign_config.CampaignLaneSelector] = []
        for raw in args.lane:
            parts = raw.split(":")
            if len(parts) != 3:
                print(
                    f"build: --lane {raw!r} must be SOURCE:BUILD:PLATFORM",
                    file=sys.stderr,
                )
                return 2
            parsed.append(campaign_config.CampaignLaneSelector(*parts))
        selectors = tuple(parsed)

    # RE48 preflight: never start a lane from a tree whose pin state is
    # drift / uncommitted-transition / unresolvable (fail closed, no
    # bypass). mid-rebase is allowed with a warning: the pipeline's source
    # identity is revision-bound. A campaign-only tree with no git checkout
    # at the clone source has nothing to guard (its lanes bind their own
    # source identity). Placed AFTER the exit-2 request-syntax checks above
    # so a malformed request reports its syntax error, not the tree state.
    try:
        pin_repo_paths = pin_status.RepoPaths(
            repo_root=context.project_root,
            llama_root=context.upstream_repo,
            releases_dir=context.project_root / "releases",
            artifacts_dir=context.project_root / "artifacts",
        )
        pin_failures, pin_mid_rebase = pin_status.campaign_preflight(
            context.upstream_repo, pin_repo_paths
        )
    except Exception as exc:
        print(f"build: pin preflight error: {exc}", file=sys.stderr)
        return 1
    if pin_failures:
        for reason in pin_failures:
            print(f"build: pin preflight FAIL: {reason}", file=sys.stderr)
        print(
            "build: run `bigcherry pin-status` for the full report and see "
            "docs/reference/PIN_BUMP.md (fail closed)",
            file=sys.stderr,
        )
        return 1
    if pin_mid_rebase:
        print(
            "WARNING: tree is mid-rebase (a declared bump is in flight); "
            "proceeding because the pipeline's source identity is "
            "revision-bound.",
            file=sys.stderr,
        )

    architectures = tuple(args.arch.split(",")) if args.arch else ()
    inventory = Path(args.inventory) if args.inventory else None
    winners = Path(args.winners) if args.winners else None
    validation = RuntimeSmokeSpec(model_path=Path(args.model)) if args.model else None
    inputs_by_build = {}
    validation_by_build = {}
    for build_name, build_cfg in cfg.builds.items():
        needed = build_cfg.needs
        provided = {}
        if inventory is not None and "inventory" in needed:
            provided["inventory"] = inventory
        if winners is not None and "promoted-winners" in needed:
            provided["promoted-winners"] = winners
        if provided:
            inputs_by_build[build_name] = tuple(sorted(provided.items()))
        if validation is not None:
            validation_by_build[build_name] = validation

    request = CampaignRequest(
        selectors=selectors,
        profile_name=profile_name,
        architectures=architectures,
        inputs_by_build=inputs_by_build,
        validation_by_build=validation_by_build,
        binary_relative_path=args.binary_relative_path,
        c_compiler=args.c_compiler,
        cxx_compiler=args.cxx_compiler,
        smoke_environment=smoke_environment_for_hip_devices(args.hip_visible_devices),
        experiment=args.experiment,
    )
    try:
        lanes = plan(request, cfg)
    except CampaignPlannerError as exc:
        print(f"build: {exc}", file=sys.stderr)
        return 2

    results = run_campaign(
        lanes, cfg=cfg, context=context, store=store, run_id=args.run_id
    )

    failed = 0
    for lid in sorted(results):
        result = results[lid]
        if isinstance(result, Exception):
            print(f"{lid}: FAILED -- {result}", file=sys.stderr)
            failed += 1
        else:
            print(
                f"{lid}: ok build_plan_id={result.build_plan_id} "
                f"workload_id={result.workload_id}"
            )
    if failed:
        print(f"build: {failed}/{len(results)} lane(s) failed", file=sys.stderr)
        return 1
    return 0
