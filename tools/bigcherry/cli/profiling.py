"""PROF01/HI132: CLI handler for ``bigcherry profile-campaign``. Mirrors
cli/tuning.py::cmd_tune_campaign's shape -- this only resolves CLI-level
context/config and renders the result; see profiling/workflow.py for the
actual orchestration."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path


def cmd_profile_campaign(args: Namespace) -> int:
    from ..core import config as campaign_config
    from ..core.artifacts import ArtifactStore
    from ..core.context import ProjectContext
    from ..profiling import workflow

    context = ProjectContext.resolve(
        work_root=None, upstream_repo=Path(args.llama_root) if args.llama_root else None
    )
    try:
        cfg = campaign_config.load(context.config_path)
    except campaign_config.ConfigError as exc:
        print(f"profile-campaign: {exc}", file=sys.stderr)
        return 2

    if args.runtime_profile not in cfg.runtime_profiles:
        print(
            f"profile-campaign: no runtime-profile named {args.runtime_profile!r} -- "
            f"known: {sorted(cfg.runtime_profiles)}",
            file=sys.stderr,
        )
        return 2

    store = ArtifactStore(context.work_root / "artifacts-store")
    try:
        report = workflow.run_profile_campaign(
            context=context,
            cfg=cfg,
            store=store,
            model_path=Path(args.model),
            platform_name=args.platform,
            devices=args.devices,
            runtime_profile_name=args.runtime_profile,
            workload_label=args.workload,
            source_name=args.source,
            build_name=args.build,
            experiment=args.experiment,
            run_id=args.run_id,
            workdir=Path(args.workdir) if args.workdir else None,
            control_reps=args.control_reps,
            profile_passes=args.profile_passes,
        )
    except workflow.ProfileCampaignError as exc:
        print(f"profile-campaign: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        r = report.receipt
        print(f"campaign_run_id: {r.campaign_run_id}")
        print(f"environment_stable: {r.environment_stable}"
              + (f" -- {r.environment_note}" if r.environment_note else ""))
        for gp in report.gpu_passes:
            print(f"gpu pass {gp.label}: capture_status={gp.capture_status} "
                  f"kernels={len(gp.kernels)} agents={gp.agent_ids_seen}")
        print(f"cpu profile: available={report.cpu.available} ({report.cpu.reason})")
        print(
            f"report: {context.work_root / 'profile-campaigns' / r.campaign_run_id / 'profile-report.md'}"
        )
    return 0
