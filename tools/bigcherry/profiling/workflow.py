"""PROF01/HI132: bigcherry profile-campaign.

Real rocprofv3 GPU/runtime profiling, interleaved with unprofiled control
blocks to detect environmental drift -- the same lesson HI130 learned the
hard way (unseeded sampling and prompt content masquerading as real
signal). Design settled with gpt (session ses_fec603fc33ee4089,
req_c3ff203a1d4b40c5 + req_8d0cd9d969314e31, 2026-08-27).

CPU call-graph profiling (perf) is NOT implemented here -- perf is
currently non-functional on Brutus (kernel/package mismatch, see HI133).
This workflow always reports ``cpu: unavailable`` rather than failing;
every stage that touches a GPU still runs for real.

Calls existing campaign/tuning APIs directly, matching HI130's own
pattern -- never shells out to the ``bigcherry`` CLI internally.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from . import rocprof as rocprof_mod
from .control import run_control_block
from .report import render_markdown, report_to_dict
from .schema import CpuProfilePass, ProfileReceipt, ProfileReport
from ..campaign import planner as campaign_planner
from ..core import config as campaign_config
from ..core import paths
from ..core.artifacts import ArtifactStore
from ..core.context import ProjectContext
from ..source.identity import atomic_write_json
from ..tuning.server_runner import ServerRunner

# Environmental-drift gate: if the unprofiled control blocks' means spread
# by more than this fraction of their own pooled mean, the run is not
# trustworthy for comparative conclusions (profiling artifacts are still
# real and useful diagnostically -- see ProfileReceipt.environment_stable).
_DRIFT_THRESHOLD_FRACTION = 0.05


class ProfileCampaignError(RuntimeError):
    pass


def _control_prompt() -> str:
    # Fixed, not sampled from the model's own state -- this is a load
    # generator for profiling/drift-detection, not a quality measurement.
    return "Explain how a compass works in three sentences."


def run_profile_campaign(
    *,
    context: ProjectContext,
    cfg: campaign_config.Config,
    store: ArtifactStore,
    model_path: Path,
    platform_name: str,
    devices: str,
    runtime_profile_name: str,
    workload_label: str,
    source_name: str = "bigcherry-native",
    build_name: str = "control",
    run_id: str | None = None,
    workdir: Path | None = None,
    control_reps: int = 10,
    profile_passes: int = 2,
    n_predict: int = 96,
) -> ProfileReport:
    import uuid

    profile = cfg.runtime_profiles.get(runtime_profile_name)
    if profile is None:
        raise ProfileCampaignError(
            f"no runtime-profile named {runtime_profile_name!r} -- known: "
            f"{sorted(cfg.runtime_profiles)}"
        )
    campaign_run_id = run_id or uuid.uuid4().hex[:12]
    workdir = workdir or (context.work_root / "profile-campaigns" / campaign_run_id)
    workdir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    devices_tuple = tuple(int(d) for d in devices.split(","))
    expected_gpu_count = len(devices_tuple)

    request = campaign_planner.CampaignRequest(
        selectors=(campaign_config.CampaignLaneSelector(
            source=source_name, build=build_name, platform=platform_name,
        ),),
        inputs_by_build={},
        binary_relative_path="bin/llama-server",
    )
    lanes = campaign_planner.plan(request, cfg)
    results = campaign_planner.run_campaign(
        lanes, cfg=cfg, context=context, store=store, run_id=f"{campaign_run_id}-build",
    )
    (lane_result,) = results.values()
    if isinstance(lane_result, Exception):
        raise ProfileCampaignError(f"build failed: {lane_result}") from lane_result
    binary_path = lane_result.binary_ref.path

    base_env = {"HIP_VISIBLE_DEVICES": devices}
    base_args = ("-ngl", "99", "-c", str(profile.tune_context), *profile.server_args)

    def _launch(*, command_prefix: tuple[str, ...] = (), log_name: str) -> ServerRunner:
        return ServerRunner(
            binary=binary_path, model=model_path, extra_args=base_args,
            env_overrides=base_env, log_path=workdir / log_name,
            command_prefix=command_prefix,
        )

    controls = []
    gpu_passes = []

    def _run_control(label: str) -> None:
        with _launch(log_name=f"control-{label}-server.log") as runner:
            controls.append(run_control_block(
                runner=runner, label=label, reps=control_reps,
                prompt=_control_prompt(), n_predict=n_predict,
            ))

    def _run_gpu_pass(label: str) -> None:
        out_dir = workdir / f"gpu-{label}"
        prefix = rocprof_mod.rocprofv3_command_prefix(output_dir=out_dir, label=label)
        with _launch(command_prefix=prefix, log_name=f"gpu-{label}-server.log") as runner:
            runner.run_completion(_control_prompt(), n_predict=n_predict)
        gpu_passes.append(rocprof_mod.build_gpu_profile_pass(
            label=label, output_dir=out_dir, expected_gpu_count=expected_gpu_count,
        ))

    # Real stage sequence (gpt-settled): control -> GPU pass 1 -> control ->
    # [CPU unavailable] -> control -> [CPU unavailable] -> control -> GPU
    # pass 2 -> control. Never run two profilers in the same server launch --
    # their observer effects would be inseparable.
    _run_control("A")
    if profile_passes >= 1:
        _run_gpu_pass("1")
    _run_control("B")
    _run_control("C")  # stands in for the (unavailable) CPU pass 1 slot
    _run_control("D")  # stands in for the (unavailable) CPU pass 2 slot
    if profile_passes >= 2:
        _run_gpu_pass("2")
    _run_control("E")

    means = [c.tg_tps_mean for c in controls if c.tg_tps_mean > 0]
    environment_stable = True
    environment_note = ""
    if len(means) >= 2:
        spread = (max(means) - min(means)) / statistics.fmean(means)
        if spread > _DRIFT_THRESHOLD_FRACTION:
            environment_stable = False
            environment_note = (
                f"control block means spread {spread:.1%} across the run "
                f"(threshold {_DRIFT_THRESHOLD_FRACTION:.0%}) -- profiling "
                "artifacts are still real and usable diagnostically, but no "
                "comparative performance conclusion should be drawn from "
                "this run alone"
            )

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    receipt = ProfileReceipt(
        schema_version=1,
        campaign_run_id=campaign_run_id,
        model_path=str(model_path),
        platform_name=platform_name,
        devices=devices,
        runtime_profile_name=runtime_profile_name,
        workload_label=workload_label,
        lane_source=source_name,
        lane_build=build_name,
        build_plan_id=lane_result.build_plan_id,
        source_slice_id=lane_result.source_slice_id,
        binary_path=str(binary_path),
        rocprofv3_version=rocprof_mod.rocprofv3_version(),
        control_reps=control_reps,
        profile_passes=profile_passes,
        expected_gpu_count=expected_gpu_count,
        started_at=started_at,
        finished_at=finished_at,
        environment_stable=environment_stable,
        environment_note=environment_note,
    )
    report = ProfileReport(
        receipt=receipt, controls=tuple(controls), gpu_passes=tuple(gpu_passes),
        cpu=CpuProfilePass(),
    )

    atomic_write_json(workdir / "profile-receipt.json", report_to_dict(report)["receipt"])
    atomic_write_json(workdir / "profile-report.json", report_to_dict(report))
    (workdir / "profile-report.md").write_text(render_markdown(report), encoding="utf-8", newline="")
    store.publish_json(f"profile-campaigns/{campaign_run_id}/report.json", report_to_dict(report))
    return report
