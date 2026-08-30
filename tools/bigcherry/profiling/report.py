"""PROF01/HI132: normalize a ProfileReport into the two artifact forms
the design settled on -- profile-report.json (machine-readable) and
profile-report.md (human-readable)."""

from __future__ import annotations

from dataclasses import asdict

from .schema import ProfileReport


def report_to_dict(report: ProfileReport) -> dict:
    return asdict(report)


def render_markdown(report: ProfileReport) -> str:
    r = report.receipt
    lines = [
        f"# Profile campaign: {r.campaign_run_id}",
        "",
        f"- model: `{r.model_path}`",
        f"- lane: `{r.lane_source}:{r.lane_build}` on `{r.platform_name}`",
        f"- devices: `{r.devices}` (expected GPU count: {r.expected_gpu_count})",
        f"- runtime profile: `{r.runtime_profile_name}`",
        f"- workload: `{r.workload_label}`",
        f"- rocprofv3: `{r.rocprofv3_version or 'unknown'}`",
        f"- control reps: {r.control_reps}, profile passes: {r.profile_passes}",
        f"- environment stable: **{r.environment_stable}**"
        + (f" -- {r.environment_note}" if r.environment_note else ""),
        "",
        "## Controls (unprofiled)",
        "",
        "| block | reps | tg mean t/s | tg stddev |",
        "|---|---|---|---|",
    ]
    for c in report.controls:
        lines.append(f"| {c.label} | {c.reps} | {c.tg_tps_mean:.2f} | {c.tg_tps_stddev:.2f} |")

    lines += ["", "## CPU profile", ""]
    if report.cpu.available:
        lines.append("available")
    else:
        lines.append(f"unavailable -- {report.cpu.reason}")

    for gp in report.gpu_passes:
        lines += [
            "",
            f"## GPU profile: {gp.label}",
            "",
            f"- capture status: **{gp.capture_status}**",
            f"- GPU agents seen: {', '.join(gp.agent_ids_seen) or '(none)'} "
            f"(expected {gp.expected_gpu_count})",
            f"- expected reduction provider: {gp.expected_reduction_provider}",
            f"- RCCL activity seen: {gp.rccl_activity_seen}",
            "",
            "| kernel | calls | total us | mean us | p95 us | VGPR | SGPR | scratch |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for k in gp.kernels[:20]:
            lines.append(
                f"| `{k.name[:80]}` | {k.calls} | {k.total_us:.1f} | {k.mean_us:.2f} | "
                f"{k.p95_us:.2f} | {k.vgpr_count} | {k.sgpr_count} | {k.scratch_size} |"
            )
    return "\n".join(lines) + "\n"
