"""PROF01/HI132: profile-campaign's real, typed evidence records.

Kept intentionally small and JSON-round-trippable: every field here ends
up either in ``profile-receipt.json`` (identity/provenance) or
``profile-report.json`` (normalized measurements), matching the design
settled with gpt (session ses_fec603fc33ee4089).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ControlBlock:
    """One block of ``control_reps`` unprofiled completions, run to detect
    environmental drift around a profiler pass (not to measure the
    profiler itself)."""

    label: str
    reps: int
    tg_tps_values: tuple[float, ...]
    tg_tps_mean: float
    tg_tps_stddev: float


@dataclass(frozen=True)
class KernelStat:
    """One kernel's aggregated rocprofv3 kernel-trace row, aggregated
    across all dispatches captured in one profiler pass."""

    name: str
    calls: int
    total_us: float
    mean_us: float
    p95_us: float
    vgpr_count: int
    sgpr_count: int
    scratch_size: int
    agent_ids: tuple[str, ...]


@dataclass(frozen=True)
class GpuProfilePass:
    """One rocprofv3 --sys-trace --rccl-trace pass."""

    label: str
    output_dir: str
    kernels: tuple[KernelStat, ...]
    agent_ids_seen: tuple[str, ...]
    rccl_activity_seen: bool
    expected_gpu_count: int
    capture_status: str  # "complete" | "incomplete_multi_gpu_capture"
    expected_reduction_provider: str = "auto"


@dataclass(frozen=True)
class CpuProfilePass:
    """Reserved for HI133 (perf). Always ``available=False`` until that
    item lands -- profile-campaign must degrade gracefully, not fail,
    when CPU call-graph profiling is unavailable."""

    available: bool = False
    reason: str = "perf is not usable in this environment yet (see HI133)"


@dataclass(frozen=True)
class ProfileReceipt:
    """Identity/provenance for one profile-campaign run -- what was run,
    against which build, with which tools, when."""

    schema_version: int
    campaign_run_id: str
    model_path: str
    platform_name: str
    devices: str
    runtime_profile_name: str
    workload_label: str
    lane_source: str
    lane_build: str
    build_plan_id: str
    source_slice_id: str
    binary_path: str
    rocprofv3_version: str | None
    control_reps: int
    profile_passes: int
    expected_gpu_count: int
    started_at: str
    finished_at: str
    environment_stable: bool
    environment_note: str = ""


@dataclass(frozen=True)
class ProfileReport:
    receipt: ProfileReceipt
    controls: tuple[ControlBlock, ...] = field(default_factory=tuple)
    gpu_passes: tuple[GpuProfilePass, ...] = field(default_factory=tuple)
    cpu: CpuProfilePass = field(default_factory=CpuProfilePass)
