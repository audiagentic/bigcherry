"""Paired-benchmark infrastructure for the patch-validation campaign: wiring,
model/topology/device-pool resolution and the paired llama-bench executor."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bigcherry.build.builds import capture_completed_build_evidence
from bigcherry.patch.campaign.build import (
    _atomic_write_json,
    _full_requested_cmake_args,
    _hip_env,
    _print,
    build_tree,
    LLAMA_CPP_SRC,
    PatchCampaignError,
)
from bigcherry.patch.campaign.contract import assert_validation_subject_parity
from bigcherry.patch.campaign.trace import _require_real_gpu_execution

if TYPE_CHECKING:
    from bigcherry.experiment import execution as experiment_execution
    from bigcherry.patch import registry as patch_registry


_PAIRED_BENCH_WORKLOAD_FLAGS: dict[str, tuple[str, ...]] = {
    "decode": ("-p", "0", "-n", "128"),
    "prefill": ("-p", "512", "-n", "0"),
}


_PAIRED_BENCH_METRIC_NAME: dict[str, str] = {"decode": "tg128", "prefill": "pp512"}


_PAIRED_BENCH_METRIC_PATTERN: dict[str, "re.Pattern[str]"] = {
    "decode": re.compile(r"tg128\s*\|\s*([0-9.]+)"),
    "prefill": re.compile(r"pp512\s*\|\s*([0-9.]+)"),
}


def _paired_llama_bench_command(
    binary: Path,
    model: Path,
    workload: str,
    *,
    patch_args: tuple[str, ...] = (),
    runtime_args: tuple[str, ...] = (),
) -> list[str]:
    """PVPS02 step 2: the one llama-bench command shape both RD04 and
    RD08's producers build. ``patch_args`` land BEFORE -ngl (RD04's
    historical position for its -fa/-ctk/-ctv flags); ``runtime_args``
    land AFTER -ngl (RD08's historical position for e.g. -sm tensor
    topology flags) -- kept as two distinct insertion points, not one
    undifferentiated list, specifically so each caller's exact historical
    argv order is reproducible byte-for-byte."""
    if workload not in _PAIRED_BENCH_WORKLOAD_FLAGS:
        raise PatchCampaignError(
            f"paired llama-bench: no flag mapping for workload {workload!r}"
        )
    return [
        str(binary),
        "-m",
        str(model),
        *_PAIRED_BENCH_WORKLOAD_FLAGS[workload],
        *patch_args,
        "-ngl",
        "99",
        *runtime_args,
    ]


def _combined_llama_bench_command(
    binary: Path,
    model: Path,
    workloads: tuple[str, ...],
    *,
    patch_args: tuple[str, ...] = (),
    runtime_args: tuple[str, ...] = (),
) -> list[str]:
    """One llama-bench command covering several workloads (e.g. -p 512 -n 128)."""
    prompt, gen = "0", "0"
    for workload in workloads:
        flags = _PAIRED_BENCH_WORKLOAD_FLAGS.get(workload)
        if flags is None:
            raise PatchCampaignError(f"paired llama-bench: no flag mapping for workload {workload!r}")
        values = dict(zip(flags[::2], flags[1::2]))
        prompt = values["-p"] if values["-p"] != "0" else prompt
        gen = values["-n"] if values["-n"] != "0" else gen
    return [str(binary), "-m", str(model), "-p", prompt, "-n", gen, *patch_args, "-ngl", "99", *runtime_args]


@dataclass(frozen=True)
class PairedBenchmarkOutcome:
    """PVPS02 step 2: the result of running one or more paired llama-bench
    workloads (decode/prefill) for one control/subject binary pair."""

    runs: dict[str, "experiment_execution.PairedLaneRun"]
    commands: dict[str, dict[str, list[str]]]
    raw_logs: list[dict[str, object]]


# PVPS02 step 3: the closed vocabulary of recognized benchmark executors.
# One entry today (the shared paired-llama-bench primitive); a genuinely
# new execution shape (e.g. a server-based benchmark) gets a new name
# here, never a silent branch on patch id.
BENCHMARK_EXECUTORS: frozenset[str] = frozenset({"paired-llama-bench-v1"})


# GPT review (req_e608313764834497, 2026-09-11): resolve_benchmark_wiring()
# validated wiring.executor against BENCHMARK_EXECUTORS but nothing actually
# dispatched on it -- _run_performance_benchmark() always called
# run_paired_llama_benchmark() directly, harmless with one executor but a
# silent-wrong-implementation trap the moment a second is added. Populated
# after run_paired_llama_benchmark() is defined, below.
BENCHMARK_EXECUTOR_FUNCS: dict[str, object] = {}


@dataclass(frozen=True)
class BenchmarkWiring:
    """PVPS02 step 3: one patch's resolved performance-benchmark wiring,
    read from its validation.toml -- never a hardcoded per-patch branch."""

    executor: str
    patch_args: tuple[str, ...]


def resolve_benchmark_wiring(
    descriptor: "patch_registry.PatchDescriptor",
    *,
    root: "str | Path | None" = None,
) -> BenchmarkWiring:
    """A patch is generically benchmarkable iff exactly one of its
    REQUIRED capability="performance" checks declares a recognized
    benchmark-executor in its validation.toml config. Zero, more than
    one, or an unrecognized executor is a configuration error -- fail
    closed, never guess which check/executor was meant.

    ``root`` matches build_plan_for_patch()'s own parameter (default:
    the real patches/ tree) -- exposed here purely so this can be
    unit-tested against isolated fixtures without touching real patches."""
    from bigcherry.patch import validation as patch_validation

    plan = patch_validation.build_plan_for_patch(descriptor, root=root)
    if plan is None:
        raise PatchCampaignError(
            f"{descriptor.patch_id}: no validation plan (no bound contract and no "
            "validation.toml adapter) -- cannot resolve benchmark wiring"
        )
    wired = [
        check
        for check in plan.checks_for("performance")
        if check.required and check.config.get("benchmark-executor") is not None
    ]
    if not wired:
        raise PatchCampaignError(
            f"{descriptor.patch_id}: no required performance check declares a "
            "benchmark-executor in validation.toml"
        )
    if len(wired) > 1:
        raise PatchCampaignError(
            f"{descriptor.patch_id}: {len(wired)} required performance checks declare a "
            "benchmark-executor -- ambiguous wiring, exactly one is required"
        )
    check = wired[0]
    executor = check.config["benchmark-executor"]
    if executor not in BENCHMARK_EXECUTORS:
        raise PatchCampaignError(
            f"{descriptor.patch_id}: unknown benchmark-executor {executor!r} "
            f"(known: {sorted(BENCHMARK_EXECUTORS)})"
        )
    extra_args = check.config.get("benchmark-extra-args", [])
    if not isinstance(extra_args, list) or not all(
        isinstance(a, str) for a in extra_args
    ):
        raise PatchCampaignError(
            f"{descriptor.patch_id}: benchmark-extra-args must be a list of strings"
        )
    return BenchmarkWiring(executor=executor, patch_args=tuple(extra_args))


# PVPS02 step 5 (hardened after GPT review req_e608313764834497, 2026-09-11:
# tensor-2 selected 2 devices but never passed llama-bench's -sm tensor flag,
# so the dual-GPU cell silently ran with llama-bench's default split mode
# instead of the declared topology): the closed vocabulary of benchmark
# topologies a models.toml entry can declare, owning BOTH how many real
# devices it needs AND which runtime args select that split mode -- so a
# topology can never again own a device count without the args that make
# that count meaningful.
BENCHMARK_TOPOLOGIES: dict[str, "BenchmarkTopology"] = {}


@dataclass(frozen=True)
class BenchmarkTopology:
    device_count: int
    runtime_args: tuple[str, ...]


BENCHMARK_TOPOLOGIES["single"] = BenchmarkTopology(device_count=1, runtime_args=())


BENCHMARK_TOPOLOGIES["tensor-2"] = BenchmarkTopology(
    device_count=2, runtime_args=("-sm", "tensor")
)


# Preserved for any external/test code still keying off device count alone.
BENCHMARK_TOPOLOGY_DEVICE_COUNT: dict[str, int] = {
    name: topo.device_count for name, topo in BENCHMARK_TOPOLOGIES.items()
}


@dataclass(frozen=True)
class ResolvedBenchmarkModel:
    """PVPS02 step 5: one config/models.toml entry resolved against a real
    host's model root -- real file path, real verified size, real
    topology-derived device count and runtime args."""

    id: str
    path: Path
    size_bytes: int
    topology: str

    @property
    def device_count(self) -> int:
        return BENCHMARK_TOPOLOGIES[self.topology].device_count

    @property
    def runtime_args(self) -> tuple[str, ...]:
        return BENCHMARK_TOPOLOGIES[self.topology].runtime_args


def resolve_benchmark_model(
    model_id: str,
    *,
    model_root: Path,
    registry_path: "Path | None" = None,
) -> ResolvedBenchmarkModel:
    """Resolve a model id from config/models.toml into a real, verified
    file path + benchmark topology (docs/planning/active/
    patching-validation-package-standard/PVPS02.md). Fails closed on: an
    unknown id, a missing file, a real-vs-declared size mismatch (wrong
    file/quantisation), or a model with no declared benchmark-topology --
    deliberately never defaults an undeclared topology to "single", since
    that would silently under-provision a model that actually needs 2+
    devices (e.g. tierL-qwen27b-q8)."""
    import tomllib

    from bigcherry.core import paths as bc_paths

    resolved_registry = registry_path if registry_path is not None else bc_paths.MODELS
    raw = tomllib.loads(resolved_registry.read_text(encoding="utf-8"))
    entries = {
        entry["id"]: entry
        for entry in raw.get("models", [])
        if isinstance(entry, dict) and entry.get("id")
    }
    entry = entries.get(model_id)
    if entry is None:
        raise PatchCampaignError(
            f"unknown benchmark model id {model_id!r} (not in {resolved_registry})"
        )
    topology = entry.get("benchmark-topology")
    if topology not in BENCHMARK_TOPOLOGIES:
        raise PatchCampaignError(
            f"{model_id}: benchmark-topology must be one of "
            f"{sorted(BENCHMARK_TOPOLOGIES)}, got {topology!r} -- a models.toml "
            "entry with no (or an unrecognized) benchmark-topology is not eligible for the "
            "generic performance-benchmark matrix"
        )
    path_value = entry.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise PatchCampaignError(f"{model_id}: models.toml entry has no valid 'path'")
    model_path = model_root / path_value
    if not model_path.is_file():
        raise PatchCampaignError(f"{model_id}: model file not found at {model_path}")
    # GPT review (req_e608313764834497, 2026-09-11): size-bytes verification
    # must be fail-closed, not skip-if-absent-or-malformed -- a missing or
    # non-integer size-bytes silently bypassed the "wrong file/quantisation"
    # check this exists for, contradicting PVPS02's explicit requirement of
    # real file-size verification.
    declared_size = entry.get("size-bytes")
    if (
        not isinstance(declared_size, int)
        or isinstance(declared_size, bool)
        or declared_size <= 0
    ):
        raise PatchCampaignError(
            f"{model_id}: models.toml entry must declare a positive integer size-bytes "
            f"(got {declared_size!r})"
        )
    real_size = model_path.stat().st_size
    if real_size != declared_size:
        raise PatchCampaignError(
            f"{model_id}: real file size {real_size} does not match models.toml's declared "
            f"size-bytes={declared_size} at {model_path} -- wrong file or quantisation?"
        )
    return ResolvedBenchmarkModel(
        id=model_id, path=model_path, size_bytes=real_size, topology=topology
    )


def resolve_device_pool(
    device_map: dict[str, tuple[str, ...]],
    architecture: str,
    device_count: int,
) -> tuple[str, ...]:
    """PVPS02 step 5: --device-map defines an ORDERED device pool per
    architecture; an N-device cell consumes the first N ids from that
    architecture's pool. Order is load-bearing (which physical card ends
    up in which tensor-split slot), never sorted/reordered. Fails closed
    if the architecture has no mapping at all, or its pool has fewer
    devices than the cell needs -- never silently infers a device index
    from the architecture name."""
    pool = device_map.get(architecture)
    if pool is None:
        raise PatchCampaignError(
            f"--device-map has no entry for architecture {architecture!r}"
        )
    if len(pool) < device_count:
        raise PatchCampaignError(
            f"--device-map {architecture}={','.join(pool)} exposes only {len(pool)} "
            f"device(s), this cell needs {device_count}"
        )
    return pool[:device_count]


def parse_device_map(entries: list[str]) -> dict[str, tuple[str, ...]]:
    """Parse repeated --device-map ARCH=ID[,ID...] CLI values into an
    ordered-pool-per-architecture dict. Fails closed on a malformed entry
    or the same architecture given twice (ambiguous which pool wins)."""
    device_map: dict[str, tuple[str, ...]] = {}
    for entry in entries:
        if "=" not in entry:
            raise PatchCampaignError(
                f"--device-map {entry!r} is malformed -- expected ARCH=ID[,ID...]"
            )
        arch, _, ids_raw = entry.partition("=")
        arch = arch.strip()
        if not arch:
            raise PatchCampaignError(
                f"--device-map {entry!r}: architecture must not be blank"
            )
        if arch in device_map:
            raise PatchCampaignError(
                f"--device-map: architecture {arch!r} given more than once"
            )
        ids = tuple(i.strip() for i in ids_raw.split(","))
        if not ids_raw or any(not i for i in ids):
            raise PatchCampaignError(
                f"--device-map {entry!r}: device id list must not be empty or contain a "
                "blank entry"
            )
        device_map[arch] = ids
    return device_map


def run_paired_llama_benchmark(
    *,
    control_binary: Path,
    subject_binary: Path,
    model: Path,
    hip_path: Path,
    workloads: tuple[str, ...] = ("decode", "prefill"),
    patch_args: tuple[str, ...] = (),
    runtime_args: tuple[str, ...] = (),
    pairs: int = 3,
    log_context: str,
    env_overrides: dict[str, str] | None = None,
    env_unset: tuple[str, ...] = (),
    execution_identity: "object | None" = None,
    combined: bool = False,
) -> PairedBenchmarkOutcome:
    """PVPS02 step 2/4: the shared execution shape behind the patch-local
    producers' paired benchmarks (1202/RD04 and 1204/RD08, via
    ProducerRuntime.run_paired_llama_benchmark()) and behind the generic
    --run-performance-benchmark path (registered as
    BENCHMARK_EXECUTOR_FUNCS["paired-llama-bench-v1"]) -- a pure,
    semantics-preserving extraction of the duplicated
    clean-env/runner/command/raw-log/paired-run logic
    (docs/planning/active/patching-validation-package-standard/PVPS02.md).
    The dedicated RD04/RD08 compatibility wrappers were retired by the
    PA36 producer migrations; the producers and the generic benchmark
    executor are the callers.

    ``env_overrides`` (step 4): applied on top of the sanitized/stripped
    environment for THIS call's own subprocesses only -- never mutates
    the real process-global os.environ, so a multi-cell caller (the
    generic matrix) can give each cell its own HIP_VISIBLE_DEVICES
    without any risk of one cell's selector leaking into the next.
    ROCR_VISIBLE_DEVICES is never a real selector here (see below) --
    a caller must not pass it in env_overrides either; if one does, it
    is stripped again, not honored. ``execution_identity`` (step 4):
    passed straight through to run_paired_lane(), which attests every
    measured process before accepting its metric -- proves architecture
    + device COUNT, not physical card identity (llama-bench's own ROCm
    attestation never carries a device locator; see DeviceVisibility's
    docstring)."""
    from bigcherry.experiment import execution as experiment_execution
    from bigcherry.campaign.benchmark import sanitize_environment

    clean_env = sanitize_environment(_hip_env(hip_path), mode="stock")
    for key in list(clean_env):
        if key.startswith("BIGCHERRY_") or key == "GGML_CUDA_DISABLE_FUSION":
            clean_env.pop(key, None)
    if env_overrides:
        clean_env.update(env_overrides)
    # GPT review (req_8429aa8e0d35496e / follow-up on req_d1ef22d846854960,
    # 2026-09-11): sanitize_environment() does not strip ROCR_VISIBLE_DEVICES,
    # so an ambient value inherited from the invoking shell would otherwise
    # reach this measured process even though this module's whole selector
    # contract is HIP-only now (require_device_visibility()/DeviceVisibility,
    # see PNRO17) -- reproducing the exact double-filtering bug that
    # contract exists to prevent. Stripped AFTER env_overrides is applied
    # (not just before) so a caller cannot reintroduce the hazard by
    # explicitly passing ROCR_VISIBLE_DEVICES in env_overrides either --
    # this key has no purpose in this harness any more, full stop.
    clean_env.pop("ROCR_VISIBLE_DEVICES", None)
    # RD58 (PA36 migration #4, dev-gpt-agent req_82fbbafe52c0472d
    # Q3/Q6): a multi-GPU producer (device=None) can name additional
    # env vars to unset beyond the always-stripped RROC/BIGCHERRY/
    # GGML_CUDA_DISABLE_FUSION set. Applied AFTER env_overrides so a
    # caller cannot reintroduce a stripped key by passing it in
    # env_overrides.
    for key in env_unset:
        clean_env.pop(key, None)

    raw_logs: list[dict[str, object]] = []

    def _make_runner(workload: str):
        def _runner(command: list[str]) -> "experiment_execution.RunnerOutput":
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                env=clean_env,
            )
            raw_logs.append(
                {
                    "workload": workload,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            if completed.returncode == 0:
                _require_real_gpu_execution(
                    completed.stdout,
                    completed.stderr,
                    context=f"{log_context} {workload} lane ({Path(command[0]).name})",
                )
            return experiment_execution.RunnerOutput(
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

        return _runner

    runs: dict[str, "experiment_execution.PairedLaneRun"] = {}
    commands: dict[str, dict[str, list[str]]] = {}
    if combined and len(workloads) > 1:
        # One llama-bench process per arm per paired round measures every
        # workload (one model load); each workload lane replays the same
        # outputs in the same deterministic round order.
        control_cmd = _combined_llama_bench_command(
            control_binary, model, workloads, patch_args=patch_args, runtime_args=runtime_args)
        subject_cmd = _combined_llama_bench_command(
            subject_binary, model, workloads, patch_args=patch_args, runtime_args=runtime_args)
        recorded: list[tuple[list[str], "experiment_execution.RunnerOutput"]] = []
        first = _make_runner("+".join(workloads))

        def _recording(command: list[str]):
            output = first(command)
            recorded.append((list(command), output))
            return output

        for index, workload in enumerate(workloads):
            if index == 0:
                runner = _recording
            else:
                replay = iter(recorded)

                def runner(command: list[str], _replay=replay):
                    expected, output = next(_replay)
                    if expected != list(command):
                        raise PatchCampaignError("combined llama-bench replay order diverged")
                    return output
            runs[workload] = experiment_execution.run_paired_lane(
                metric=_PAIRED_BENCH_METRIC_NAME[workload],
                control_command=control_cmd,
                subject_command=subject_cmd,
                pattern=_PAIRED_BENCH_METRIC_PATTERN[workload],
                pairs=pairs,
                runner=runner,
                execution_identity=execution_identity if index == 0 else None,
            )
            commands[workload] = {"control": control_cmd, "subject": subject_cmd}
        return PairedBenchmarkOutcome(runs=runs, commands=commands, raw_logs=raw_logs)
    for workload in workloads:
        control_cmd = _paired_llama_bench_command(
            control_binary,
            model,
            workload,
            patch_args=patch_args,
            runtime_args=runtime_args,
        )
        subject_cmd = _paired_llama_bench_command(
            subject_binary,
            model,
            workload,
            patch_args=patch_args,
            runtime_args=runtime_args,
        )
        runs[workload] = experiment_execution.run_paired_lane(
            metric=_PAIRED_BENCH_METRIC_NAME[workload],
            control_command=control_cmd,
            subject_command=subject_cmd,
            pattern=_PAIRED_BENCH_METRIC_PATTERN[workload],
            pairs=pairs,
            runner=_make_runner(workload),
            execution_identity=execution_identity,
        )
        commands[workload] = {"control": control_cmd, "subject": subject_cmd}

    return PairedBenchmarkOutcome(runs=runs, commands=commands, raw_logs=raw_logs)


BENCHMARK_EXECUTOR_FUNCS["paired-llama-bench-v1"] = run_paired_llama_benchmark


_STANDARD_BENCHMARK_MODEL_IDS: tuple[str, ...] = (
    "tierM-ministral14b-q4km",
    "tierB-qwen9b-q6k",
    "tierL-qwen27b-q8",
)


def _resolve_benchmark_architectures(args: argparse.Namespace, descriptor, cfg):
    """Return the architectures to benchmark (explicit, or recipe/patch intersection)."""
    requested_arches = tuple(args.benchmark_architecture or ())
    if requested_arches:
        architectures = requested_arches
    else:
        platform_targets = tuple(cfg.platforms["linux-multi"].targets)
        architectures = tuple(
            arch
            for arch in platform_targets
            if arch in descriptor.validation_architectures
        )
    if not architectures:
        raise PatchCampaignError(
            f"{descriptor.patch_id}: no applicable architecture -- pass --benchmark-architecture "
            "explicitly, or declare validation-architectures overlapping "
            "config/recipes.toml's platform.linux-multi targets"
        )
    return architectures


def _build_performance_binary_pair(
    args: argparse.Namespace, cfg, architectures
):
    """Materialize control/subject once and build ONE fat llama-bench pair.

    Returns (control_binary, subject_binary).
    """
    from bigcherry.patch import source as psi

    baseline_source = getattr(args, "baseline_source", "bigcherry")
    control_revision, control_composition = psi.resolve_source_composition(
        baseline_source,
        focal=None,
        base_ref=cfg.pinned,
        base_repo=LLAMA_CPP_SRC,
    )
    subject_revision, subject_composition = psi.resolve_source_composition(
        baseline_source,
        focal=args.patch,
        base_ref=cfg.pinned,
        base_repo=LLAMA_CPP_SRC,
    )
    if control_revision != subject_revision:
        raise RuntimeError(
            "control and subject source plans resolved different base revisions"
        )
    base_revision = subject_revision
    _print(
        f"performance benchmark: materializing control/subject source @ {base_revision[:12]} ..."
    )
    control_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC,
        worktree_root=args.worktree_root / "control",
        resolved_revision=base_revision,
        composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=cfg.pinned,
    )
    subject_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC,
        worktree_root=args.worktree_root / "subject",
        resolved_revision=base_revision,
        composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=cfg.pinned,
    )

    build_root: Path = (args.build_root or args.workdir) / subject_src.name
    exe = ".exe" if sys.platform == "win32" else ""
    build_env = _hip_env(args.hip_path)

    # User direction (2026-09-11): ONE fat multi-ISA control/subject
    # build covering every applicable architecture, not a separate build
    # per architecture -- AMDGPU_TARGETS accepts a semicolon-separated
    # list natively. GPT review (req_e608313764834497, 2026-09-11): the
    # legacy validation path builds control and validation-subject with
    # IDENTICAL extra_cmake_args=[] and asserts that parity
    # (assert_validation_subject_parity) precisely because differing
    # build instrumentation would confound a measured patch effect with
    # build-configuration noise -- both binaries here remain plain,
    # symmetric, autotune-instrumentation-free builds, just built once
    # for the joined target list instead of once per architecture.
    joined_targets = ";".join(architectures)
    control_bin = build_tree(
        name="perf-control",
        hip_path=args.hip_path,
        amdgpu_targets=joined_targets,
        workdir=build_root,
        targets=["llama-bench"],
        source=control_src,
        extra_cmake_args=[],
    )
    subject_bin = build_tree(
        name="perf-subject",
        hip_path=args.hip_path,
        amdgpu_targets=joined_targets,
        workdir=build_root,
        targets=["llama-bench"],
        source=subject_src,
        extra_cmake_args=[],
    )
    control_binary = control_bin / f"llama-bench{exe}"
    subject_binary = subject_bin / f"llama-bench{exe}"
    control_cmake_args = _full_requested_cmake_args(
        hip_path=args.hip_path,
        amdgpu_targets=joined_targets,
        extra_cmake_args=[],
    )
    control_build_evidence = capture_completed_build_evidence(
        build_root / "perf-control",
        source_root=control_src,
        architecture=joined_targets,
        binary=control_binary,
        requested_cmake_args=control_cmake_args,
        build_env=build_env,
    )
    subject_build_evidence = capture_completed_build_evidence(
        build_root / "perf-subject",
        source_root=subject_src,
        architecture=joined_targets,
        binary=subject_binary,
        requested_cmake_args=control_cmake_args,
        build_env=build_env,
    )
    assert_validation_subject_parity(
        control_build_evidence,
        subject_build_evidence,
        patch_id=args.patch,
    )
    return control_binary, subject_binary


def _run_performance_cell(
    args: argparse.Namespace,
    *,
    wiring,
    architecture: str,
    model_id: str,
    model_root: Path,
    device_map,
    control_binary: Path,
    subject_binary: Path,
) -> dict[str, object]:
    """Run (or skip) one architecture/model cell of the performance matrix."""
    from bigcherry.experiment import attestation
    from bigcherry.experiment.execution import require_device_visibility

    cell: dict[str, object] = {"architecture": architecture, "model": model_id}
    try:
        resolved_model = resolve_benchmark_model(
            model_id,
            model_root=model_root,
        )
    except PatchCampaignError as exc:
        cell.update(status="skipped", reason=f"model resolution failed: {exc}")
        return cell
    try:
        device_ids = resolve_device_pool(
            device_map,
            architecture,
            resolved_model.device_count,
        )
    except PatchCampaignError as exc:
        cell.update(status="skipped", reason=str(exc))
        return cell

    # Real-hardware finding (2026-09-11, PNRO17): only
    # HIP_VISIBLE_DEVICES is set here -- also setting
    # ROCR_VISIBLE_DEVICES to the same value actively breaks
    # non-prefix-from-0 device selection (confirmed on real
    # gfx1100/gfx1201/gfx1030 hardware during this matrix's own
    # merge-gate run). See require_device_visibility()/
    # DeviceVisibility's docstrings for the full mechanism.
    env_overrides = {"HIP_VISIBLE_DEVICES": ",".join(device_ids)}
    visibility = require_device_visibility(
        context=f"performance-benchmark {architecture}/{model_id}",
        env=env_overrides,
        exact_count=resolved_model.device_count,
    )
    execution_identity = attestation.ExecutionIdentity(
        backend="ROCm",
        architectures=(architecture,) * resolved_model.device_count,
    )
    executor_func = BENCHMARK_EXECUTOR_FUNCS[wiring.executor]
    outcome = executor_func(
        control_binary=control_binary,
        subject_binary=subject_binary,
        model=resolved_model.path,
        hip_path=args.hip_path,
        patch_args=wiring.patch_args,
        runtime_args=resolved_model.runtime_args,
        pairs=args.bench_repetitions,
        log_context=f"performance-benchmark {architecture}/{model_id}",
        env_overrides=env_overrides,
        execution_identity=execution_identity,
    )
    cell.update(
        status="executed",
        device_visibility=visibility.document(),
        commands=outcome.commands,
        raw_logs=outcome.raw_logs,
        metrics={
            workload: {"stats": run.stats, "runs": list(run.runs)}
            for workload, run in outcome.runs.items()
        },
    )
    return cell


def _run_performance_benchmark(args: argparse.Namespace, descriptor, cfg) -> int:
    """PVPS02 step 4: the generic --run-performance-benchmark entry point.

    Deliberately does NOT go through the legacy --model/--manifest/one-
    architecture tune/replay/stock/control/subject flow run() otherwise
    builds -- that flow is keyed to exactly one amdgpu-targets value and
    a single --model, neither of which fits a cross-architecture,
    cross-model matrix. This materializes source ONCE and builds ONE
    control/subject llama-bench binary pair covering every applicable
    architecture (AMDGPU_TARGETS=";".join(architectures), a single fat
    multi-ISA build -- cmake/HIP natively support a semicolon-separated
    target list), reused across every architecture/model cell instead of
    rebuilding per architecture. User direction (2026-09-11, after the
    prior per-architecture-build version's real-hardware merge gate
    passed on all 3 architectures): device SELECTION is still per-cell
    via HIP_VISIBLE_DEVICES (see the real-hardware finding on that
    mechanism above this function), only the BUILD is now shared.

    execution_identity/device-visibility enforcement is turned on here
    for the FIRST time in this module (steps 1-3 deliberately left it
    off) -- every cell fails closed before launch on a missing/
    insufficient/malformed --device-map entry for its architecture, and
    every measured process is attested for real architecture + device
    count (not physical card identity -- see DeviceVisibility's own
    docstring for why that distinction is load-bearing, never
    overclaimed in evidence)."""
    wiring = resolve_benchmark_wiring(descriptor)

    architectures = _resolve_benchmark_architectures(args, descriptor, cfg)

    model_ids = tuple(args.benchmark_model or _STANDARD_BENCHMARK_MODEL_IDS)
    device_map = parse_device_map(list(args.device_map or ()))
    model_root: Path = args.model_root

    cells: list[dict[str, object]] = []
    control_binary, subject_binary = _build_performance_binary_pair(
        args, cfg, architectures
    )

    for architecture in architectures:
        for model_id in model_ids:
            cells.append(
                _run_performance_cell(
                    args,
                    wiring=wiring,
                    architecture=architecture,
                    model_id=model_id,
                    model_root=model_root,
                    device_map=device_map,
                    control_binary=control_binary,
                    subject_binary=subject_binary,
                )
            )

    performance_doc = {
        "patch_id": descriptor.patch_id,
        "executor": wiring.executor,
        "architectures": list(architectures),
        "models": list(model_ids),
        "cells": cells,
    }
    performance_path = args.workdir / "performance-matrix.json"
    _atomic_write_json(performance_path, performance_doc)
    executed = [c for c in cells if c["status"] == "executed"]
    skipped = [c for c in cells if c["status"] == "skipped"]
    _print(
        f"performance benchmark matrix: {len(executed)} executed, {len(skipped)} skipped "
        f"(of {len(cells)} planned cells) -- {performance_path}"
    )
    for cell in skipped:
        _print(f"  skipped {cell['architecture']}/{cell['model']}: {cell['reason']}")
    return 0 if executed else 1
