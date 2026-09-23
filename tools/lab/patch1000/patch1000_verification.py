"""patch 1000 (1000_rdna4_mmq_q2k_q6k_fix) backend-ops + llama-bench
verification helpers (PA35).

Lab code, not a package and never imported by production: moved verbatim out
of bigcherry.patch.validation_campaign by PA43 (lab/non-package boundary in
docs/reference/tooling/TOOLING.md). Nothing in the production CLI ever
dispatched to these functions; the PA35 driver (run_pa35_step1.py, next to
this file) and tools/tests/patch/test_patch1000_verification.py load it by
path.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from bigcherry.core.paths import REPO_ROOT
from bigcherry.experiment.attestation import ExecutionIdentity
from bigcherry.patch.campaign.benchmark import run_paired_llama_benchmark
from bigcherry.patch.campaign.build import (
    _hip_env,
    build_tree,
    LLAMA_CPP_SRC,
    PatchCampaignError,
)
from bigcherry.patch.campaign.trace import _require_real_gpu_execution


# PA35: real verification producer for patch 1000_rdna4_mmq_q2k_q6k_fix.
# Its own README documents an unverified gap (2026-09-11 process audit):
# the patch's claimed Q6_K 1.90x / Q2_K 28.2x gains are entirely upstream
# PR #25940's own reported numbers -- no independent measurement on this
# project's own hardware exists. Those upstream numbers are themselves
# test-backend-ops `perf -o MUL_MAT` FLOP/s throughput at n=512, not
# llama-bench token/s -- so the exact-shape microbenchmark below
# reproduces the upstream claim's own methodology, and the llama-bench
# lane is a separate, model-level analogue (prefill ~= matrix n=512),
# not a re-statement of the same number.
_PATCH1000_ID = "1000_rdna4_mmq_q2k_q6k_fix"


_PATCH1000_QUANT_NAMES = {
    "Q2_K": "q2_K",
    "Q6_K": "q6_K",
}


_PATCH1000_PERF_TIME_PATTERN = re.compile(
    r"MUL_MAT\([^)]*\):\s+\d+\s+runs\s+-\s+([0-9.]+)\s+us/run"
)


def _patch1000_backend_ops_command(
    binary: Path,
    quant: str,
    *,
    mode: str,
) -> list[str]:
    try:
        type_name = _PATCH1000_QUANT_NAMES[quant]
    except KeyError:
        raise PatchCampaignError(
            f"patch1000: unsupported quant {quant!r}; "
            f"expected one of {tuple(_PATCH1000_QUANT_NAMES)}"
        ) from None

    if mode == "perf":
        # The exact n=512 MUL_MAT shape corresponding to upstream PR
        # #25940's Q2_K/Q6_K n=512 comparison.
        params = f"type_a={type_name},type_b=f32,m=4096,n=512,k=14336"
    elif mode == "test":
        # Correctness mode filters all real MUL_MAT cases for this
        # quant/F32 activation pair rather than assuming the perf-only
        # 4096x512x14336 case is also registered in the correctness corpus.
        params = f"type_a={type_name},type_b=f32"
    else:
        raise PatchCampaignError(
            f"patch1000: unsupported test-backend-ops mode {mode!r}"
        )

    return [str(binary), mode, "-o", "MUL_MAT", "-p", params]


def _patch1000_run_env(
    hip_path: Path,
    env_overrides: dict[str, str],
) -> dict[str, str]:
    from bigcherry.campaign.benchmark import sanitize_environment

    env = sanitize_environment(_hip_env(hip_path), mode="stock")
    for key in list(env):
        if key.startswith("BIGCHERRY_") or key == "GGML_CUDA_DISABLE_FUSION":
            env.pop(key, None)
    env.update(env_overrides)
    # ROCR_VISIBLE_DEVICES + HIP_VISIBLE_DEVICES to the same index
    # double-filters to zero devices (STANDARDIZED_PATCH_VALIDATION_
    # CRITERIA.md's known trap) -- HIP_VISIBLE_DEVICES alone is sufficient.
    env.pop("ROCR_VISIBLE_DEVICES", None)
    return env


def run_patch1000_backend_ops_perf(
    *,
    control_binary: Path,
    subject_binary: Path,
    quant: str,
    hip_path: Path,
    env_overrides: dict[str, str],
    pairs: int = 5,
    execution_identity: object | None = None,
) -> dict[str, object]:
    """Paired exact-shape reproduction of PR #25940's n=512 claim."""
    from bigcherry.experiment import execution as experiment_execution

    env = _patch1000_run_env(hip_path, env_overrides)
    experiment_execution.require_device_visibility(
        context=f"patch1000 {quant} backend-ops perf",
        env=env,
        exact_count=1,
    )

    control_command = _patch1000_backend_ops_command(control_binary, quant, mode="perf")
    subject_command = _patch1000_backend_ops_command(subject_binary, quant, mode="perf")

    raw_logs: list[dict[str, object]] = []

    def runner(command: list[str]) -> "experiment_execution.RunnerOutput":
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        raw_logs.append(
            {
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
                context=f"patch1000 {quant} backend-ops perf ({Path(command[0]).name})",
            )
        return experiment_execution.RunnerOutput(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    paired = experiment_execution.run_paired_lane(
        metric="time_us",
        control_command=control_command,
        subject_command=subject_command,
        pattern=_PATCH1000_PERF_TIME_PATTERN,
        pairs=pairs,
        runner=runner,
        lower_is_better=True,
        execution_identity=execution_identity,
    )

    stats = dict(paired.stats)
    # lower_is_better=True's effect_pct is control_time/subject_time - 1;
    # convert to a direct speedup multiplier comparable with 28.2x/1.90x.
    effect_pct = float(stats["geometric_effect_pct"])
    ci_low_pct = float(stats["ci95_low_pct"])
    ci_high_pct = float(stats["ci95_high_pct"])
    stats["geometric_speedup_x"] = 1.0 + effect_pct / 100.0
    stats["ci95_low_x"] = 1.0 + ci_low_pct / 100.0
    stats["ci95_high_x"] = 1.0 + ci_high_pct / 100.0

    return {
        "quant": quant,
        "shape": {"op": "MUL_MAT", "type_b": "f32", "m": 4096, "n": 512, "k": 14336},
        "control_command": control_command,
        "subject_command": subject_command,
        "run": asdict(paired),
        "stats": stats,
        "raw_logs": raw_logs,
    }


def run_patch1000_backend_ops_correctness(
    *,
    binary: Path,
    quant: str,
    hip_path: Path,
    env_overrides: dict[str, str],
    log_context: str,
    execution_identity: object | None = None,
) -> dict[str, object]:
    """Real CPU-reference/NMSE MUL_MAT correctness check for one arm."""
    from bigcherry.experiment import attestation
    from bigcherry.experiment import execution as experiment_execution

    env = _patch1000_run_env(hip_path, env_overrides)
    experiment_execution.require_device_visibility(
        context=log_context,
        env=env,
        exact_count=1,
    )
    command = _patch1000_backend_ops_command(binary, quant, mode="test")

    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, env=env
    )
    if completed.returncode != 0:
        raise PatchCampaignError(
            f"{log_context}: test-backend-ops exited "
            f"{completed.returncode}; stderr={completed.stderr[-1000:]!r}"
        )

    _require_real_gpu_execution(completed.stdout, completed.stderr, context=log_context)

    if execution_identity is not None:
        attestation.require_execution_identity(
            execution_identity,
            attestation.parse_rocm_attestation(
                completed.stdout + "\n" + completed.stderr
            ),
            context=log_context,
        )

    return {
        "passed": True,
        "quant": quant,
        "mechanism": "test-backend-ops MUL_MAT CPU-reference/NMSE",
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_patch1000_verification(
    *,
    base_revision: str,
    hip_path: Path,
    build_root: Path,
    q2k_model: Path,
    q6k_model: Path,
    devices: Mapping[str, str],
    architectures: tuple[str, ...] = ("gfx1100", "gfx1201", "gfx1030"),
    pairs: int = 5,
    _source_module=None,
    _build_func=None,
    _perf_func=None,
    _correctness_func=None,
    _bench_func=None,
    _recipes_path: Path | None = None,
) -> dict[str, object]:
    """PA35 audit producer for patch 1000.

    Builds stock/control/subject ONCE as fat multi-arch trees, then varies
    only HIP_VISIBLE_DEVICES during execution (STANDARDIZED_PATCH_
    VALIDATION_CRITERIA.md's "build once, run per-device" rule).

      stock   = pinned upstream, no overlay/patches
      control = BigCherry overlay + serving-core, patch 1000 absent
      subject = control + patch 1000

    The subject composition is additionally required to equal the current
    named bigcherry-serving-base composition -- if config later gains another
    upstream-fix, fail closed rather than silently changing this audit's
    meaning.

    PA29 cutover (GPT design review req_964ec5fc21c14848): this is live
    producer code, not PA26/PA30 historical comparison apparatus, so it
    migrates with every other real consumer -- framework -> serving-core,
    bigcherry-native -> bigcherry-serving-base.
    """
    import tomllib

    from bigcherry.experiment import execution as experiment_execution
    from bigcherry.patch import source as real_source

    if pairs < 3:
        raise PatchCampaignError("patch1000: pairs must be >= 3")
    if not architectures or len(set(architectures)) != len(architectures):
        raise PatchCampaignError(
            "patch1000: architectures must be non-empty and unique"
        )

    missing_devices = [a for a in architectures if a not in devices]
    if missing_devices:
        raise PatchCampaignError(
            "patch1000: missing device selectors for " + ", ".join(missing_devices)
        )

    for model in (q2k_model, q6k_model):
        if not model.is_file():
            raise PatchCampaignError(f"patch1000: model file does not exist: {model}")

    psi = _source_module or real_source
    build_func = _build_func or build_tree
    perf_func = _perf_func or run_patch1000_backend_ops_perf
    correctness_func = _correctness_func or run_patch1000_backend_ops_correctness
    bench_func = _bench_func or run_paired_llama_benchmark

    recipes_path = _recipes_path or (REPO_ROOT / "config" / "recipes.toml")
    recipes = tomllib.loads(recipes_path.read_text(encoding="utf-8"))
    try:
        serving_core_ids = tuple(recipes["patch-set"]["serving-core"]["patches"])
    except (KeyError, TypeError):
        raise PatchCampaignError(
            "patch1000: recipes.toml has no valid patch-set.serving-core"
        ) from None

    if _PATCH1000_ID in serving_core_ids:
        raise PatchCampaignError(
            "patch1000: focal patch unexpectedly appears in serving-core"
        )

    stock_revision, stock_composition = psi.resolve_source_composition(
        "llama-native",
        base_ref=base_revision,
        base_repo=LLAMA_CPP_SRC,
    )
    control_revision, control_composition = psi.resolve_source_composition(
        "llama-native",
        extra_patches=serving_core_ids,
        base_ref=base_revision,
        base_repo=LLAMA_CPP_SRC,
    )
    subject_revision, subject_composition = psi.resolve_source_composition(
        "llama-native",
        extra_patches=(*serving_core_ids, _PATCH1000_ID),
        base_ref=base_revision,
        base_repo=LLAMA_CPP_SRC,
    )
    production_revision, production_composition = psi.resolve_source_composition(
        "bigcherry-serving-base",
        base_ref=base_revision,
        base_repo=LLAMA_CPP_SRC,
    )

    revisions = {
        stock_revision,
        control_revision,
        subject_revision,
        production_revision,
    }
    if len(revisions) != 1:
        raise PatchCampaignError(
            "patch1000: validation arms resolved different base revisions"
        )

    if subject_composition != production_composition:
        raise PatchCampaignError(
            "patch1000: serving-core + focal no longer equals the current bigcherry-serving-base "
            "composition; update this audit producer rather than silently including/"
            "excluding another upstream fix"
        )

    source_root = build_root / "patch1000-sources"
    stock_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC,
        worktree_root=source_root / "stock",
        resolved_revision=stock_revision,
        composition=stock_composition,
        overlay_root=None,
        requested_revision=base_revision,
    )
    control_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC,
        worktree_root=source_root / "control",
        resolved_revision=control_revision,
        composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=base_revision,
    )
    subject_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC,
        worktree_root=source_root / "subject",
        resolved_revision=subject_revision,
        composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=base_revision,
    )

    # Load-bearing PA35 rule: these builds are outside the per-device loop.
    fat_targets = ";".join(architectures)
    build_workdir = build_root / "patch1000-builds"
    build_args = {
        "hip_path": hip_path,
        "amdgpu_targets": fat_targets,
        "workdir": build_workdir,
        "targets": ["test-backend-ops", "llama-bench"],
        "extra_cmake_args": [],
    }

    arm_sources = {"stock": stock_src, "control": control_src, "subject": subject_src}
    arm_bins: dict[str, Path] = {}
    for arm, source in arm_sources.items():
        arm_bins[arm] = build_func(name=f"patch1000-{arm}", source=source, **build_args)

    exe = ".exe" if sys.platform == "win32" else ""
    backend_ops = {
        arm: bin_dir / f"test-backend-ops{exe}" for arm, bin_dir in arm_bins.items()
    }
    llama_bench = {
        arm: bin_dir / f"llama-bench{exe}" for arm, bin_dir in arm_bins.items()
    }

    comparisons = {
        "stock_vs_control": ("stock", "control"),
        "control_vs_subject": ("control", "subject"),
        "stock_vs_subject": ("stock", "subject"),
    }
    models = {"Q2_K": q2k_model, "Q6_K": q6k_model}

    cells: dict[str, object] = {}
    for architecture in architectures:
        selector = devices[architecture]
        selector_env = {"HIP_VISIBLE_DEVICES": selector}
        visibility = experiment_execution.require_device_visibility(
            context=f"patch1000 {architecture}",
            env=selector_env,
            exact_count=1,
        )
        expected_execution = ExecutionIdentity(
            backend="ROCm", architectures=(architecture,)
        )

        arch_doc: dict[str, object] = {
            "architecture": architecture,
            "device_visibility": visibility.document(),
            "correctness": {},
            "microbenchmark": {},
            "llama_bench": {},
        }

        for quant in ("Q2_K", "Q6_K"):
            correctness_doc: dict[str, object] = {}
            for arm in ("control", "subject"):
                correctness_doc[arm] = correctness_func(
                    binary=backend_ops[arm],
                    quant=quant,
                    hip_path=hip_path,
                    env_overrides=selector_env,
                    log_context=f"patch1000 {architecture} {quant} {arm} correctness",
                    execution_identity=expected_execution,
                )
            arch_doc["correctness"][quant] = correctness_doc

            perf_doc: dict[str, object] = {}
            for comparison, (left, right) in comparisons.items():
                perf_doc[comparison] = perf_func(
                    control_binary=backend_ops[left],
                    subject_binary=backend_ops[right],
                    quant=quant,
                    hip_path=hip_path,
                    env_overrides=selector_env,
                    pairs=pairs,
                    execution_identity=expected_execution,
                )
            arch_doc["microbenchmark"][quant] = perf_doc

            bench_doc: dict[str, object] = {}
            for comparison, (left, right) in comparisons.items():
                outcome = bench_func(
                    control_binary=llama_bench[left],
                    subject_binary=llama_bench[right],
                    model=models[quant],
                    hip_path=hip_path,
                    workloads=("prefill", "decode"),
                    pairs=pairs,
                    log_context=f"patch1000 {architecture} {quant} {comparison}",
                    env_overrides=selector_env,
                    execution_identity=expected_execution,
                )
                bench_doc[comparison] = asdict(outcome)
            arch_doc["llama_bench"][quant] = bench_doc

        cells[architecture] = arch_doc

    return {
        "schema_version": 1,
        "producer": "patch1000-verification-v1",
        "patch": _PATCH1000_ID,
        "base_revision": base_revision,
        "compiled_targets": list(architectures),
        "amdgpu_targets": fat_targets,
        "pairs": pairs,
        "models": {
            quant: {"path": str(model), "size_bytes": model.stat().st_size}
            for quant, model in models.items()
        },
        "compositions": {
            "stock": list(stock_composition),
            "control": list(control_composition),
            "subject": list(subject_composition),
        },
        "cells": cells,
    }
