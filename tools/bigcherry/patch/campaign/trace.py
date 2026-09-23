"""Trace-activation probes for the patch-validation campaign, including the
fail-closed real-GPU-execution guard."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from bigcherry.patch.activation import ActivationEvidence
from bigcherry.patch.campaign.build import _hip_env, _print, PatchCampaignError


def _trace_probe_env(*, hip_path: Path, disable_fusion: bool) -> dict[str, str]:
    env = _hip_env(hip_path)
    # A parent campaign/shell must not accidentally carry a dispatch mode
    # or an earlier trace/fusion setting into this isolated probe.
    for key in list(env):
        if key.startswith("GGML_HIP_DISPATCH_") or key.startswith("BIGCHERRY_"):
            env.pop(key, None)
    env["BIGCHERRY_PATCH_TRACE"] = "1"
    env["GGML_HIP_DISPATCH_MODE"] = "native"
    if disable_fusion:
        env["GGML_CUDA_DISABLE_FUSION"] = "1"
    else:
        env.pop("GGML_CUDA_DISABLE_FUSION", None)
    return env


# VA21 real-hardware finding: a llama.cpp binary whose ROCm/HIP device
# init fails silently falls back to CPU execution, but still prints a
# normal-looking llama-bench table under a "backend: ROCm" label with
# real-looking (if CPU-speed) numbers -- a benchmark/trigger-probe run's
# exit code and printed metrics alone cannot be trusted as proof that it
# actually executed on the GPU. Fail closed: require the real, positive
# "ggml_cuda_init: found N ROCm devices" line and reject any known
# ROCm-init-failure signature, rather than accepting output that never
# actually touched the GPU (which is exactly what let RD08's real
# subject_hit=false confound with a genuine dispatch-routing question on
# 2026-09-01 -- gfx1030's HIP runtime failed to detect the device at all,
# and nothing caught it before the trigger check quietly "ran" and
# reported a negative).
_ROCM_INIT_FAILURE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"failed to initialize ROCm"),
    re.compile(r"no ROCm-capable device is detected"),
    re.compile(r"hipErrorNoDevice"),
)


_ROCM_INIT_SUCCESS_PATTERN = re.compile(
    r"ggml_cuda_init:\s*found\s+(\d+)\s+ROCm device"
)


def _require_real_gpu_execution(stdout: str, stderr: str, *, context: str) -> None:
    """Fail closed unless the combined output carries real, positive
    evidence of ROCm/HIP device initialization (not merely the absence of
    an error) -- see the module comment above this function for why
    absence-of-failure is not itself sufficient evidence here."""
    combined = f"{stdout}\n{stderr}"
    for pattern in _ROCM_INIT_FAILURE_PATTERNS:
        if pattern.search(combined):
            raise PatchCampaignError(
                f"{context}: ROCm/HIP device initialization failed -- real GPU execution "
                f"cannot be confirmed (matched {pattern.pattern!r} in the process output)"
            )
    match = _ROCM_INIT_SUCCESS_PATTERN.search(combined)
    if match is None or int(match.group(1)) < 1:
        raise PatchCampaignError(
            f"{context}: no real GPU execution evidence in the process output -- expected a "
            "'ggml_cuda_init: found N ROCm devices' line; a benchmark/probe that silently ran "
            "CPU-only must never be accepted as real hardware evidence"
        )


def _run_one_trace_probe(
    *,
    name: str,
    binary: Path,
    model: Path,
    hip_path: Path,
    workdir: Path,
    bench_prompt: int,
    bench_gen: int,
    disable_fusion: bool,
    extra_flags: tuple[str, ...] = (),
    env_overrides: Mapping[str, str] | None = None,
    env_unset: tuple[str, ...] = (),
) -> str:
    import subprocess

    binary = Path(binary)
    model = Path(model)
    if not binary.is_file():
        raise PatchCampaignError(f"activation probe binary does not exist: {binary}")
    if not model.is_file():
        raise PatchCampaignError(f"activation probe model does not exist: {model}")

    # VA21 real-hardware finding (2026-09-01): llama-bench.cpp itself gates
    # ggml's log level on its OWN --verbose flag (GGML_LOG_LEVEL_DEBUG if
    # verbose else GGML_LOG_LEVEL_ERROR) -- without it, BOTH GGML_LOG_INFO
    # and GGML_LOG_WARN are filtered before this probe's whole reason for
    # existing (observing a log-based activation marker) ever has a chance.
    # Confirmed directly: the exact same binary/env only emits
    # BIGCHERRY_PATCH_HIT with --verbose present. This function's entire
    # purpose is reading ggml log output, so it must always request it.
    command = [
        str(binary.resolve()),
        "-m",
        str(model.resolve()),
        "-p",
        str(bench_prompt),
        "-n",
        str(bench_gen),
        "-r",
        "1",
        "-ngl",
        "99",
        "--verbose",
        *extra_flags,
    ]
    env = _trace_probe_env(hip_path=hip_path, disable_fusion=disable_fusion)
    # PA36 RD13/1206 migration (GPT req_760c0fe82d7b4609 MAJOR): a
    # producer trace probe must run on the SAME physical device as the
    # correctness measurement -- apply the selected device's HIP-only
    # selector overrides, then its unsets LAST (the sanctioned selector
    # always wins; never ambient visibility).
    if env_overrides:
        env.update(env_overrides)
    for key in env_unset:
        env.pop(key, None)

    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"activation-{name}.log"

    completed = subprocess.run(
        command,
        cwd=workdir,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = completed.stdout + "\n" + completed.stderr
    log_path.write_text(
        f"command: {command!r}\n"
        f"GGML_CUDA_DISABLE_FUSION={env.get('GGML_CUDA_DISABLE_FUSION')!r}\n\n"
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise PatchCampaignError(
            f"activation probe {name!r} failed with exit code {completed.returncode}; "
            f"see {log_path}"
        )
    _require_real_gpu_execution(
        completed.stdout,
        completed.stderr,
        context=f"activation probe {name!r}",
    )
    return combined


def run_trace_activation_probes(
    *,
    marker_regex: str | None,
    description: str | None,
    binary: Path,
    model: Path,
    hip_path: Path,
    workdir: Path,
    bench_prompt: int,
    bench_gen: int,
    env_overrides: Mapping[str, str] | None = None,
    env_unset: tuple[str, ...] = (),
) -> tuple[ActivationEvidence, dict[str, object]] | None:
    """Run positive + fusion-disabled negative-control activation probes.

    Returns None for patches which do not use the shared trace-marker
    mechanism.

    Positive: BIGCHERRY_PATCH_TRACE=1 -- expected marker MUST occur to
    prove the patch's path executed.

    Negative control: BIGCHERRY_PATCH_TRACE=1 + GGML_CUDA_DISABLE_FUSION=1
    -- marker MUST NOT occur. A marker that survives the negative control
    is not trustworthy activation evidence (it would fire regardless of
    whether the specific fusion this patch adds actually ran) and is
    classified unobservable rather than executed.
    """
    if not marker_regex or not description:
        return None

    pattern = re.compile(marker_regex)

    _print(f"activation probe: {description} (positive)")
    positive_output = _run_one_trace_probe(
        name="positive",
        binary=binary,
        model=model,
        hip_path=hip_path,
        workdir=workdir,
        bench_prompt=bench_prompt,
        bench_gen=bench_gen,
        disable_fusion=False,
        env_overrides=env_overrides,
        env_unset=env_unset,
    )
    positive_hit = pattern.search(positive_output) is not None

    _print(f"activation probe: {description} (fusion-disabled control)")
    negative_output = _run_one_trace_probe(
        name="fusion-disabled",
        binary=binary,
        model=model,
        hip_path=hip_path,
        workdir=workdir,
        bench_prompt=bench_prompt,
        bench_gen=bench_gen,
        disable_fusion=True,
        env_overrides=env_overrides,
        env_unset=env_unset,
    )
    negative_hit = pattern.search(negative_output) is not None

    if negative_hit:
        evidence = ActivationEvidence(
            status="unobservable",
            mechanism="BIGCHERRY_PATCH_TRACE two-probe control",
            detail=(
                f"{description}: marker was present even with GGML_CUDA_DISABLE_FUSION=1. "
                "The marker therefore does not uniquely prove execution of the intended "
                "fusion path."
            ),
        )
    elif positive_hit:
        evidence = ActivationEvidence(
            status="executed",
            mechanism="BIGCHERRY_PATCH_TRACE two-probe control",
            detail=(
                f"{description}: expected marker was observed with fusion enabled and "
                "was absent with GGML_CUDA_DISABLE_FUSION=1."
            ),
        )
    else:
        evidence = ActivationEvidence(
            status="not_executed",
            mechanism="BIGCHERRY_PATCH_TRACE two-probe control",
            detail=(
                f"{description}: expected marker was absent from the positive probe and "
                "remained absent in the fusion-disabled control. This model/workload did not "
                "prove execution of the patch path."
            ),
        )

    detail: dict[str, object] = {
        "description": description,
        "marker_regex": marker_regex,
        "positive": {
            "BIGCHERRY_PATCH_TRACE": "1",
            "GGML_CUDA_DISABLE_FUSION": None,
            "marker_observed": positive_hit,
            "log": "logs/activation-positive.log",
        },
        "negative_control": {
            "BIGCHERRY_PATCH_TRACE": "1",
            "GGML_CUDA_DISABLE_FUSION": "1",
            "marker_observed": negative_hit,
            "log": "logs/activation-fusion-disabled.log",
        },
    }
    return evidence, detail
