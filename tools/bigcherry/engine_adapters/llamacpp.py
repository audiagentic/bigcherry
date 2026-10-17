"""llama.cpp: GGML_HIP_DISPATCH builds, the dispatch modes, and a parsed device-execution attestation."""

from __future__ import annotations

from pathlib import Path
import json
from typing import TYPE_CHECKING, Any

from .base import (
    CompletionApi, CpuSmokeSpec, DeviceAssertion, EngineAdapter, QualityCommand, SidecarSpec, SourceCheck, SMOKE_SEED,
    build_cpu_smoke,
)

if TYPE_CHECKING:
    from ..core.context import ProjectContext


class LlamaCppAdapter(EngineAdapter):
    parses_execution_attestation = True

    def dispatch_modes(self) -> tuple[str, ...]:
        return ("stock", "native", "record", "tune", "replay")

    def inspect_build(self, build_dir: Path) -> dict[str, Any] | None:
        # The dispatch-instrumentation contract is llama.cpp (GGML_HIP_DISPATCH builds).
        from ..build.builds import inspect_dispatch_build
        return inspect_dispatch_build(build_dir)

    def check_arm_environment(self, controls: dict[str, str]) -> None:
        if any(
            key != "GGML_CUDA_ALLREDUCE"
            and not key.startswith(("GGML_HIP_DISPATCH_", "GGML_HIP_TUNE_", "GGML_HIP_AUTOTUNE_", "GGML_CUDA_AR_",
                                    "NCCL_", "RCCL_"))
            and not (key.startswith("BIGCHERRY_") and key != "BIGCHERRY_PATCH_TRACE")
            for key in controls
        ):
            raise ValueError("arm-specific environment is limited to dispatch/tuning/allreduce/RCCL and BigCherry patch controls (not tracing); topology belongs to the shared environment")
        if "GGML_HIP_DISPATCH_MODE" in controls:
            raise ValueError("arm environment cannot override the declared dispatch mode")

    def sidecars(self, arm: dict[str, Any], env: dict[str, str], run_dir: Path, binary: Path) -> list[SidecarSpec]:
        return []

    def quality_command(self, kind, arm, env, run_dir, binary, model, quality) -> QualityCommand | None:
        return None  # no reference-based KL mode is wired for llama.cpp yet

    def cpu_smoke(self, context: ProjectContext, build_dir: Path, model_root: Path) -> CpuSmokeSpec | None:
        """llama-server with no layer offloaded (-ngl 0) and no device listed (--device none: ``-dev``/``--device`` in
        common/arg.cpp, "none = don't offload"), on top of the emptied HIP/ROCR/CUDA device variables. ``/completion``
        takes ``n_predict`` and replies with ``content``; ``cache_prompt`` is off so the second pass recomputes."""
        return build_cpu_smoke(
            context, build_dir, model_root,
            engine_args=("-ngl", "0", "--device", "none"), extra_env={}, env_unset_prefixes=("GGML_", "LLAMA_ARG_", "LLAMA_SERVER_", "BIGCHERRY_"),
            completion=CompletionApi(
                path="/completion", prompt_field="prompt", token_field="n_predict",
                fixed=(("temperature", 0), ("seed", SMOKE_SEED), ("cache_prompt", False)),
                text_path=("content",),
            ),
            # ggml-cuda/hip init prints "found N ROCm devices" and a "Device 0:" line only when it opened a device; a
            # layer on a GPU shows as "offloaded K/N layers to GPU" with K > 0.
            device=DeviceAssertion((
                r"ggml_cuda_init: found [1-9]\d* (?:ROCm|CUDA) devices?",
                r"^\s*Device \d+: .*(?:AMD|gfx\d+|NVIDIA)",
                r"offloaded [1-9]\d*/\d+ layers to GPU",
            )),
        )

    def source_checks(self, context: ProjectContext) -> tuple[SourceCheck, ...]:
        # The strict audit encodes the ggml HIP/CUDA invariants of the llama.cpp tree and its overlay files.
        from ..source import audit as source_audit

        def run() -> str | None:
            report = source_audit.audit(context.vendor_root, engine=context.engine)
            return None if source_audit.passed(report, strict=True) else json.dumps(report, sort_keys=True)

        return (SourceCheck("source-audit", "default", run),)
