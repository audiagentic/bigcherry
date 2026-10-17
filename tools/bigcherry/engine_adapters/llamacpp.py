"""llama.cpp: GGML_HIP_DISPATCH builds, the dispatch modes, and a parsed device-execution attestation."""

from __future__ import annotations

from pathlib import Path
import json
from typing import TYPE_CHECKING, Any

from .base import EngineAdapter, SidecarSpec, SourceCheck

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

    def source_checks(self, context: ProjectContext) -> tuple[SourceCheck, ...]:
        # The strict audit encodes the ggml HIP/CUDA invariants of the llama.cpp tree and its overlay files.
        from ..source import audit as source_audit

        def run() -> str | None:
            report = source_audit.audit(context.vendor_root, engine=context.engine)
            return None if source_audit.passed(report, strict=True) else json.dumps(report, sort_keys=True)

        return (SourceCheck("source-audit", "default", run),)
