"""Per-engine adapters for generic campaign tooling (MEN11). With core/engines.py, the place that names engines."""

from __future__ import annotations

from .base import (
    BumpCheck, BumpCheckError, CompletionApi, CpuSmokeSpec, DeviceAssertion, DriftCheck, EngineAdapter, QualityCommand, SidecarError, SidecarSet, SidecarSpec,
    SourceCheck,
)
from .llamacpp import LlamaCppAdapter
from .radiance import RadianceAdapter

_ADAPTERS: dict[str, EngineAdapter] = {
    "llamacpp": LlamaCppAdapter(),
    "radiance": RadianceAdapter(),
}


def adapter(engine: str) -> EngineAdapter:
    try:
        return _ADAPTERS[engine]
    except KeyError:
        raise ValueError(f"no engine adapter for {engine!r}; known: {', '.join(sorted(_ADAPTERS))}") from None


__all__ = ["BumpCheck", "BumpCheckError", "CompletionApi", "CpuSmokeSpec", "DeviceAssertion", "DriftCheck", "EngineAdapter", "QualityCommand", "SidecarError", "SidecarSet", "SidecarSpec", "SourceCheck", "adapter"]
