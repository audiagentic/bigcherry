"""Campaign operation rehydration policy.

Kept as a separate seam so future stage runners do not infer reuse from file
existence.  Reuse means an exact succeeded result whose spec, verified input
bindings and output bytes still match.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .operations import (
    ArtifactBinding,
    OperationResult,
    OperationSpec,
    durable_state,
    rehydrate_succeeded,
)


def rehydrate_result(
    stage_root: Path,
    spec: OperationSpec,
    inputs: Iterable[ArtifactBinding],
) -> OperationResult | None:
    return rehydrate_succeeded(stage_root, spec, inputs)


__all__ = ["durable_state", "rehydrate_result"]
