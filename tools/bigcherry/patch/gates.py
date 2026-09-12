"""Shared patch-gate contracts.

This module is intentionally a thin composition layer.  Gate decisions must
come from the existing registry, patchset, evidence, disposition, rebase,
validation-policy, and admission authorities; this file owns only the
immutable request/result types and their ordered composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from ..core import paths
from . import catalog as patch_catalog
from . import patchset
from . import registry as patch_registry


class GateId(str, Enum):
    G0 = "G0"
    G1 = "G1"
    G2 = "G2"
    G3 = "G3"
    G4 = "G4"
    G5 = "G5"
    G6 = "G6"
    G7 = "G7"


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NA = "NA"


class GateIntent(str, Enum):
    AUTHOR = "author"
    VALIDATE = "validate"
    PROMOTE = "promote"
    BUILD = "build"
    REBASE = "rebase"


@dataclass(frozen=True, slots=True)
class GateResult:
    id: GateId
    status: GateStatus
    phase: str
    authority: str
    detail: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GateContext:
    descriptor: patch_registry.PatchDescriptor
    composition: patchset.ResolvedPatchSet
    pinned_ref: str
    intent: GateIntent
    patch_context: patch_catalog.PatchContext
    patches_dir: Path
    resolved_base_revision: str | None = None
    catalog_path: Path | None = None
    evidence_root: Path | None = None
    external_sources_path: Path | None = None
    validation_baseline_path: Path | None = None
    dispositions_dir: Path = paths.DISPOSITIONS
    source_root: Path | None = None
    rebase_report: Mapping[str, Any] | None = None
    allow_legacy_grandfather: bool = True


def gate_applies(gate_id: GateId, intent: GateIntent) -> bool:
    """Return the fixed PA21 applicability matrix."""
    applicable = {
        GateIntent.AUTHOR: frozenset((GateId.G0, GateId.G1)),
        GateIntent.VALIDATE: frozenset((GateId.G0, GateId.G1, GateId.G2, GateId.G3)),
        GateIntent.PROMOTE: frozenset((GateId.G0, GateId.G1, GateId.G2, GateId.G3, GateId.G4, GateId.G5)),
        GateIntent.REBASE: frozenset((GateId.G0, GateId.G1, GateId.G2, GateId.G6)),
        GateIntent.BUILD: frozenset((GateId.G0, GateId.G1, GateId.G2, GateId.G4, GateId.G6, GateId.G7)),
    }
    return gate_id in applicable[intent]
