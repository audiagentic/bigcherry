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
from . import docs as patch_docs
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


def evaluate_composition_gate(context: GateContext) -> GateResult:
    """Evaluate G0 using the canonical exact-composition resolver."""
    ids = tuple(module.patch_id for module in context.composition.modules)
    try:
        resolved = patchset.resolve_exact(
            ids, directory=context.patches_dir, allow_rejected=False,
        )
    except (OSError, TypeError, ValueError) as exc:
        return GateResult(
            GateId.G0, GateStatus.BLOCKED, "composition", "patchset.resolve_exact",
            (f"composition could not be resolved: {exc}",),
        )
    expected = tuple((module.patch_id, module.content_hash) for module in context.composition.modules)
    actual = tuple((module.patch_id, module.content_hash) for module in resolved.modules)
    if actual != expected:
        return GateResult(
            GateId.G0, GateStatus.BLOCKED, "composition", "patchset.resolve_exact",
            ("resolved composition identity differs from the supplied composition",),
        )
    return GateResult(GateId.G0, GateStatus.PASS, "composition", "patchset.resolve_exact")


def evaluate_summary_gate(context: GateContext) -> GateResult:
    """Evaluate focal SUMMARY consistency through the scoped authority."""
    try:
        problems = patch_docs.check_summary_for_patch(
            context.descriptor,
            context.patches_dir,
        )
    except (OSError, TypeError, ValueError) as exc:
        return GateResult(GateId.G1, GateStatus.BLOCKED, "documentation", "patch.docs", (str(exc),))
    if problems:
        return GateResult(GateId.G1, GateStatus.FAIL, "documentation", "patch.docs", problems)
    return GateResult(GateId.G1, GateStatus.PASS, "documentation", "patch.docs")
