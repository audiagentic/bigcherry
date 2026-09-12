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
from . import disposition as patch_disposition
from . import rebase
from . import validation_policy
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
    catalog_states: Mapping[str, str] | None = None
    coverage_report: Mapping[str, Any] | None = None
    recipe_patch_ids: frozenset[str] = frozenset()
    target_revision: str | None = None


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


def evaluate_rebase_gate(context: GateContext) -> GateResult:
    """Evaluate G2 through the canonical rebase freshness authority."""
    if context.rebase_report is None:
        return GateResult(
            GateId.G2, GateStatus.BLOCKED, "rebase", "patch.rebase",
            ("no rebase report was supplied",),
        )
    try:
        known_good = rebase.require_fresh_report(context.rebase_report, context.patches_dir)
    except (OSError, TypeError, ValueError, rebase.StaleRebaseReportError) as exc:
        return GateResult(GateId.G2, GateStatus.BLOCKED, "rebase", "patch.rebase", (str(exc),))
    if context.descriptor.patch_id not in known_good:
        return GateResult(
            GateId.G2, GateStatus.FAIL, "rebase", "patch.rebase",
            (f"focal patch {context.descriptor.patch_id!r} is not known-good",),
        )
    return GateResult(GateId.G2, GateStatus.PASS, "rebase", "patch.rebase")


def evaluate_package_gate(context: GateContext) -> GateResult:
    """Evaluate the focal package policy without unrelated patch poisoning."""
    try:
        report = validation_policy.check_validation_packages(
            root=context.patches_dir,
            registry_path=context.patches_dir,
            external_sources_path=context.external_sources_path,
            baseline_path=context.validation_baseline_path,
        )
    except (OSError, TypeError, ValueError, validation_policy.PolicyError) as exc:
        return GateResult(GateId.G3, GateStatus.BLOCKED, "package", "patch.validation_policy", (str(exc),))
    status = next((item for item in report.statuses if item.patch_id == context.descriptor.patch_id), None)
    if status is None:
        return GateResult(
            GateId.G3, GateStatus.BLOCKED, "package", "patch.validation_policy",
            (f"no package-policy result for {context.descriptor.patch_id!r}",),
        )
    if status.status == "invalid":
        return GateResult(GateId.G3, GateStatus.FAIL, "package", "patch.validation_policy", status.problems)
    if status.status == "not-required":
        return GateResult(GateId.G3, GateStatus.NA, "package", "patch.validation_policy")
    if context.intent in (GateIntent.VALIDATE, GateIntent.PROMOTE):
        try:
            validation_policy.require_execution_package(context.descriptor, root=context.patches_dir)
        except (OSError, TypeError, ValueError, validation_policy.PolicyError) as exc:
            return GateResult(GateId.G3, GateStatus.FAIL, "package", "patch.validation_policy", (str(exc),))
    performance_problems = validation_policy.check_performance_evidence_for_patch(
        context.descriptor,
        root=context.patches_dir,
        assume_validated=context.intent is GateIntent.PROMOTE,
    )
    if performance_problems:
        return GateResult(
            GateId.G3, GateStatus.FAIL, "package", "patch.validation_policy",
            (*status.problems, *performance_problems),
        )
    return GateResult(GateId.G3, GateStatus.PASS, "package", "patch.validation_policy", status.problems)


def evaluate_evidence_gate(context: GateContext) -> GateResult:
    """Evaluate G4 through the catalog/evidence authority.

    Promotion is prospective: the focal untested module is evaluated as
    validated without mutating registry state.  Build keeps the existing
    ``not-required`` result as NA, while validation and promotion require an
    actual evidence result.
    """
    try:
        statuses = patch_catalog.validation_evidence_statuses(
            (context.descriptor.patch_id,),
            catalog_path=context.catalog_path,
            patches_dir=context.patches_dir,
            pinned_ref=context.pinned_ref,
            evidence_root=context.evidence_root,
            allow_legacy_grandfather=context.allow_legacy_grandfather,
            resolved_base_revision=context.resolved_base_revision,
            assume_validated=(
                frozenset((context.descriptor.patch_id,))
                if context.intent is GateIntent.PROMOTE else frozenset()
            ),
        )
    except (OSError, TypeError, ValueError, patch_catalog.patch_validation_evidence.ValidationEvidenceError) as exc:
        return GateResult(GateId.G4, GateStatus.BLOCKED, "evidence", "patch.catalog", (str(exc),))

    status = statuses.get(context.descriptor.patch_id)
    if status is None:
        return GateResult(
            GateId.G4, GateStatus.BLOCKED, "evidence", "patch.catalog",
            (f"no evidence result for {context.descriptor.patch_id!r}",),
        )
    if status.status == "not-required":
        if context.intent in (GateIntent.VALIDATE, GateIntent.PROMOTE):
            return GateResult(
                GateId.G4, GateStatus.FAIL, "evidence", "patch.catalog",
                ("validation or promotion requires an evidence obligation",),
            )
        return GateResult(GateId.G4, GateStatus.NA, "evidence", "patch.catalog")
    if not status.ok:
        return GateResult(GateId.G4, GateStatus.FAIL, "evidence", "patch.catalog", status.problems)
    return GateResult(GateId.G4, GateStatus.PASS, "evidence", "patch.catalog", status.problems)


def evaluate_admission_gate(context: GateContext) -> GateResult:
    """Evaluate G7 against the complete resolved production composition."""
    ids = tuple(module.patch_id for module in context.composition.modules)
    try:
        # Keep admission ownership in patch_admission; this import is lazy to
        # preserve the patch package's dependency direction.
        from .. import patch_admission

        result = patch_admission.admit(
            ids,
            mode="production",
            catalog_path=context.catalog_path,
            patches_dir=context.patches_dir,
            pinned_ref=context.pinned_ref,
            resolved_base_revision=context.resolved_base_revision,
            evidence_root=context.evidence_root,
            allow_legacy_grandfather=context.allow_legacy_grandfather,
        )
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        return GateResult(GateId.G7, GateStatus.BLOCKED, "admission", "patch_admission", (str(exc),))

    status = getattr(result, "status", None)
    admissible = getattr(result, "admissible", None)
    gate_active = getattr(result, "gate_active", None)
    failures = tuple(getattr(result, "failures", ()) or ())
    warnings = tuple(getattr(result, "warnings", ()) or ())
    if not isinstance(status, str) or not isinstance(admissible, bool) or not isinstance(gate_active, bool):
        return GateResult(
            GateId.G7, GateStatus.BLOCKED, "admission", "patch_admission",
            ("admission returned a malformed result",),
        )
    if status == "not-ready" and not gate_active:
        return GateResult(GateId.G7, GateStatus.BLOCKED, "admission", "patch_admission", warnings)
    if status in {"rejected", "escape-hatch"} or not admissible:
        return GateResult(GateId.G7, GateStatus.FAIL, "admission", "patch_admission", failures or warnings)
    if status == "admitted" and gate_active and not failures:
        return GateResult(GateId.G7, GateStatus.PASS, "admission", "patch_admission", warnings)
    return GateResult(
        GateId.G7, GateStatus.BLOCKED, "admission", "patch_admission",
        (f"unrecognized admission result status {status!r}",),
    )


def evaluate_disposition_gate(context: GateContext) -> GateResult:
    """Evaluate G6 through the revision-bound disposition coverage authority."""
    if context.catalog_states is None or context.coverage_report is None or context.target_revision is None:
        return GateResult(
            GateId.G6, GateStatus.BLOCKED, "disposition", "patch.disposition",
            ("complete disposition coverage inputs were not supplied",),
        )
    try:
        coverage = patch_disposition.compute_coverage(
            catalog_states=dict(context.catalog_states),
            all_report=dict(context.coverage_report),
            recipe_patch_ids=context.recipe_patch_ids,
            dispositions=patch_disposition.list_dispositions(context.dispositions_dir),
            target_revision=context.target_revision,
        )
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        return GateResult(GateId.G6, GateStatus.BLOCKED, "disposition", "patch.disposition", (str(exc),))
    if not isinstance(coverage.complete, bool):
        return GateResult(
            GateId.G6, GateStatus.BLOCKED, "disposition", "patch.disposition",
            ("disposition coverage returned a malformed result",),
        )
    if coverage.complete:
        return GateResult(GateId.G6, GateStatus.PASS, "disposition", "patch.disposition")
    return GateResult(
        GateId.G6, GateStatus.FAIL, "disposition", "patch.disposition",
        tuple(coverage.uncovered_patch_ids),
    )
