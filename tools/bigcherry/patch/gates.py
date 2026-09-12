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
    LINT = "lint"


@dataclass(frozen=True, slots=True)
class GateResult:
    id: GateId
    status: GateStatus
    phase: str
    authority: str
    detail: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScopedGateResult:
    """One shared gate result scoped to a repository patch descriptor."""

    patch_id: str
    result: GateResult


@dataclass(frozen=True, slots=True)
class LintGateReport:
    """Repository-static lint results and the legacy CLI projection inputs."""

    results: tuple[ScopedGateResult, ...]
    problems: tuple[str, ...]
    grandfathered: tuple[str, ...]


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
    recipe_patch_ids: frozenset[str] | None = None
    target_revision: str | None = None


def gate_applies(gate_id: GateId, intent: GateIntent) -> bool:
    """Return the fixed PA21 applicability matrix."""
    applicable = {
        GateIntent.AUTHOR: frozenset((GateId.G0, GateId.G1)),
        GateIntent.VALIDATE: frozenset((GateId.G0, GateId.G1, GateId.G2, GateId.G3)),
        GateIntent.PROMOTE: frozenset((GateId.G0, GateId.G1, GateId.G2, GateId.G3, GateId.G4, GateId.G5)),
        GateIntent.REBASE: frozenset((GateId.G0, GateId.G1, GateId.G2, GateId.G6)),
        GateIntent.BUILD: frozenset((GateId.G0, GateId.G1, GateId.G2, GateId.G4, GateId.G6, GateId.G7)),
        GateIntent.LINT: frozenset((GateId.G1, GateId.G3)),
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
        problems = patch_docs.check_summary_for_patch(context.descriptor, context.patches_dir)
    except (OSError, TypeError, ValueError) as exc:
        return GateResult(GateId.G1, GateStatus.BLOCKED, "documentation", "patch.docs", (str(exc),))
    if problems:
        return GateResult(GateId.G1, GateStatus.FAIL, "documentation", "patch.docs", problems)
    return GateResult(GateId.G1, GateStatus.PASS, "documentation", "patch.docs")


def _evaluate_lint_summary(
    descriptor: patch_registry.PatchDescriptor,
    patches_dir: Path,
) -> GateResult:
    """Evaluate one repository-static SUMMARY gate without fake GateContext."""
    problems = patch_docs.check_summary_for_patch(descriptor, patches_dir)
    if problems:
        return GateResult(GateId.G1, GateStatus.FAIL, "documentation", "patch.docs", problems)
    return GateResult(GateId.G1, GateStatus.PASS, "documentation", "patch.docs")


def _evaluate_lint_package(
    descriptor: patch_registry.PatchDescriptor,
    status: validation_policy.PackagePolicyStatus | None,
    performance_problems: tuple[str, ...],
) -> GateResult:
    """Project existing static policy authorities into the shared G3 result."""
    if status is None:
        package_status = GateStatus.NA
        package_detail: tuple[str, ...] = ()
    elif status.status == "invalid":
        package_status = GateStatus.FAIL
        package_detail = status.problems
    elif status.status in ("current", "grandfathered"):
        package_status = GateStatus.PASS
        package_detail = status.problems
    elif status.status == "not-required":
        package_status = GateStatus.NA
        package_detail = ()
    else:
        return GateResult(
            GateId.G3, GateStatus.BLOCKED, "package", "patch.validation_policy",
            (f"unknown package-policy status {status.status!r} for {descriptor.patch_id!r}",),
        )
    if performance_problems:
        return GateResult(
            GateId.G3, GateStatus.FAIL, "package", "patch.validation_policy",
            (*package_detail, *performance_problems),
        )
    return GateResult(GateId.G3, package_status, "package", "patch.validation_policy", package_detail)


def evaluate_repository_lint_gates(
    *,
    patches_dir: Path = paths.PATCHES,
    external_sources_path: Path = paths.EXTERNAL_SOURCES,
    validation_baseline_path: Path = paths.VALIDATION_PACKAGE_GRANDFATHER,
) -> LintGateReport:
    """Evaluate the repository-static subset of the shared gate taxonomy.

    Lint has no focal composition, upstream checkout, current-pin evidence,
    disposition target, or production admission decision.  Its two gates are
    therefore evaluated directly against the canonical registry and existing
    static policy authorities rather than by fabricating a ``GateContext``.
    """
    registry = patch_registry.load_registry(patches_dir)
    def evaluate_summaries() -> tuple[tuple[ScopedGateResult, ...], tuple[str, ...], tuple[str, ...]]:
        results: list[ScopedGateResult] = []
        problems: list[str] = []
        for descriptor in registry.descriptors:
            result = _evaluate_lint_summary(descriptor, registry.root)
            results.append(ScopedGateResult(descriptor.patch_id, result))
            problems.extend(result.detail)
        return tuple(results), tuple(problems), ()

    def evaluate_packages() -> tuple[tuple[ScopedGateResult, ...], tuple[str, ...], tuple[str, ...]]:
        package_report = validation_policy.check_validation_packages(
            root=patches_dir,
            registry_path=patches_dir,
            external_sources_path=external_sources_path,
            baseline_path=validation_baseline_path,
        )
        package_statuses = {status.patch_id: status for status in package_report.statuses}
        performance_problems: list[str] = []
        package_results: list[ScopedGateResult] = []
        for descriptor in registry.descriptors:
            problems = validation_policy.check_performance_evidence_for_patch(
                descriptor, root=patches_dir, assume_validated=False,
            )
            performance_problems.extend(problems)
            package_results.append(
                ScopedGateResult(
                    descriptor.patch_id,
                    _evaluate_lint_package(
                        descriptor, package_statuses.get(descriptor.patch_id), problems,
                    ),
                )
            )
        return (
            tuple(package_results),
            tuple((*package_report.problems, *performance_problems)),
            tuple(package_report.grandfathered),
        )

    evaluators = {
        GateId.G1: evaluate_summaries,
        GateId.G3: evaluate_packages,
    }
    results: list[ScopedGateResult] = []
    problems: list[str] = []
    grandfathered: tuple[str, ...] = ()
    for gate_id in GateId:
        if not gate_applies(gate_id, GateIntent.LINT):
            continue
        evaluator = evaluators.get(gate_id)
        if evaluator is None:
            result = GateResult(
                gate_id, GateStatus.BLOCKED, "lint", "patch.gates",
                (f"applicable repository lint gate {gate_id.value} is not implemented",),
            )
            results.append(ScopedGateResult("<repository>", result))
            problems.extend(result.detail)
            continue
        scoped, gate_problems, gate_grandfathered = evaluator()
        results.extend(scoped)
        problems.extend(gate_problems)
        if gate_grandfathered:
            grandfathered = gate_grandfathered
        for scoped_result in scoped:
            if scoped_result.result.status is GateStatus.BLOCKED:
                problems.extend(scoped_result.result.detail)

    return LintGateReport(tuple(results), tuple(problems), grandfathered)


def evaluate_rebase_gate(context: GateContext) -> GateResult:
    """Evaluate G2 through the canonical rebase freshness authority."""
    if context.rebase_report is None:
        return GateResult(
            GateId.G2, GateStatus.BLOCKED, "rebase", "patch.rebase",
            ("no rebase report was supplied",),
        )
    if context.source_root is None:
        return GateResult(
            GateId.G2, GateStatus.BLOCKED, "rebase", "patch.rebase",
            ("no upstream source root was supplied",),
        )
    try:
        known_good = rebase.require_fresh_report(context.rebase_report, context.source_root)
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
        framework_configuration = validation_policy.is_framework_configuration_patch(context.descriptor)
    except AttributeError:
        framework_configuration = False
    if context.intent is GateIntent.PROMOTE and framework_configuration:
        return GateResult(
            GateId.G4, GateStatus.BLOCKED, "evidence", "patch.catalog",
            ("framework PROMOTE requires the prospective canonical-composition seam",),
        )
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
    evidence_status = getattr(status, "status", None)
    evidence_ok = getattr(status, "ok", None)
    recognized_ok_statuses = frozenset({
        "not-required", "validated-evidence", "legacy-grandfathered",
        "ported-benched-evidence", "deferred-hardware-evidence",
        "framework-configuration-evidence",
    })
    recognized_statuses = recognized_ok_statuses | {"missing-or-stale"}
    expected_ok = evidence_status in recognized_ok_statuses
    if (not isinstance(evidence_status, str) or evidence_status not in recognized_statuses
            or not isinstance(evidence_ok, bool) or evidence_ok is not expected_ok):
        return GateResult(GateId.G4, GateStatus.BLOCKED, "evidence", "patch.catalog",
                          ("evidence returned a malformed result",))
    problems = getattr(status, "problems", None)
    if not isinstance(problems, (tuple, list)) or not all(isinstance(item, str) for item in problems):
        return GateResult(GateId.G4, GateStatus.BLOCKED, "evidence", "patch.catalog",
                          ("evidence returned malformed problems",))
    if evidence_status == "not-required":
        if context.intent in (GateIntent.VALIDATE, GateIntent.PROMOTE):
            return GateResult(
                GateId.G4, GateStatus.FAIL, "evidence", "patch.catalog",
                ("validation or promotion requires an evidence obligation",),
            )
        return GateResult(GateId.G4, GateStatus.NA, "evidence", "patch.catalog")
    if not evidence_ok:
        return GateResult(GateId.G4, GateStatus.FAIL, "evidence", "patch.catalog", tuple(problems))
    return GateResult(GateId.G4, GateStatus.PASS, "evidence", "patch.catalog", tuple(problems))


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
    raw_failures = getattr(result, "failures", None)
    raw_warnings = getattr(result, "warnings", None)
    if (not isinstance(raw_failures, (tuple, list)) or not isinstance(raw_warnings, (tuple, list))
            or not all(isinstance(item, str) for item in (*raw_failures, *raw_warnings))):
        return GateResult(GateId.G7, GateStatus.BLOCKED, "admission", "patch_admission",
                          ("admission returned malformed failures or warnings",))
    failures = tuple(raw_failures)
    warnings = tuple(raw_warnings)
    if not isinstance(status, str) or not isinstance(admissible, bool) or not isinstance(gate_active, bool):
        return GateResult(
            GateId.G7, GateStatus.BLOCKED, "admission", "patch_admission",
            ("admission returned a malformed result",),
        )
    # An inactive admission gate is not an approval, regardless of any
    # provisional status/admissible fields carried beside it.
    if not gate_active:
        return GateResult(GateId.G7, GateStatus.BLOCKED, "admission", "patch_admission", warnings)
    if status == "admitted" and admissible and not failures:
        return GateResult(GateId.G7, GateStatus.PASS, "admission", "patch_admission", warnings)
    if status == "rejected" and not admissible:
        return GateResult(GateId.G7, GateStatus.FAIL, "admission", "patch_admission", failures or warnings)
    return GateResult(
        GateId.G7, GateStatus.BLOCKED, "admission", "patch_admission",
        (f"inconsistent or unrecognized admission result status {status!r}",),
    )


def evaluate_disposition_gate(context: GateContext) -> GateResult:
    """Evaluate G6 through the revision-bound disposition coverage authority."""
    if (context.catalog_states is None or context.coverage_report is None
            or context.recipe_patch_ids is None or context.target_revision is None):
        return GateResult(
            GateId.G6, GateStatus.BLOCKED, "disposition", "patch.disposition",
            ("complete disposition coverage inputs were not supplied",),
        )
    if context.source_root is None:
        return GateResult(
            GateId.G6, GateStatus.BLOCKED, "disposition", "patch.disposition",
            ("no upstream source root was supplied",),
        )
    selection = context.coverage_report.get("selection")
    if not isinstance(selection, Mapping) or selection.get("all_patches") is not True:
        return GateResult(
            GateId.G6, GateStatus.BLOCKED, "disposition", "patch.disposition",
            ("coverage report is not an all-patches report",),
        )
    if context.coverage_report.get("upstream_revision") != context.target_revision:
        return GateResult(
            GateId.G6, GateStatus.BLOCKED, "disposition", "patch.disposition",
            ("coverage report upstream revision does not match target revision",),
        )
    try:
        rebase.require_fresh_report(context.coverage_report, context.source_root)
        coverage = patch_disposition.compute_coverage(
            catalog_states=dict(context.catalog_states),
            all_report=dict(context.coverage_report),
            recipe_patch_ids=context.recipe_patch_ids,
            dispositions=patch_disposition.list_dispositions(context.dispositions_dir),
            target_revision=context.target_revision,
        )
        complete = getattr(coverage, "complete", None)
        uncovered = getattr(coverage, "uncovered_patch_ids", None)
        if (not isinstance(complete, bool) or not isinstance(uncovered, (tuple, list))
                or not all(isinstance(item, str) for item in uncovered)):
            raise TypeError("disposition coverage returned malformed fields")
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        return GateResult(GateId.G6, GateStatus.BLOCKED, "disposition", "patch.disposition", (str(exc),))
    if complete:
        return GateResult(GateId.G6, GateStatus.PASS, "disposition", "patch.disposition")
    return GateResult(
        GateId.G6, GateStatus.FAIL, "disposition", "patch.disposition",
        tuple(uncovered),
    )


def evaluate_patch_gates(context: GateContext) -> tuple[GateResult, ...]:
    """Evaluate applicable gates in stable G0-G7 order.

    An applicable gate without an implementation is reported BLOCKED.  This
    keeps consumers honest while the staged PA21 implementation is completed;
    no gate may disappear merely because its adapter is not wired yet.
    """
    evaluators = {
        GateId.G0: evaluate_composition_gate,
        GateId.G1: evaluate_summary_gate,
        GateId.G2: evaluate_rebase_gate,
        GateId.G3: evaluate_package_gate,
        GateId.G4: evaluate_evidence_gate,
        GateId.G6: evaluate_disposition_gate,
        GateId.G7: evaluate_admission_gate,
    }
    results: list[GateResult] = []
    results_by_id: dict[GateId, GateResult] = {}
    for gate_id in GateId:
        if not gate_applies(gate_id, context.intent):
            continue
        evaluator = evaluators.get(gate_id)
        if evaluator is None:
            result = (
                evaluate_lifecycle_gate(context, results_by_id)
                if gate_id is GateId.G5
                else GateResult(
                    gate_id, GateStatus.BLOCKED, "lifecycle", "patch.gates",
                    ("applicable gate is not implemented",),
                )
            )
        else:
            result = evaluator(context)
        results.append(result)
        results_by_id[gate_id] = result
    return tuple(results)


def evaluate_lifecycle_gate(
    context: GateContext,
    prior_results: Mapping[GateId, GateResult],
) -> GateResult:
    """Evaluate G5 from current state and the already-computed G0-G4 gates."""
    if context.intent is not GateIntent.PROMOTE:
        return GateResult(GateId.G5, GateStatus.NA, "lifecycle", "patch.catalog")
    module = next(
        (item for item in context.composition.modules
         if item.patch_id == context.descriptor.patch_id),
        None,
    )
    if module is None:
        return GateResult(
            GateId.G5, GateStatus.BLOCKED, "lifecycle", "patch.catalog",
            ("focal patch is absent from the resolved composition",),
        )
    if getattr(module, "state", None) != "untested":
        return GateResult(GateId.G5, GateStatus.NA, "lifecycle", "patch.catalog")
    missing = tuple(
        gate_id.value for gate_id in (GateId.G0, GateId.G1, GateId.G2, GateId.G3, GateId.G4)
        if gate_id not in prior_results
    )
    if missing:
        return GateResult(
            GateId.G5, GateStatus.BLOCKED, "lifecycle", "patch.gates",
            ("missing prerequisite gate results: " + ", ".join(missing),),
        )
    blocked = tuple(
        f"{gate_id.value}={prior_results[gate_id].status.value}"
        for gate_id in (GateId.G0, GateId.G1, GateId.G2, GateId.G3, GateId.G4)
        if prior_results[gate_id].status is GateStatus.BLOCKED
    )
    if blocked:
        return GateResult(GateId.G5, GateStatus.BLOCKED, "lifecycle", "patch.gates", blocked)
    failures = tuple(
        f"{gate_id.value}={prior_results[gate_id].status.value}"
        for gate_id in (GateId.G0, GateId.G1, GateId.G2, GateId.G3, GateId.G4)
        if prior_results[gate_id].status is not GateStatus.PASS
    )
    if failures:
        return GateResult(GateId.G5, GateStatus.FAIL, "lifecycle", "patch.gates", failures)
    return GateResult(GateId.G5, GateStatus.PASS, "lifecycle", "patch.gates")
