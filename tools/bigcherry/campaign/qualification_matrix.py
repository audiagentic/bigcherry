"""VA26 P0 slice: PQM-v1, the standard single-GPU Patch Qualification
Matrix planner.

Design consulted with dev-gpt-agent (req_5b7d0dacef604446, 2026-09-08)
against the real current campaign/patch primitives -- this module composes
those primitives, it does not reimplement them:

- ``campaign.resolution.resolve_lane`` / ``resolve_lane_overlay`` for
  "control" (current release / bigcherry-native) vs "subject"
  (control + candidate patch) compositions and their canonical
  ``patch_set_id`` identity.
- ``patch.patchset.expand_composition`` for the candidate patch's REQUIRES
  closure.
- ``patch.registry.PatchDescriptor.validation_architectures`` for ship
  coverage, and ``ExperimentContract.scope.architectures`` for the
  contract's own scientific/gain scope.
- ``experiment.contract.bootstrap_session_effect`` (unchanged, not called
  from here) remains the inferential machinery a cell's evidence is
  eventually measured with; this module only PLANS which cells exist, on
  which architecture, at which evidence level. No GPU execution, no VA25
  attestation plumbing, and no evidence-schema changes happen here -- this
  is deliberately planner-only, per the design review's explicit PR1 scope
  cut.

WHY GAIN/SHIP ARCHITECTURES ARE DERIVED, NOT DECLARED. VA26's own owner
spec requires zero new per-patch architecture axes: ``gain_arches`` is
``contract.scope.architectures`` (the scientific claim), ``safety_arches``
is ``descriptor.validation_architectures - gain_arches`` (ship coverage
beyond the claim). A misconfigured contract whose gain scope is not a
subset of the patch's own declared ship coverage is a configuration error,
not a matrix the planner silently narrows.

EXPLICITLY DEFERRED PAST THIS PR (design review, not oversight): multi-GPU/
topology, 2v1/4v1 calibration scheduling, a new build-cache layer, any
automatic statistical execution, safety-smoke escalation trigger logic (no
metric/threshold exists to encode yet), a runtime negative-control fallback
for counterfactual_not_composable, a persistent reverse-dependency index,
and multi-contract aggregation (a candidate is qualified against exactly
one caller-supplied ``ExperimentContract``, verified to be one of the
patch's own ``descriptor.experiment_contracts``, not silently unioned
across every contract the patch happens to declare).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import resolution
from ..core import config
from ..experiment.contract import ExperimentContract
from ..patch import patchset
from ..patch import registry as patch_registry

QUALIFICATION_POLICY = "pqm-v1"

ArchitectureTier = Literal["gain", "safety"]
Contrast = Literal["isolated", "release_delta"]
EvidenceLevel = Literal["inferential", "smoke"]


class QualificationMatrixError(ValueError):
    pass


@dataclass(frozen=True)
class QualificationComposition:
    """One resolved lane's identity, as it participates in a qualification
    cell -- carries enough to re-resolve and re-verify later without
    re-deriving it from the plan's other fields."""

    source_name: str
    patch_set_id: str
    module_ids: tuple[str, ...]
    module_hashes: tuple[tuple[str, str], ...]


def _composition_from_lane(lane: resolution.ResolvedLane) -> QualificationComposition:
    return QualificationComposition(
        source_name=lane.source_name,
        patch_set_id=lane.patch_set.patch_set_id,
        module_ids=lane.patch_set.module_ids,
        module_hashes=lane.patch_set.module_hashes,
    )


@dataclass(frozen=True)
class QualificationCell:
    """One patch x one architecture x one contrast: the atomic unit of
    qualification evidence. ``control``/``subject`` are exact resolved
    compositions, not names -- a cell's identity is fully determined by
    ``contract_hash`` + the two composition ``patch_set_id``\\ s + policy +
    architecture + contrast (module docstring), so no separate per-cell
    hash is computed in this first pass."""

    patch_id: str
    contract_id: str
    contract_hash: str
    architecture: str
    architecture_tier: ArchitectureTier
    contrast: Contrast
    evidence_level: EvidenceLevel
    control: QualificationComposition
    subject: QualificationComposition


@dataclass(frozen=True)
class QualificationMatrixPlan:
    schema_version: int
    qualification_policy: str
    patch_id: str
    contract_id: str
    contract_hash: str
    dependency_closure: tuple[str, ...]
    gain_arches: tuple[str, ...]
    safety_arches: tuple[str, ...]
    release_composition_hash: str
    cells: tuple[QualificationCell, ...]


def _cell_evidence(tier: ArchitectureTier) -> EvidenceLevel:
    return "inferential" if tier == "gain" else "smoke"


def build_qualification_matrix_plan(
    patch_id: str,
    contract: ExperimentContract,
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    registry: patch_registry.PatchRegistry,
    *,
    catalog_directory: object = None,
    host_architectures: tuple[str, ...] | None = None,
) -> QualificationMatrixPlan:
    """Generate the deterministic single-GPU PQM-v1 matrix for
    ``patch_id`` against ``contract``.

    Fails closed (``QualificationMatrixError``) on: unknown patch,
    ``contract`` not declared by the patch's own
    ``descriptor.experiment_contracts``, ``contract.scope.architectures``
    not a subset of the patch's ``validation_architectures``, or (when
    ``host_architectures`` is supplied) a declared architecture the current
    host inventory cannot run -- an architecture the matrix cannot actually
    execute must never silently drop out of the plan.
    """
    try:
        descriptor = registry.get(patch_id)
    except patch_registry.PatchRegistryError as exc:
        raise QualificationMatrixError(f"unknown patch {patch_id!r}") from exc
    if contract.id not in descriptor.experiment_contracts:
        raise QualificationMatrixError(
            f"{patch_id}: contract {contract.id!r} is not one of this patch's own "
            f"declared experiment_contracts {list(descriptor.experiment_contracts)!r}"
        )

    gain_arches = tuple(contract.scope.architectures)
    ship_arches = tuple(descriptor.validation_architectures)
    missing_ship_coverage = sorted(set(gain_arches) - set(ship_arches))
    if missing_ship_coverage:
        raise QualificationMatrixError(
            f"{patch_id}: contract {contract.id!r} claims gain architecture(s) "
            f"{missing_ship_coverage} that are not in the patch's own declared "
            f"validation_architectures {list(ship_arches)!r} -- fix one of the two "
            f"declarations, the matrix will not silently narrow its own coverage"
        )
    if host_architectures is not None:
        unavailable = sorted(set(ship_arches) - set(host_architectures))
        if unavailable:
            raise QualificationMatrixError(
                f"{patch_id}: validation_architectures {unavailable} are not present "
                f"in the current single-GPU host inventory {list(host_architectures)!r}"
            )
    safety_arches = tuple(a for a in ship_arches if a not in set(gain_arches))

    closure = patchset.expand_composition((patch_id,), directory=catalog_directory).expanded

    native_lane = resolution.resolve_lane(
        "bigcherry-native", cfg, catalog, catalog_directory=catalog_directory,
    )
    release_lane = resolution.resolve_lane(
        "bigcherry", cfg, catalog, catalog_directory=catalog_directory,
    )
    if patch_id in release_lane.patch_set.module_ids:
        raise QualificationMatrixError(
            f"{patch_id} is already part of the current release composition -- PQM-v1 "
            f"plans a CANDIDATE's isolated/release-delta contrasts, not an already-"
            f"shipped patch's (see counterfactual_not_composable handling for that case)"
        )

    isolated_overlay = tuple(m for m in closure if m not in set(native_lane.patch_set.module_ids))
    release_overlay = tuple(m for m in closure if m not in set(release_lane.patch_set.module_ids))

    isolated_subject_lane = resolution.resolve_lane_overlay(
        "bigcherry-native", cfg, catalog,
        overlay_patch_ids=isolated_overlay, overlay_name=f"pqm:{patch_id}",
        catalog_directory=catalog_directory,
    )
    release_subject_lane = resolution.resolve_lane_overlay(
        "bigcherry", cfg, catalog,
        overlay_patch_ids=release_overlay, overlay_name=f"pqm:{patch_id}",
        catalog_directory=catalog_directory,
    )

    isolated_control = _composition_from_lane(native_lane)
    isolated_subject = _composition_from_lane(isolated_subject_lane)
    release_control = _composition_from_lane(release_lane)
    release_subject = _composition_from_lane(release_subject_lane)

    cells: list[QualificationCell] = []
    for tier, arches in (("gain", gain_arches), ("safety", safety_arches)):
        for architecture in arches:
            for contrast, control, subject in (
                ("isolated", isolated_control, isolated_subject),
                ("release_delta", release_control, release_subject),
            ):
                cells.append(QualificationCell(
                    patch_id=patch_id, contract_id=contract.id,
                    contract_hash=contract.contract_hash, architecture=architecture,
                    architecture_tier=tier, contrast=contrast,
                    evidence_level=_cell_evidence(tier),
                    control=control, subject=subject,
                ))

    return QualificationMatrixPlan(
        schema_version=1, qualification_policy=QUALIFICATION_POLICY, patch_id=patch_id,
        contract_id=contract.id, contract_hash=contract.contract_hash,
        dependency_closure=closure, gain_arches=gain_arches, safety_arches=safety_arches,
        release_composition_hash=release_lane.patch_set.patch_set_id,
        cells=tuple(cells),
    )


def stale_release_delta_cells(
    plan: QualificationMatrixPlan,
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    *,
    catalog_directory: object = None,
) -> tuple[QualificationCell, ...]:
    """Admission-order staleness check (design review): recompute the
    CURRENT release composition's identity and compare against the plan's
    frozen ``release_composition_hash``. Verification-time recomputation
    only -- there is deliberately no subscription/invalidation mechanism,
    and this must never be generalised into "current release must always
    equal the release this evidence was measured against": once a patch is
    admitted, the release itself changes to include it, and that admission
    is exactly what makes its OWN prior evidence current, not stale.

    Only ``release_delta`` cells are affected; ``isolated`` cells never
    reference the release composition and remain valid regardless."""
    current_release = resolution.resolve_lane(
        "bigcherry", cfg, catalog, catalog_directory=catalog_directory,
    )
    if current_release.patch_set.patch_set_id == plan.release_composition_hash:
        return ()
    return tuple(cell for cell in plan.cells if cell.contrast == "release_delta")


@dataclass(frozen=True)
class CounterfactualDisposition:
    composable: bool
    blocking_dependents: tuple[str, ...]


def check_counterfactual_composable(
    patch_id: str,
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    *,
    catalog_directory: object = None,
) -> CounterfactualDisposition:
    """For an ALREADY-SHIPPED patch (the release-minus-P admission
    counterfactual, distinct from ``build_qualification_matrix_plan``'s
    candidate-not-yet-shipped case above): can the current release be
    validly resolved with ``patch_id`` removed?

    A linear scan over the catalog's forward ``requires`` edges (there is
    no reverse-dependency index in this codebase, and the design review
    judged a persistent one not justified at this project's scale/
    resolution-is-not-hot-path). Never recursively drops the dependents
    too -- a non-empty ``blocking_dependents`` is reported as-is; NO
    fallback (e.g. a runtime negative control) is attempted here, per the
    design review's explicit PR1 cut."""
    release_lane = resolution.resolve_lane(
        "bigcherry", cfg, catalog, catalog_directory=catalog_directory,
    )
    if patch_id not in release_lane.patch_set.module_ids:
        raise QualificationMatrixError(
            f"{patch_id} is not part of the current release composition"
        )
    remaining = tuple(m for m in release_lane.patch_set.module_ids if m != patch_id)
    by_id = {module.patch_id: module for module in catalog}
    blocking_dependents = tuple(
        pid for pid in remaining
        if pid in by_id and patch_id in by_id[pid].requires
    )
    if blocking_dependents:
        return CounterfactualDisposition(composable=False, blocking_dependents=blocking_dependents)
    try:
        patchset.resolve_exact(remaining, directory=catalog_directory, required_state=None)
    except ValueError as exc:
        raise QualificationMatrixError(
            f"release-minus-{patch_id} failed exact resolution unexpectedly: {exc}"
        ) from exc
    return CounterfactualDisposition(composable=True, blocking_dependents=())
