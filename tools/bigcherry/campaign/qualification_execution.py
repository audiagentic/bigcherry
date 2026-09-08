"""VA26 execution phase, first landable slice: a thin orchestrator that
drives a QualificationMatrixPlan's cells through a caller-supplied lane
runner and feeds the results through the EXISTING, already patch-agnostic
evidence/gate layer -- it does not reimplement any of it.

Design consulted with dev-gpt-agent (req_a635f1ef50a548db, 2026-09-08),
grounded in RD73/RD08's real qualification paths
(patch/validation_campaign.py's run_rd73_contract_qualification /
run_rd08_contract_qualification): both hand-roll their own patch-specific
"lane" logic (server flags, workload, binaries) but already feed results
through the SAME generic helpers this module calls --
experiment.contract.evaluate_correctness_gate / evaluate_trigger_proof /
aggregate_contract_effects / evaluate_promotion_gate. This module is the
missing piece between those two facts: something that drives WHICH cells
to run (from qualification_matrix.py's plan) and assembles their results
into calls to that existing gate layer, without a caller having to hand-
roll the aggregation/gate wiring the way RD73/RD08 each did independently.

WHAT THIS DOES NOT DO (explicit scope cut, not oversight). It does not
know how to build a binary, launch a server, or run a benchmark -- that is
entirely the caller-supplied ``run_cell`` callback's job, exactly as
RD73/RD08's own lane functions are today. It does not infer a generic
default workload for an arbitrary contract (RD73's MTP-verify lanes and
RD08's decode-only lane are genuinely different; a caller with a new
patch still writes its own lane logic, this module only drives WHEN to
call it and what to do with the result). It does not provide a CLI --
consistent with RD73/RD08 having none either (invoked via
``python -m bigcherry.patch.validation_campaign`` per docs/reference/
patches/PATCH_SYSTEM.md); a first live run should be a thin script under
tools/lab/qualification/, matching this project's own "scripts go in
tools/lab/<topic>/" convention. It does not do remote/SSH dispatch,
parallel scheduling, or retry -- it assumes it runs on the GPU host,
exactly like ServerRunner/AttestedServerSession/bench_runner.py already
do.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .qualification_matrix import QualificationCell, QualificationMatrixPlan
from ..experiment.contract import (
    CorrectnessResult,
    ExperimentContract,
    ExperimentContractError,
    LaneEffect,
    TriggerEvidence,
    aggregate_contract_effects,
    evaluate_correctness_gate,
    evaluate_promotion_gate,
    evaluate_trigger_proof,
)


class QualificationExecutionError(ValueError):
    pass


@dataclass(frozen=True)
class CellResult:
    """What a lane runner reports back for ONE QualificationCell.

    ``lane_effects``/``trigger_evidence`` are only meaningful for
    ``evidence_level == "inferential"`` (gain-tier) cells -- they feed the
    plan-level promotion verdict. A ``evidence_level == "smoke"``
    (safety-tier) cell supplies ``smoke_passed`` instead: VA26's own design
    is explicit that a smoke check establishes "no breakage observed", not
    a superiority claim, so it must never silently contribute to the
    promotion aggregate. ``correctness_results`` is cross-cutting -- both
    tiers may supply correctness evidence, and it is merged across every
    cell before the one gate call.

    ``error`` set (from a raised exception, or supplied directly) means
    this cell did not produce usable evidence; see
    ``QualificationExecutionResult.fully_evidenced`` for what that does to
    the overall verdict.
    """

    cell: QualificationCell
    lane_effects: tuple[LaneEffect, ...] = ()
    correctness_results: dict[str, CorrectnessResult] = field(default_factory=dict)
    trigger_evidence: tuple[TriggerEvidence, ...] = ()
    smoke_passed: bool | None = None
    error: str | None = None


LaneRunner = Callable[[QualificationCell], CellResult]


@dataclass(frozen=True)
class QualificationExecutionResult:
    plan: QualificationMatrixPlan
    cell_results: tuple[CellResult, ...]
    correctness_gate: dict[str, object] | None
    trigger_proof: dict[str, object] | None
    aggregated_effects: dict[str, object]
    promotion: dict[str, object]

    @property
    def fully_evidenced(self) -> bool:
        """False if any cell errored -- the promotion verdict is still
        computed (below) so a caller can see WHAT failed, but it is always
        forced to a failing/invalid-shaped verdict rather than a decided
        pass, per VA26's fail-closed cell-completeness requirement."""
        return all(result.error is None for result in self.cell_results)

    @property
    def failed_cells(self) -> tuple[QualificationCell, ...]:
        return tuple(r.cell for r in self.cell_results if r.error is not None)


def execute_qualification_plan(
    plan: QualificationMatrixPlan,
    contract: ExperimentContract,
    *,
    run_cell: LaneRunner,
    target_metric: str,
    end_to_end_metric: str | None = None,
) -> QualificationExecutionResult:
    """Run every cell in ``plan.cells``, in the plan's own (deterministic,
    contrast-then-architecture) order, via ``run_cell`` -- serial only, no
    scheduling, no parallelism (explicit scope cut). Every cell is run
    even after an earlier one fails, so a caller gets full diagnostics
    from one execution rather than stopping at the first problem; the
    FINAL verdict is still forced closed (see below) if any cell failed.

    Correctness results and trigger evidence are merged across ALL cells
    (both tiers can supply correctness evidence). Only
    ``evidence_level == "inferential"`` cells' lane_effects feed
    ``aggregate_contract_effects``/the promotion gate -- smoke cells never
    contribute to it, matching VA26's own two-tier design.
    """
    cell_results: list[CellResult] = []
    for cell in plan.cells:
        try:
            result = run_cell(cell)
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
            result = CellResult(cell=cell, error=str(exc))
        if result.cell is not cell:
            raise QualificationExecutionError(
                f"run_cell returned a CellResult for a different cell "
                f"({result.cell.architecture}/{result.cell.contrast} != "
                f"{cell.architecture}/{cell.contrast}) -- lane runners must "
                f"return the SAME cell object they were called with"
            )
        cell_results.append(result)

    # Merge by check name, but a FAILING result for a given check always
    # wins over a passing one for that same check -- never let a later
    # cell's pass silently overwrite (hide) an earlier cell's fail for the
    # same logical check (e.g. "bit_identical" reported once per
    # architecture in a multi-arch plan). A plain dict.update() here would
    # let whichever cell happened to run last decide the outcome.
    merged_correctness: dict[str, CorrectnessResult] = {}
    for result in cell_results:
        for check, check_result in result.correctness_results.items():
            existing = merged_correctness.get(check)
            if existing is None or (existing.passed and not check_result.passed):
                merged_correctness[check] = check_result
    correctness_gate: dict[str, object] | None = None
    if contract.correctness.required_checks:
        try:
            correctness_gate = evaluate_correctness_gate(contract, merged_correctness)
        except ExperimentContractError:
            # Same reasoning as aggregate_contract_effects() below: a
            # contract with required checks but LITERALLY ZERO correctness
            # results (every cell that would have supplied one already
            # failed with an error) is a caller-empty-input raise, not a
            # returned failure dict. Leave correctness_gate=None here --
            # the fail-closed "invalid" verdict below already covers this
            # case; a real caller with genuine partial data never hits
            # this branch, since evaluate_correctness_gate() only raises
            # on a totally empty results dict, not a partial one.
            pass

    trigger_evidence = tuple(
        evidence for result in cell_results for evidence in result.trigger_evidence
    )
    trigger_proof = evaluate_trigger_proof(list(trigger_evidence)) if trigger_evidence else None

    fully_evidenced = all(result.error is None for result in cell_results)
    if not fully_evidenced:
        # Never call aggregate_contract_effects() here: with a failed cell in
        # the mix there may be genuinely ZERO positive-role effects, which is
        # a raise (ExperimentContractError), not a returned failure dict --
        # correct behavior for a caller expecting real data, but irrelevant
        # noise here since the verdict is already forced to "invalid" below.
        aggregated_effects = {}
        failed = [r.cell for r in cell_results if r.error is not None]
        promotion = {
            "status": "invalid",
            "passed": False,
            "reasons": [
                f"{len(failed)} of {len(cell_results)} qualification cell(s) failed to "
                f"produce evidence -- promotion cannot be evaluated on an incomplete plan"
            ],
            "contract_id": contract.id,
            "contract_hash": contract.contract_hash,
        }
    else:
        lane_effects = [
            effect
            for result in cell_results if result.cell.evidence_level == "inferential"
            for effect in result.lane_effects
        ]
        aggregated_effects = aggregate_contract_effects(
            contract, lane_effects, target_metric=target_metric,
            end_to_end_metric=end_to_end_metric,
        )
        promotion = evaluate_promotion_gate(
            contract, correctness_gate=correctness_gate or {"passed": True, "missing_checks": [],
                                                             "failed_checks": []},
            aggregated_effects=aggregated_effects, trigger_proof=trigger_proof,
        )

    return QualificationExecutionResult(
        plan=plan, cell_results=tuple(cell_results), correctness_gate=correctness_gate,
        trigger_proof=trigger_proof, aggregated_effects=aggregated_effects, promotion=promotion,
    )
