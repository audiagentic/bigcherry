"""Contract gates for the patch-validation campaign: correctness gate,
subject parity, lane-effect records and persisted validation eligibility."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

from bigcherry.build.builds import CompletedBuildEvidence
from bigcherry.experiment import contract as experiment_contract
from bigcherry.patch.campaign.build import PatchCampaignError


def compute_contract_correctness_gate(
    contract: experiment_contract.ExperimentContract | None,
    named_results: "dict[str, object] | None" = None,
) -> dict[str, object] | None:
    """VA14 final slice (GPT session ses_5bbee8ce5c9a4265, req_75c09f14757640af):
    delegates to the real, native `experiment_contract.evaluate_correctness_gate()`
    instead of always reporting BLOCKED -- a real per-named-check evidence
    producer now exists for at least one contract (RD08's patch-local
    producer, orchestrated by the generic --validation-producer path).
    Returns None when there is no
    bound contract at all, or the contract declares no required correctness
    checks (a pure-performance contract passes trivially).

    ``named_results`` must be real ``CorrectnessResult``s keyed by check
    name, never a generic --correctness-evidence summary standing in for a
    specific named check (GPT round 8, req_84fca34f83064678 -- that
    confusion is exactly what this signature change prevents: there is no
    longer a `correctness_summary` parameter to misuse this way). Missing
    results (None, or a check simply absent from the dict) are reported via
    evaluate_correctness_gate()'s own missing_checks -- caught here as a
    BLOCKED-shaped dict only when the gate would otherwise raise for
    receiving literally zero results against a contract that requires some
    (its own hard-fail-on-truly-empty-input behavior); any check present in
    ``named_results`` is judged on its own passed/failed value."""
    if contract is None or not contract.correctness.required_checks:
        return None
    from bigcherry.experiment import contract as experiment_contract

    try:
        return experiment_contract.evaluate_correctness_gate(
            contract, dict(named_results or {})
        )
    except (
        experiment_contract.ExperimentContractError
    ):  # pi-lens-ignore: no-bare-except
        required_checks = contract.correctness.required_checks
        return {
            "passed": False,
            "status": "blocked",
            "required_checks": list(required_checks),
            "missing_checks": list(required_checks),
            "failed_checks": [],
            "results": {},
            "detail": (
                f"contract requires correctness check(s) {list(required_checks)!r}; no "
                "per-named-check evidence was supplied for this run"
            ),
        }


def assert_validation_subject_parity(
    control_build_evidence: CompletedBuildEvidence,
    validation_subject_build_evidence: CompletedBuildEvidence,
    *,
    patch_id: str,
) -> None:
    """VA14-B (GPT session ses_5bbee8ce5c9a4265, req_cb50258c7f4c40f1): the
    validation-subject build must be a real build-parity match to control --
    same requested cmake args, same effective configure, same effective
    build id -- so a measured RD08 lane effect can be attributed to the
    patch alone, never to an accidental build-option drift between the two
    binaries. Deliberately does NOT compare runtime_bundle_hash,
    compile_verification_id, or full campaign_identity(): those are
    correctly source-content-sensitive, and control/subject sources
    legitimately differ (that is the whole point of the comparison)."""
    control_configure = control_build_evidence.effective_configure
    subject_configure = validation_subject_build_evidence.effective_configure
    if control_configure != subject_configure:
        raise PatchCampaignError(
            f"{patch_id}: validation-subject build is not configure-parity with control -- "
            f"control={control_configure!r} subject={subject_configure!r}"
        )
    control_id = control_build_evidence.effective_build_id
    subject_id = validation_subject_build_evidence.effective_build_id
    if control_id != subject_id:
        raise PatchCampaignError(
            f"{patch_id}: validation-subject build_id {subject_id!r} does not match "
            f"control build_id {control_id!r} despite matching effective_configure -- "
            "refusing to run parity-dependent RD08 lanes against a non-parity build"
        )


def rd08_validation_lane_commands(
    *,
    control_binary: Path,
    subject_binary: Path,
    model: Path,
    workload: str,
    extra_flags: tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    """VA14-B: the real, minimal llama-bench command pair for one RD08 lane
    -- control_command, subject_command -- differing only by binary path,
    consistent with metric_for_workload()'s decode->tg128/prefill->pp512
    mapping (decode: -p 0 -n 128; prefill: -p 512 -n 0). ``extra_flags``
    (VA06: e.g. ("-sm", "tensor") for a multi-GPU model) is appended
    after the workload shape/-ngl flags -- empty by default, so RD08's
    own existing behavior is unchanged."""
    if workload == "decode":
        workload_flags = ["-p", "0", "-n", "128"]
    elif workload == "prefill":
        workload_flags = ["-p", "512", "-n", "0"]
    else:
        raise PatchCampaignError(
            f"rd08 lane: no llama-bench flag mapping for workload {workload!r}"
        )
    control_command = [
        str(control_binary),
        "-m",
        str(model),
        *workload_flags,
        "-ngl",
        "99",
        *extra_flags,
    ]
    subject_command = [
        str(subject_binary),
        "-m",
        str(model),
        *workload_flags,
        "-ngl",
        "99",
        *extra_flags,
    ]
    return control_command, subject_command


_LANE_EFFECT_FIELDS = (
    "geometric_effect_pct",
    "decision",
    "ci95_low_pct",
    "ci95_high_pct",
    "paired_rounds",
)


def collect_lane_effect_records(
    *,
    rd08_qualification: "dict[str, object] | None",
    rd73_qualification: "dict[str, object] | None",
) -> list[dict[str, object]]:
    """RV99: the per-lane measurements to persist in the validation record.

    The record used to keep identity, provenance and verdicts but not the
    numbers those verdicts came from -- per-lane effects and their
    ``pair_ratios`` existed only under ``artifacts/``, which is gitignored. An
    interval therefore could not be re-derived, re-aggregated across sessions,
    re-analysed under a new estimator, or audited from committed evidence.

    The legacy run() path no longer produces lane effects: RD08's
    historical stats-dict shape was retired with the 1204/RD08 producer
    migration, and RD73's LaneEffect shape was retired with the RD73 legacy
    compatibility retirement -- both patches' lane effects now flow through
    the generic producer path. This collector therefore returns no records;
    it is retained so make_record()'s lane_effects argument stays uniform.
    """
    records: list[dict[str, object]] = []

    def _add(role: str, metric: str, source: object) -> None:
        if source is None:
            return
        raw = (
            dataclasses.asdict(source)
            if dataclasses.is_dataclass(source)
            else dict(source)
        )
        ratios = raw.get("pair_ratios") or ()
        entry: dict[str, object] = {
            "role": raw.get("role") or role,
            "metric": raw.get("metric") or metric,
            "pair_ratios": [float(value) for value in ratios],
        }
        for field in _LANE_EFFECT_FIELDS:
            if raw.get(field) is not None:
                entry[field] = raw[field]
        records.append(entry)

    if rd73_qualification is not None:
        _add("positive", "mtp_wall_tps", rd73_qualification["mtp"].get("effect"))
        _add(
            "control", "decode_tps", rd73_qualification["decode_control"].get("effect")
        )
    if rd08_qualification is not None:
        for role, lane in (rd08_qualification.get("lanes") or {}).items():
            if isinstance(lane, Mapping):
                _add(role, str(lane.get("metric") or ""), lane.get("stats"))
    return records


def compute_persisted_validation_eligible(
    descriptor: object,
    validation_verdict: object | None,
    contract_promotions: "dict[str, dict[str, object]] | None",
    *,
    activation_disposition: str | None,
    correctness: "dict[str, object] | None",
) -> bool | None:
    """VA14 final slice (GPT req_75c09f14757640af): a bound-contract patch
    is eligible_for_validated_state only when BOTH the adapter verdict
    (compute_verdict() -- validation.toml's own checks) AND every one of
    the patch's bound Experiment Contracts have a passing
    evaluate_promotion_gate() result in ``contract_promotions`` (keyed by
    contract id). Uses the plural ``descriptor.experiment_contracts`` --
    never the singular ``.experiment_contract`` compatibility property,
    which raises for a multi-contract patch. A patch with NO bound contract
    is unaffected -- the adapter verdict alone is the only qualification
    such a patch ever claims, exactly as before.

    RV95: this predicate must also require what evidence.py's
    ``_record_qualifies()`` requires of the record's OWN top-level
    activation/correctness fields, because the two are read as answering the
    same question and previously did not. The (now-retired) legacy
    --run-rd73-contract path populated the adapter verdict and the contract
    promotion but left activation/correctness at disposition="unknown", so this returned True while
    verify_validated_patch() rejected the very same record with "activation
    is not executed+activation-verified; correctness did not pass". The
    campaign then printed "STATE='validated' eligible: yes" for a record no
    verifier would accept -- a fail-OPEN disagreement in a system whose
    whole contract is to fail closed.

    Keeping the two predicates in sync structurally (rather than by
    convention) is the point: a producer that cannot populate these fields
    now reports ineligible, which is the safe direction. The literals below
    are deliberately the same ones evidence.py:745-754 tests."""
    if not descriptor.experiment_contracts:
        if validation_verdict is None:
            return None
        return validation_verdict.eligible
    if validation_verdict is None or not validation_verdict.eligible:
        return False
    if activation_disposition != "activation-verified":
        return False
    if (
        not isinstance(correctness, Mapping)
        or correctness.get("disposition") != "passed"
    ):
        return False
    promotions = contract_promotions or {}
    return all(
        promotions.get(contract_id, {}).get("passed")
        is True  # pi-lens-ignore: no-identity-operator-on-literals
        for contract_id in descriptor.experiment_contracts
    )


def build_contract_evidence_for_persistence(
    plan_contracts: "tuple[object, ...]",
    contract_promotions: "dict[str, dict[str, object]] | None",
) -> "tuple[list[dict[str, str]], dict[str, dict[str, object]]]":
    """VA18 persistence plumbing: derive make_record()'s plural
    ``contracts``/``contract_verdicts`` arguments from
    ``validation_plan.contracts`` (never the singular
    ``descriptor.experiment_contract`` compatibility property, which fails
    closed for a real multi-contract patch) and the existing
    ``contract_promotions`` dict (populated by the generic
    --validation-producer path).
    A bound contract with no produced promotion result gets an
    explicit BLOCKED verdict ({"passed": False, "status": "blocked", ...})
    -- never an inferred PASS."""
    promotions = contract_promotions or {}
    contracts = [
        {"id": binding.contract_id, "hash": binding.contract_hash}
        for binding in plan_contracts
    ]
    verdicts = {
        binding.contract_id: (
            {
                "passed": bool(promotion.get("passed")),
                "status": promotion.get("status"),
                "detail": promotion,
            }
            if (promotion := promotions.get(binding.contract_id)) is not None
            else {
                "passed": False,
                "status": "blocked",
                "detail": {"reasons": ["no promotion result produced"]},
            }
        )
        for binding in plan_contracts
    }
    return contracts, verdicts
