"""Experiment Contract: schema, validator, and registry (EC01/EC02).

The missing layer between an external/experimental optimization and BigCherry's
existing autotune/campaign/evidence machinery is *experimental intent*: what an
optimization claims to improve, the signatures/workloads that should trigger
it, controls that must not regress, boundary cases that define its safe
envelope, correctness requirements, and promotion thresholds.

This module is deliberately NOT a second candidate schema or a second
benchmark framework -- see docs/reference/BigCherry_Experiment_Contract_Implementation_Guide.md
section 2's non-negotiable architecture. A contract's `hypothesis.family` names
one of the FIVE existing kernel families (mmvq/mmq/mmvf/mmf/blas, imported from
autotune_schema.FAMILIES, not redefined here); workload tags are experiment
metadata, never kernel-family identity; and everything downstream of contract
evaluation (campaign lanes, A/B benchmarking, replay, promotion) reuses the
existing machinery unchanged. A contract's own identity/hash (`contract_hash`)
is likewise kept strictly separate from runtime candidate identity and source
provenance -- see EC04/EC05's identity-separation requirement.

Contracts live in ``config/experiment-contracts.toml`` (``paths.EXPERIMENT_CONTRACTS``),
one ``[contract.<id>]`` table per contract, the same registry shape as
``config/recipes.toml``'s ``[experiment.<name>]``/``[patch-set.<name>]`` sections
and ``config/external-sources.toml``'s per-source tables -- this project's
established pattern for "a registry of named things extended incrementally,
one git diff per addition."
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .autotune_schema import FAMILIES

EXPECTED_EFFECTS: tuple[str, ...] = ("performance", "correctness", "both")

# Guide section 2 + the worked example in section 4. Not closed against
# hypothetical future workloads by paranoia -- closed because an unlisted
# workload tag is far more likely a typo than a genuinely new evaluation
# axis, and a contract silently doing nothing under a misspelled tag is a
# worse failure mode than a loud rejection.
WORKLOAD_TAGS: tuple[str, ...] = (
    "decode", "prefill", "mtp_verify", "moe_prefill", "moe_decode",
    "long_context", "gdn_prefill", "multi_gpu_copy", "small_m",
)

# Guide Appendix A's concrete correctness requirements across the 12xx
# patches: backend_reference (0100-class HI correctness harness), greedy_parity
# (temp-0 determinism, the 1002 MTP case), bit_identical (1204/1205's VDR/dual-
# output gates), ppl_equality (1207's MoE fusion gate).
CORRECTNESS_CHECKS: tuple[str, ...] = (
    "backend_reference", "greedy_parity", "bit_identical", "ppl_equality",
)

ACCEPTANCE_FIELDS: tuple[str, ...] = (
    "target_kernel_gain_pct", "end_to_end_gain_pct", "max_control_regression_pct",
)

# EC16: orthogonal experiment-target classification, separate from
# hypothesis.family. FAMILIES stays exactly the 5 runtime kernel-dispatch
# families (EC02's own "never redefined here" rule) -- most materialized
# optimizations are NOT matmul-family-dispatch work (flash-attention, graph
# fusion, TP topology, orchestration/stream scheduling, SSM/GDN kernels,
# hardware-workaround correctness fixes, cross-batch determinism), and
# forcing them into FAMILIES would misrepresent runtime candidate identity
# elsewhere in the tuning system. `target.family` is populated only when
# `target.kind == "kernel_family"`, and is still validated against FAMILIES
# in that case -- this enum grows the *contract's* classification, never
# the runtime family list. Vocabulary matches the guide's own Appendix A
# patch-table language (orchestration perf / multi-GPU correctness / FA
# precision+performance / graph fusion / hardware correctness / determinism
# partial) rather than inventing divergent terms; "ssm_gdn" is a new
# addition for the GatedDeltaNet cluster (RD50+), not in the guide's
# original 1200-1210 table but following the same pattern.
TARGET_KINDS: tuple[str, ...] = (
    "kernel_family", "attention", "graph_fusion", "tp_topology",
    "orchestration", "hardware_correctness", "determinism", "ssm_gdn",
)


class ExperimentContractError(ValueError):
    pass


# --------------------------------------------------------------------- schema


def _table(raw: object, where: str) -> dict[str, object]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ExperimentContractError(f"{where} must be a table")
    return raw


def _strings(raw: object, where: str, *, choices: tuple[str, ...] | None = None,
             required: bool = False) -> tuple[str, ...]:
    if raw is None:
        if required:
            raise ExperimentContractError(f"{where} is required")
        return ()
    if not isinstance(raw, list) or not all(isinstance(v, str) and v for v in raw):
        raise ExperimentContractError(f"{where} must be a list of non-empty strings")
    if len(set(raw)) != len(raw):
        raise ExperimentContractError(f"{where} contains duplicates")
    if choices is not None:
        unknown = sorted(set(raw) - set(choices))
        if unknown:
            raise ExperimentContractError(
                f"{where} names unknown value(s): {', '.join(unknown)} "
                f"(expected one of {', '.join(sorted(choices))})"
            )
    return tuple(raw)


def _required_string(raw: object, where: str, *, choices: tuple[str, ...] | None = None) -> str:
    if not isinstance(raw, str) or not raw:
        raise ExperimentContractError(f"{where} must be a non-empty string")
    if choices is not None and raw not in choices:
        raise ExperimentContractError(
            f"{where}={raw!r} is not one of {', '.join(sorted(choices))}"
        )
    return raw


def _percent(raw: object, where: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ExperimentContractError(f"{where} must be a number")
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ExperimentContractError(
            f"{where} must be a finite number >= 0 (a negative, NaN, or "
            f"infinite gain/regression-budget threshold is not a meaningful "
            f"requirement -- {value!r} given)"
        )
    return value


@dataclass(frozen=True)
class SourceRef:
    source_id: str
    commits: tuple[str, ...]
    atomic_part: str


@dataclass(frozen=True)
class Hypothesis:
    # EC16: optional now -- a contract with an explicit [target] section
    # (target.kind != "kernel_family") has no matmul-family identity to
    # name here. Still required, and still validated against FAMILIES,
    # for any contract that does NOT supply [target] (the legacy shape
    # EC02's 5 backfilled contracts use unchanged -- see parse_contract).
    family: str | None
    expected_effect: str
    rationale: str


@dataclass(frozen=True)
class Target:
    """EC16: orthogonal experiment-target classification (see TARGET_KINDS).
    Independent of hypothesis.family -- a contract's `target` says WHAT KIND
    of optimization this is; hypothesis.family (when kind=="kernel_family")
    says WHICH of the 5 runtime kernel-dispatch families it targets. Every
    contract has exactly one Target, either read from an explicit [target]
    section or derived from a legacy hypothesis.family for backward
    compatibility (see parse_contract)."""
    kind: str
    family: str | None  # populated only when kind == "kernel_family"


@dataclass(frozen=True)
class Scope:
    backend: str
    architectures: tuple[str, ...]
    weight_types: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationSet:
    """One role's (positive/controls) models and workload tags."""
    models: tuple[str, ...]
    workloads: tuple[str, ...]


@dataclass(frozen=True)
class Boundary:
    # Ordered (dimension_name, values) pairs rather than a dict: dataclass
    # instances stay hashable/order-stable for the identity hash below, and
    # a boundary dimension's value ORDER is itself part of what a sweep
    # report (EC10) walks in.
    dimensions: tuple[tuple[str, tuple[object, ...]], ...]


@dataclass(frozen=True)
class CorrectnessRequirements:
    required_checks: tuple[str, ...]


@dataclass(frozen=True)
class Acceptance:
    target_kernel_gain_pct: float | None
    end_to_end_gain_pct: float | None
    max_control_regression_pct: float | None


@dataclass(frozen=True)
class ExperimentContract:
    id: str
    title: str
    source: SourceRef
    hypothesis: Hypothesis
    target: Target
    prerequisites: tuple[str, ...]
    scope: Scope
    positive: EvaluationSet
    controls: EvaluationSet
    boundary: Boundary
    correctness: CorrectnessRequirements
    acceptance: Acceptance

    @property
    def contract_hash(self) -> str:
        """Immutable content identity (EC01). Two contracts with identical
        semantic content hash identically regardless of TOML key order;
        any semantic edit changes the hash. Deliberately excludes nothing --
        every field the schema defines participates, so a hash match is a
        real guarantee the whole contract is unchanged, not just its id."""
        return _contract_digest(self)


def _identity_payload(contract: ExperimentContract) -> dict[str, object]:
    return {
        "id": contract.id,
        "title": contract.title,
        "source": {
            "source_id": contract.source.source_id,
            "commits": list(contract.source.commits),
            "atomic_part": contract.source.atomic_part,
        },
        "hypothesis": {
            "family": contract.hypothesis.family,
            "expected_effect": contract.hypothesis.expected_effect,
            "rationale": contract.hypothesis.rationale,
        },
        "target": {
            "kind": contract.target.kind,
            "family": contract.target.family,
        },
        "prerequisites": list(contract.prerequisites),
        "scope": {
            "backend": contract.scope.backend,
            "architectures": list(contract.scope.architectures),
            "weight_types": list(contract.scope.weight_types),
        },
        "positive": {"models": list(contract.positive.models),
                     "workloads": list(contract.positive.workloads)},
        "controls": {"models": list(contract.controls.models),
                     "workloads": list(contract.controls.workloads)},
        "boundary": {name: list(values) for name, values in contract.boundary.dimensions},
        "correctness": {"required_checks": list(contract.correctness.required_checks)},
        "acceptance": {
            "target_kernel_gain_pct": contract.acceptance.target_kernel_gain_pct,
            "end_to_end_gain_pct": contract.acceptance.end_to_end_gain_pct,
            "max_control_regression_pct": contract.acceptance.max_control_regression_pct,
        },
    }


def _contract_digest(contract: ExperimentContract) -> str:
    encoded = json.dumps(
        _identity_payload(contract), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.blake2b(
        b"bigcherry/experiment-contract/v1\0" + encoded, digest_size=16
    ).hexdigest()


def parse_contract(document: object, *, contract_id: str) -> ExperimentContract:
    """Parse one contract from an already-TOML-decoded table. Format-agnostic
    on purpose: takes a plain dict so both ``load_contracts()`` (TOML) and a
    future EC02 backfill-skeleton generator (Appendix C) can produce the same
    input shape without this function knowing where it came from."""
    where = f"contract.{contract_id}"
    data = _table(document, where)

    title = _required_string(data.get("title"), f"{where}.title")

    source_data = _table(data.get("source"), f"{where}.source")
    source = SourceRef(
        source_id=_required_string(source_data.get("source_id"), f"{where}.source.source_id"),
        commits=_strings(source_data.get("commits"), f"{where}.source.commits", required=True),
        atomic_part=_required_string(source_data.get("atomic_part"), f"{where}.source.atomic_part"),
    )

    hyp_data = _table(data.get("hypothesis"), f"{where}.hypothesis")
    target_raw = data.get("target")

    # EC16: [target] is the new, orthogonal classification. A contract
    # written before EC16 (or one that's still simple matmul-family work)
    # may omit [target] entirely -- in that case hypothesis.family is
    # REQUIRED, validated against FAMILIES exactly as before EC16, and
    # target is derived from it (kind="kernel_family"). A contract that
    # DOES supply [target] is explicit about its own classification;
    # hypothesis.family becomes optional there (required+validated only
    # when target.kind == "kernel_family", and in that case must equal
    # target.family -- one family, not two possibly-disagreeing sources
    # of truth).
    if target_raw is None:
        family = _required_string(
            hyp_data.get("family"), f"{where}.hypothesis.family", choices=FAMILIES)
        target = Target(kind="kernel_family", family=family)
    else:
        target_data = _table(target_raw, f"{where}.target")
        kind = _required_string(target_data.get("kind"), f"{where}.target.kind", choices=TARGET_KINDS)
        if kind == "kernel_family":
            target_family = _required_string(
                target_data.get("family"), f"{where}.target.family", choices=FAMILIES)
        else:
            if target_data.get("family") is not None:
                raise ExperimentContractError(
                    f"{where}.target.family must be absent when target.kind "
                    f"!= \"kernel_family\" (got kind={kind!r}) -- family is "
                    f"only meaningful for matmul-dispatch-family targets"
                )
            target_family = None
        target = Target(kind=kind, family=target_family)

        raw_hyp_family = hyp_data.get("family")
        if kind == "kernel_family":
            if raw_hyp_family is not None and raw_hyp_family != target_family:
                raise ExperimentContractError(
                    f"{where}.hypothesis.family={raw_hyp_family!r} disagrees "
                    f"with {where}.target.family={target_family!r} -- a "
                    f"kernel_family contract must not name two different "
                    f"families"
                )
            family = target_family
        else:
            if raw_hyp_family is not None:
                raise ExperimentContractError(
                    f"{where}.hypothesis.family must be absent when "
                    f"target.kind != \"kernel_family\" (got kind={kind!r})"
                )
            family = None

    hypothesis = Hypothesis(
        family=family,
        expected_effect=_required_string(
            hyp_data.get("expected_effect"), f"{where}.hypothesis.expected_effect",
            choices=EXPECTED_EFFECTS,
        ),
        rationale=_required_string(hyp_data.get("rationale"), f"{where}.hypothesis.rationale"),
    )

    prerequisites = _strings(data.get("prerequisites"), f"{where}.prerequisites")

    scope_data = _table(data.get("scope"), f"{where}.scope")
    scope = Scope(
        backend=_required_string(scope_data.get("backend"), f"{where}.scope.backend"),
        architectures=_strings(scope_data.get("architectures"), f"{where}.scope.architectures",
                                required=True),
        weight_types=_strings(scope_data.get("weight_types"), f"{where}.scope.weight_types"),
    )

    def _evaluation_set(key: str) -> EvaluationSet:
        section = _table(data.get(key), f"{where}.{key}")
        return EvaluationSet(
            models=_strings(section.get("models"), f"{where}.{key}.models", required=True),
            workloads=_strings(section.get("workloads"), f"{where}.{key}.workloads",
                                choices=WORKLOAD_TAGS, required=True),
        )

    positive = _evaluation_set("positive")
    controls = _evaluation_set("controls")

    boundary_data = _table(data.get("boundary"), f"{where}.boundary")
    dims_data = _table(boundary_data.get("dimensions"), f"{where}.boundary.dimensions")
    dimensions: list[tuple[str, tuple[object, ...]]] = []
    for dim_name, values in dims_data.items():
        if not isinstance(values, list) or not values:
            raise ExperimentContractError(
                f"{where}.boundary.dimensions.{dim_name} must be a non-empty list"
            )
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ExperimentContractError(
                    f"{where}.boundary.dimensions.{dim_name} values must be "
                    f"numbers or strings"
                )
        if len(set(values)) != len(values):
            raise ExperimentContractError(
                f"{where}.boundary.dimensions.{dim_name} contains duplicate values"
            )
        dimensions.append((dim_name, tuple(values)))
    boundary = Boundary(dimensions=tuple(sorted(dimensions, key=lambda item: item[0])))

    correctness_data = _table(data.get("correctness"), f"{where}.correctness")
    required_checks = tuple(sorted(
        name for name, requirement in correctness_data.items()
        if requirement == "required"
    ))
    unknown_checks = sorted(set(correctness_data) - set(CORRECTNESS_CHECKS))
    if unknown_checks:
        raise ExperimentContractError(
            f"{where}.correctness names unknown check(s): {', '.join(unknown_checks)} "
            f"(expected one of {', '.join(sorted(CORRECTNESS_CHECKS))})"
        )
    for name, requirement in correctness_data.items():
        if requirement not in ("required", "optional"):
            raise ExperimentContractError(
                f"{where}.correctness.{name} must be \"required\" or \"optional\", "
                f"not {requirement!r}"
            )

    acceptance_data = _table(data.get("acceptance"), f"{where}.acceptance")
    unknown_acceptance = sorted(set(acceptance_data) - set(ACCEPTANCE_FIELDS))
    if unknown_acceptance:
        raise ExperimentContractError(
            f"{where}.acceptance names unknown field(s): {', '.join(unknown_acceptance)}"
        )
    acceptance = Acceptance(
        target_kernel_gain_pct=_percent(
            acceptance_data.get("target_kernel_gain_pct"), f"{where}.acceptance.target_kernel_gain_pct"),
        end_to_end_gain_pct=_percent(
            acceptance_data.get("end_to_end_gain_pct"), f"{where}.acceptance.end_to_end_gain_pct"),
        max_control_regression_pct=_percent(
            acceptance_data.get("max_control_regression_pct"),
            f"{where}.acceptance.max_control_regression_pct"),
    )
    if acceptance.max_control_regression_pct is None:
        raise ExperimentContractError(
            f"{where}.acceptance.max_control_regression_pct is required -- every "
            f"contract must declare the regression budget its controls are held to, "
            f"even a correctness-only contract with no performance claim (see "
            f"docs/reference/BigCherry_Experiment_Contract_Implementation_Guide.md "
            f"Appendix A: RD20/RD22/RD26-class contracts still need a regression budget)"
        )

    unknown_top = sorted(set(data) - {
        "title", "source", "hypothesis", "target", "prerequisites", "scope",
        "positive", "controls", "boundary", "correctness", "acceptance",
    })
    if unknown_top:
        raise ExperimentContractError(f"{where}: unknown field(s): {', '.join(unknown_top)}")

    return ExperimentContract(
        id=contract_id, title=title, source=source, hypothesis=hypothesis, target=target,
        prerequisites=prerequisites, scope=scope, positive=positive, controls=controls,
        boundary=boundary, correctness=CorrectnessRequirements(required_checks),
        acceptance=acceptance,
    )


# ------------------------------------------------------------------- registry


@dataclass(frozen=True)
class ContractRegistry:
    contracts: dict[str, ExperimentContract]

    def __getitem__(self, contract_id: str) -> ExperimentContract:
        return self.contracts[contract_id]

    def __iter__(self):
        return iter(self.contracts.values())

    def __len__(self) -> int:
        return len(self.contracts)

    def prerequisite_order(self) -> tuple[str, ...]:
        """Topologically sorted contract IDs, prerequisites first. Raises on
        a cycle -- a contract can never legitimately depend on itself,
        directly or transitively."""
        return _topological_order(self.contracts)


def _topological_order(contracts: dict[str, ExperimentContract]) -> tuple[str, ...]:
    state: dict[str, str] = {}  # "visiting" | "done"
    order: list[str] = []

    def visit(contract_id: str, path: tuple[str, ...]) -> None:
        current = state.get(contract_id)
        if current == "done":
            return
        if current == "visiting":
            cycle = " -> ".join((*path, contract_id))
            raise ExperimentContractError(f"prerequisite cycle: {cycle}")
        state[contract_id] = "visiting"
        for prerequisite in contracts[contract_id].prerequisites:
            visit(prerequisite, (*path, contract_id))
        state[contract_id] = "done"
        order.append(contract_id)

    for contract_id in sorted(contracts):
        visit(contract_id, ())
    return tuple(order)


def known_source_ids_from_external_sources(
    path: str | Path | None = None,
) -> frozenset[str]:
    """Real cross-check input for ``load_contracts(known_source_ids=...)``:
    every ``[[sources]] id`` currently registered in
    ``config/external-sources.toml`` (``paths.EXTERNAL_SOURCES`` by default).
    Kept as a separate, explicit call rather than an automatic default inside
    ``load_contracts`` -- a caller who wants the real registry cross-check
    (the CLI, a promotion gate) asks for it explicitly; a caller building or
    testing a contract in isolation (this module's own unit tests) is not
    forced to also maintain a real external-sources.toml fixture."""
    import tomllib as _tomllib

    resolved = Path(path) if path is not None else None
    if resolved is None:
        from . import paths
        resolved = paths.EXTERNAL_SOURCES
    try:
        raw = _tomllib.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ExperimentContractError(f"no external-sources registry at {resolved}") from None
    except _tomllib.TOMLDecodeError as exc:
        raise ExperimentContractError(f"{resolved}: {exc}") from None
    return frozenset(
        entry["id"] for entry in raw.get("sources", []) if isinstance(entry, dict) and entry.get("id")
    )


def load_contracts(path: str | Path, *,
                    known_source_ids: frozenset[str] | None = None) -> ContractRegistry:
    """Load every ``[contract.<id>]`` table in ``path`` (default
    ``experiment-contracts.toml`` at the repo root -- see
    ``paths.EXPERIMENT_CONTRACTS``) deterministically. Rejects duplicate IDs
    (impossible via TOML table syntax itself, kept as a defensive check for
    any future non-TOML loader), unknown source IDs (when
    ``known_source_ids`` is supplied -- normally cross-checked against
    ``external-sources.toml`` by the caller), unknown family/workload tags
    (enforced inside ``parse_contract``), and prerequisite cycles."""
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ExperimentContractError(f"no contract file at {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ExperimentContractError(f"{path}: {exc}") from None

    unknown_top_level = sorted(set(raw) - {"contract"})
    if unknown_top_level:
        raise ExperimentContractError(
            f"{path}: unknown top-level field(s): {', '.join(unknown_top_level)}"
        )

    contracts: dict[str, ExperimentContract] = {}
    for contract_id, body in _table(raw.get("contract"), "contract").items():
        if contract_id in contracts:
            raise ExperimentContractError(f"duplicate contract id: {contract_id}")
        contracts[contract_id] = parse_contract(body, contract_id=contract_id)

    if known_source_ids is not None:
        for contract in contracts.values():
            if contract.source.source_id not in known_source_ids:
                raise ExperimentContractError(
                    f"contract.{contract.id}.source.source_id="
                    f"{contract.source.source_id!r} is not a known "
                    f"external-sources.toml entry"
                )

    unknown_prerequisites: dict[str, tuple[str, ...]] = {}
    for contract in contracts.values():
        missing = tuple(sorted(set(contract.prerequisites) - set(contracts)))
        if missing:
            unknown_prerequisites[contract.id] = missing
    if unknown_prerequisites:
        detail = "; ".join(
            f"{contract_id} -> {', '.join(missing)}"
            for contract_id, missing in sorted(unknown_prerequisites.items())
        )
        raise ExperimentContractError(f"unknown prerequisite(s): {detail}")

    registry = ContractRegistry(contracts)
    registry.prerequisite_order()  # raises on a cycle; result unused here
    return registry


# ------------------------------------------------------------- evidence (EC05)

ROLES: tuple[str, ...] = ("positive", "control", "boundary", "holdout")


@dataclass(frozen=True)
class ContractEvidenceRef:
    """A contract-provenance sidecar (EC05) -- deliberately NOT a field of
    provenance.ProvenanceV2. ProvenanceV2 is a closed, fixed five-namespace
    schema (project/source/build/workload/campaign) consumed throughout the
    RE10/RE12/RE13 lifecycle/promotion machinery; adding a sixth required
    namespace there would be a breaking schema change for every existing
    caller, and even an optional field would blur the guide's own
    non-negotiable "keep runtime candidate identity separate from source
    provenance and from experiment identity" -- a report or bundle that
    embeds contract role/workload/model INSIDE the identity document makes
    it too easy for something downstream to accidentally key off them.

    Instead this rides ALONGSIDE a ProvenanceV2 document as a sibling key
    (see ``attach_to_document``) -- present only where a lane/report/bundle
    is contract-driven, absent (not null-filled) otherwise, and never read
    by anything that computes signature/dispatch/build identity.
    """
    contract_id: str
    contract_hash: str
    optimization_id: str
    role: str
    workload_tag: str | None = None
    model_ref: str | None = None
    boundary_dimension: str | None = None
    boundary_value: str | None = None

    def document(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "optimization_id": self.optimization_id,
            "role": self.role,
            "workload_tag": self.workload_tag,
            "model_ref": self.model_ref,
            "boundary_dimension": self.boundary_dimension,
            "boundary_value": self.boundary_value,
        }

    @classmethod
    def from_document(cls, document: object) -> ContractEvidenceRef:
        data = _table(document, "contract_evidence")
        role = _required_string(data.get("role"), "contract_evidence.role", choices=ROLES)
        for optional_field in ("workload_tag", "model_ref", "boundary_dimension", "boundary_value"):
            value = data.get(optional_field)
            if value is not None and not isinstance(value, str):
                raise ExperimentContractError(
                    f"contract_evidence.{optional_field} must be a string or absent"
                )
        return cls(
            contract_id=_required_string(data.get("contract_id"), "contract_evidence.contract_id"),
            contract_hash=_required_string(data.get("contract_hash"), "contract_evidence.contract_hash"),
            optimization_id=_required_string(
                data.get("optimization_id"), "contract_evidence.optimization_id"),
            role=role,
            workload_tag=data.get("workload_tag"),
            model_ref=data.get("model_ref"),
            boundary_dimension=data.get("boundary_dimension"),
            boundary_value=data.get("boundary_value"),
        )


def evidence_ref_for_lane(contract: ExperimentContract, *, role: str,
                          workload_tag: str | None = None, model_ref: str | None = None,
                          boundary_dimension: str | None = None,
                          boundary_value: str | None = None) -> ContractEvidenceRef:
    """Build the evidence sidecar for one contract-expanded lane (matches
    campaign_planner.expand_contract()'s own per-lane fields exactly --
    call this with the same arguments a CampaignLaneExecutionSpec's
    contract_id/role/workload_tag/model_ref/boundary_* were set to)."""
    if role not in ROLES:
        raise ExperimentContractError(f"role={role!r} is not one of {', '.join(ROLES)}")
    return ContractEvidenceRef(
        contract_id=contract.id, contract_hash=contract.contract_hash,
        optimization_id=contract.source.atomic_part, role=role,
        workload_tag=workload_tag, model_ref=model_ref,
        boundary_dimension=boundary_dimension, boundary_value=boundary_value,
    )


def attach_to_document(document: dict[str, object], evidence: ContractEvidenceRef) -> dict[str, object]:
    """Return a NEW dict with ``document`` (typically a ProvenanceV2.document()
    or an experiment_bundle report body) plus a sibling ``contract_evidence``
    key -- never merged into or overwriting any of the caller's own keys.
    Raises if the caller's document already uses that key for something
    else, rather than silently clobbering it."""
    if not isinstance(document, dict):
        raise ExperimentContractError("document must be an object")
    if "contract_evidence" in document:
        raise ExperimentContractError(
            "document already has a 'contract_evidence' key -- refusing to "
            "overwrite it (attach_to_document is meant to be called once, "
            "on a document that has no contract binding yet)"
        )
    return {**document, "contract_evidence": evidence.document()}


def read_from_document(document: object) -> ContractEvidenceRef | None:
    """The inverse of attach_to_document(): None when the document was never
    contract-bound (the normal, non-contract case), a validated
    ContractEvidenceRef when it was."""
    if not isinstance(document, dict) or "contract_evidence" not in document:
        return None
    return ContractEvidenceRef.from_document(document["contract_evidence"])


# ----------------------------------------------------------- aggregation (EC06)


@dataclass(frozen=True)
class LaneEffect:
    """One expanded lane's measured effect for one metric -- the shape
    comparisons.run_comparison()'s published report's ``effects[metric]``
    already has (geometric_effect_pct/ci95_low_pct/ci95_high_pct/decision,
    the exact fields RE15 stage H's balanced comparison produces). Not
    redefined here as a new statistics format -- this is purely a labeled
    wrapper so aggregate_contract_effects() knows which contract ROLE
    (positive/control/boundary) produced a given effect."""
    role: str
    metric: str
    geometric_effect_pct: float
    decision: str | None = None


def aggregate_contract_effects(
    contract: ExperimentContract, lane_effects: list[LaneEffect], *,
    target_metric: str, end_to_end_metric: str | None = None,
) -> dict[str, object]:
    """EC06: roll up per-lane balanced-comparison effects (from
    comparisons.run_comparison()/ab_benchmark.paired_summary(), one call per
    contract-expanded lane pair -- see campaign_planner.expand_contract())
    into the three numbers ExperimentContract.acceptance declares.

    - target_kernel_gain_pct: mean of ``target_metric``'s effect across
      POSITIVE-role lanes only. Never reads control/boundary effects --
      a contract's own performance claim is about what it targets, not
      diluted by what it deliberately isn't supposed to touch.
    - max_control_regression_pct: the WORST (most negative-for-target-
      direction, i.e. numerically most negative) ``target_metric`` effect
      across CONTROL-role lanes, never an average -- a contract that wins
      on average while quietly regressing one control model must not pass.
      A contract with zero control-role effects raises rather than
      silently reporting 0.0 (undefined is not the same as "no
      regression").
    - end_to_end_gain_pct: mean of ``end_to_end_metric``'s effect (falls
      back to ``target_metric`` if no separate end-to-end metric is named)
      across positive-role lanes -- the whole-workload effect, which can
      be much smaller than the kernel-level gain when the affected op is a
      small fraction of total time.

    Returns a dict shaped exactly like ExperimentContract.acceptance's own
    fields, suitable for a direct threshold comparison against
    ``contract.acceptance`` by EC09's promotion gate.
    """
    positive_target = [
        effect.geometric_effect_pct for effect in lane_effects
        if effect.role == "positive" and effect.metric == target_metric
    ]
    if not positive_target:
        raise ExperimentContractError(
            f"contract {contract.id!r}: no positive-role effects for metric "
            f"{target_metric!r} -- cannot compute target_kernel_gain_pct"
        )

    control_effects = [
        effect.geometric_effect_pct for effect in lane_effects
        if effect.role == "control" and effect.metric == target_metric
    ]
    if not control_effects:
        raise ExperimentContractError(
            f"contract {contract.id!r}: no control-role effects for metric "
            f"{target_metric!r} -- every contract must measure its declared "
            f"controls, an empty set is not evidence of no regression"
        )
    # A negative effect (replay/candidate slower than native/reference) is a
    # regression; the WORST control result is the most negative one.
    worst_control = min(control_effects)
    max_control_regression_pct = max(0.0, -worst_control)

    e2e_metric = end_to_end_metric or target_metric
    e2e_effects = [
        effect.geometric_effect_pct for effect in lane_effects
        if effect.role == "positive" and effect.metric == e2e_metric
    ]

    return {
        "target_kernel_gain_pct": statistics.mean(positive_target),
        "end_to_end_gain_pct": statistics.mean(e2e_effects) if e2e_effects else None,
        "max_control_regression_pct": max_control_regression_pct,
    }


# ------------------------------------------------------------- correctness (EC07)


@dataclass(frozen=True)
class CorrectnessResult:
    """One required check's outcome, however it was actually produced --
    parity.check_parity() for build-identity parity, a test-backend-ops
    native/tune-mode run (see validate_rd_patches.py's own pattern this
    session) for backend_reference/bit_identical, an external PPL harness
    for ppl_equality, a temp-0 MTP repro for greedy_parity. This module
    does not run any of them -- EC07's job is the GATE (every check the
    contract declares required must have a passing result attached), not
    a new correctness-checking engine (guide section 2: do not build a
    second benchmark framework)."""
    check: str
    passed: bool
    detail: str = ""


def evaluate_correctness_gate(
    contract: ExperimentContract, results: dict[str, CorrectnessResult],
) -> dict[str, object]:
    """EC07: every check in contract.correctness.required_checks must have
    a CorrectnessResult in ``results`` with passed=True. Missing or failed
    checks are named explicitly, never silently treated as "not required".
    A contract declaring no required checks (a pure-performance contract)
    passes trivially -- guide Appendix B: not every optimization needs a
    correctness claim, but the ones it does declare are non-negotiable."""
    if contract.correctness.required_checks and not results:
        raise ExperimentContractError(
            f"contract {contract.id!r} requires correctness check(s) "
            f"{', '.join(contract.correctness.required_checks)} but no "
            f"results were supplied at all"
        )
    missing: list[str] = []
    failed: list[str] = []
    for check in contract.correctness.required_checks:
        result = results.get(check)
        if result is None:
            missing.append(check)
        elif not result.passed:
            failed.append(check)
    passed = not missing and not failed
    return {
        "passed": passed,
        "required_checks": list(contract.correctness.required_checks),
        "missing_checks": missing,
        "failed_checks": failed,
        "results": {name: {"passed": r.passed, "detail": r.detail} for name, r in results.items()},
    }


# ------------------------------------------------------------ generalisation (EC08)


def generalisation_floor() -> dict[str, float]:
    """The threshold FLOOR every contract-driven generalisation handoff must
    meet or exceed -- generalise.REQUIRED_THRESHOLDS, verbatim, imported
    lazily (generalise.py has its own argparse-driven CLI; importing it at
    module scope here would pull that in for every experiment_contract.py
    caller, most of which never touch generalisation at all). A contract
    may only tighten these values, never loosen them -- see
    require_generalisation_policy()."""
    from . import generalise
    return dict(generalise.REQUIRED_THRESHOLDS)


def require_generalisation_policy(
    policy_thresholds: dict[str, float] | None = None,
) -> dict[str, float]:
    """EC08: the actual handoff contract -- validates (and returns) the
    threshold dict a caller is about to pass into generalise.prove()'s
    policy, guaranteeing every value is at least as strict as
    generalisation_floor(). ``None`` returns the floor unchanged (the
    common case: most contracts don't need stricter-than-default holdout
    proof). Never lowers a threshold, even if a caller's own dict asks for
    a looser one -- guide section 13: "do not weaken these globally to
    make a patch pass."""
    floor = generalisation_floor()
    if policy_thresholds is None:
        return floor
    violations = [
        name for name, floor_value in floor.items()
        if name in policy_thresholds and (
            # min_* thresholds must not go DOWN; max_* thresholds must not
            # go UP -- both directions of "loosening" for this threshold set.
            (name.startswith("min_") and policy_thresholds[name] < floor_value)
            or (name.startswith("max_") and policy_thresholds[name] > floor_value)
        )
    ]
    if violations:
        raise ExperimentContractError(
            f"generalisation policy threshold(s) looser than the required "
            f"floor: {', '.join(sorted(violations))} -- a contract may "
            f"tighten generalise.py's default thresholds, never loosen them"
        )
    merged = dict(floor)
    merged.update(policy_thresholds)
    return merged


# --------------------------------------------------------------- promotion (EC09)


def evaluate_promotion_gate(
    contract: ExperimentContract, *, correctness_gate: dict[str, object],
    aggregated_effects: dict[str, object], generalisation_result: dict[str, object] | None = None,
) -> dict[str, object]:
    """EC09: the final pass/fail, combining EC07's correctness gate, EC06's
    aggregated effects against contract.acceptance's own thresholds, and
    (when supplied) EC08's generalisation proof. This is ADDITIONAL to
    every existing BigCherry promotion gate (promotion.py/tune_promotion.py's
    production-class provenance, decision-grade report, replay coverage,
    etc.) -- a contract-backed promotion must still clear those, this
    function does not replace or re-check them.

    Handles the asymmetric correctness-first case explicitly (guide
    Appendix A: RD20/RD22/RD26-class contracts have no expected throughput
    gain): a contract whose acceptance.target_kernel_gain_pct and
    end_to_end_gain_pct are both None can still promote on correctness +
    the regression budget alone -- performance checks are skipped when the
    contract never declared a performance claim, not treated as failures.
    max_control_regression_pct is always checked: EC01's parser requires
    it on every contract, so there is always a declared regression budget
    to hold aggregated_effects to, regardless of whether a gain was
    claimed."""
    reasons: list[str] = []

    if not correctness_gate.get("passed"):
        missing = correctness_gate.get("missing_checks") or []
        failed = correctness_gate.get("failed_checks") or []
        reasons.append(
            f"correctness gate failed (missing={list(missing)}, failed={list(failed)})"
        )

    acceptance = contract.acceptance
    if acceptance.target_kernel_gain_pct is not None:
        measured = aggregated_effects.get("target_kernel_gain_pct")
        if not isinstance(measured, (int, float)) or measured < acceptance.target_kernel_gain_pct:
            reasons.append(
                f"target_kernel_gain_pct {measured} below required "
                f"{acceptance.target_kernel_gain_pct}"
            )
    if acceptance.end_to_end_gain_pct is not None:
        measured = aggregated_effects.get("end_to_end_gain_pct")
        if not isinstance(measured, (int, float)) or measured < acceptance.end_to_end_gain_pct:
            reasons.append(
                f"end_to_end_gain_pct {measured} below required "
                f"{acceptance.end_to_end_gain_pct}"
            )

    measured_regression = aggregated_effects.get("max_control_regression_pct")
    if (not isinstance(measured_regression, (int, float))
            or measured_regression > acceptance.max_control_regression_pct):
        reasons.append(
            f"max_control_regression_pct {measured_regression} exceeds budget "
            f"{acceptance.max_control_regression_pct}"
        )

    if generalisation_result is not None and not generalisation_result.get("passed"):
        reasons.append("generalisation proof did not pass")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "contract_id": contract.id,
        "contract_hash": contract.contract_hash,
    }


# ---------------------------------------------------------------- reporting (EC10)


def render_report(
    contract: ExperimentContract, *, correctness_gate: dict[str, object],
    aggregated_effects: dict[str, object], promotion_gate: dict[str, object],
    generalisation_result: dict[str, object] | None = None,
) -> str:
    """EC10: one human-readable report per contract, covering both wins AND
    rejections as first-class results (guide section 12 step 12: "Report
    failures and losing envelopes as first-class results; rejected
    optimizations are useful evidence.") -- this function never
    short-circuits on a failing promotion_gate; every section renders
    regardless of outcome."""
    lines: list[str] = []
    lines.append(f"# Experiment Contract report: {contract.id}")
    lines.append("")
    lines.append(f"**{contract.title}**")
    lines.append("")

    lines.append("## Target")
    lines.append(f"- kind: {contract.target.kind}")
    if contract.target.family is not None:
        lines.append(f"- family: {contract.target.family}")
    lines.append("")

    lines.append("## Hypothesis")
    lines.append(f"- expected_effect: {contract.hypothesis.expected_effect}")
    lines.append(f"- rationale: {contract.hypothesis.rationale}")
    lines.append("")

    lines.append("## Source")
    lines.append(f"- source_id: {contract.source.source_id}")
    lines.append(f"- commits: {', '.join(contract.source.commits)}")
    lines.append(f"- atomic_part: {contract.source.atomic_part}")
    lines.append(f"- contract_hash: {contract.contract_hash}")
    lines.append("")

    lines.append("## Scope")
    lines.append(f"- backend: {contract.scope.backend}")
    lines.append(f"- architectures: {', '.join(contract.scope.architectures)}")
    if contract.scope.weight_types:
        lines.append(f"- weight_types: {', '.join(contract.scope.weight_types)}")
    lines.append("")

    lines.append("## Winners (positive lanes)")
    lines.append(f"- models: {', '.join(contract.positive.models)}")
    lines.append(f"- workloads: {', '.join(contract.positive.workloads)}")
    target_gain = aggregated_effects.get("target_kernel_gain_pct")
    e2e_gain = aggregated_effects.get("end_to_end_gain_pct")
    lines.append(f"- measured target_kernel_gain_pct: {target_gain}")
    lines.append(f"- measured end_to_end_gain_pct: {e2e_gain}")
    lines.append("")

    lines.append("## Non-trigger / losing envelope (boundary)")
    if contract.boundary.dimensions:
        for name, values in contract.boundary.dimensions:
            lines.append(f"- {name}: {', '.join(str(v) for v in values)}")
    else:
        lines.append("- (no boundary dimensions declared)")
    lines.append("")

    lines.append("## Controls")
    lines.append(f"- models: {', '.join(contract.controls.models)}")
    lines.append(f"- workloads: {', '.join(contract.controls.workloads)}")
    regression = aggregated_effects.get("max_control_regression_pct")
    lines.append(f"- measured max_control_regression_pct (worst control): {regression}")
    lines.append("")

    lines.append("## Correctness")
    if contract.correctness.required_checks:
        lines.append(f"- required: {', '.join(contract.correctness.required_checks)}")
    else:
        lines.append("- required: (none declared)")
    lines.append(f"- gate passed: {correctness_gate.get('passed')}")
    missing = correctness_gate.get("missing_checks") or []
    failed = correctness_gate.get("failed_checks") or []
    if missing:
        lines.append(f"- missing: {', '.join(missing)}")
    if failed:
        lines.append(f"- failed: {', '.join(failed)}")
    lines.append("")

    lines.append("## Generalised rule")
    if generalisation_result is None:
        lines.append("- (no generalisation attempted for this contract)")
    else:
        lines.append(f"- passed: {generalisation_result.get('passed')}")
        for key in ("proven_groups", "added_coverage_pct", "median_regret_pct", "upper95_regret_pct"):
            if key in generalisation_result:
                lines.append(f"- {key}: {generalisation_result[key]}")
    lines.append("")

    lines.append("## Promotion decision")
    lines.append(f"- passed: {promotion_gate.get('passed')}")
    reasons = promotion_gate.get("reasons") or []
    if reasons:
        lines.append("- blocked by:")
        for reason in reasons:
            lines.append(f"  - {reason}")
    else:
        lines.append("- no blocking reasons")
    lines.append("")

    return "\n".join(lines)
