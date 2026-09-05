"""Experiment Contract: schema, validator, and registry (EC01/EC02).

The missing layer between an external/experimental optimization and BigCherry's
existing autotune/campaign/evidence machinery is *experimental intent*: what an
optimization claims to improve, the signatures/workloads that should trigger
it, controls that must not regress, boundary cases that define its safe
envelope, correctness requirements, and promotion thresholds.

This module is deliberately NOT a second candidate schema or a second
benchmark framework; the current contract reference is
docs/reference/experiments/EXPERIMENT_CONTRACT.md. A contract's `hypothesis.family` names
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
import random
import math
import statistics
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..tuning.schema import FAMILIES

EXPECTED_EFFECTS: tuple[str, ...] = ("performance", "correctness", "both")

# Guide section 2 + the worked example in section 4. Not closed against
# hypothetical future workloads by paranoia -- closed because an unlisted
# workload tag is far more likely a typo than a genuinely new evaluation
# axis, and a contract silently doing nothing under a misspelled tag is a
# worse failure mode than a loud rejection.
WORKLOAD_TAGS: tuple[str, ...] = (
    "decode", "prefill", "mtp_verify", "moe_prefill", "moe_decode",
    "long_context", "gdn_prefill", "multi_gpu_copy", "small_m",
    # VA10: RD58's real claim (repeated multi-GPU state-restore integrity)
    # is not multi_gpu_copy -- that tag is a generic transfer workload, not
    # a save/restore cycle. Do not misuse multi_gpu_copy for this claim
    # class (GPT round 3/4 review).
    "state_restore",
)

# Guide Appendix A's concrete correctness requirements across the 12xx
# patches: backend_reference (0100-class HI correctness harness), greedy_parity
# (temp-0 determinism, the 1002 MTP case), bit_identical (1204/1205's VDR/dual-
# output gates), ppl_equality (1207's MoE fusion gate).
CORRECTNESS_CHECKS: tuple[str, ...] = (
    "backend_reference", "greedy_parity", "bit_identical", "ppl_equality",
    # VA10: RD58's claim needs an affirmative "state correctly restored"
    # check, not an absence-of-fault claim -- "zero observed faults" is not
    # proof (Brutus's own 530+ cycles never reproduced the originating SDMA
    # fault, so failing to observe it again would prove nothing). This
    # token names what IS checked: saved state -> repeated multi-GPU
    # restore -> restored state/continuation agrees with reference.
    "state_restore_integrity",
)

ACCEPTANCE_FIELDS: tuple[str, ...] = (
    "target_kernel_gain_pct", "end_to_end_gain_pct", "max_control_regression_pct",
    "resource_limits", "effect_evidence_policy", "min_paired_rounds",
)

# VA24: how a declared acceptance bound must be evidenced.
#
#   point_estimate_v1     -- legacy. Compare the point estimate against the
#                            bound. This is what every contract predating
#                            VA24 used, and it cannot distinguish a real
#                            effect from a lucky sample.
#   ci95_threshold_bound_v1 -- the bound must be established by the interval,
#                            not the point estimate:
#                              gain:       ci95_low  >= declared threshold
#                              regression: ci95_high <= declared budget
#
# Note the gain rule is deliberately STRONGER than "CI excludes zero"
# (dev-gpt-agent, req_cd86e5fd4a3b4328). "Excludes zero" only establishes
# "probably positive"; +1.1% with CI [0.1, 2.1] would clear a 1.0 bound
# without ever establishing a 1.0% gain. Requiring the lower bound to reach
# the threshold establishes the claim the contract actually makes.
EFFECT_EVIDENCE_POLICIES: tuple[str, ...] = (
    "point_estimate_v1",
    "ci95_threshold_bound_v1",
)
DEFAULT_EFFECT_EVIDENCE_POLICY = "point_estimate_v1"

# VA12: RD73-class patches (a stable graph-cache key) trade a timing claim
# against a resource-cost claim -- retaining more shape-specific cache
# entries can win on timing while quietly growing memory/entry-count
# unboundedly. `metric` is deliberately NOT a closed enum (like
# SourceEvidence.metric above): resource kinds genuinely vary
# (graph_cache_entries, graph_cache_resident_bytes, ...) and a fixed
# vocabulary would either lose precision or invite a wrong-but-close
# mapping. `unit` IS closed -- a dimensional mismatch (bytes read as a
# count or vice versa) is a real, dangerous class of error.
RESOURCE_UNITS: tuple[str, ...] = ("bytes", "count")
RESOURCE_LIMIT_FIELDS: tuple[str, ...] = ("metric", "unit", "max_value", "max_increase_pct")

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


def _effect_evidence_policy(raw: object, where: str) -> str:
    """VA24: parse acceptance.effect_evidence_policy, defaulting for
    contracts that predate it. Unknown values are a parse error rather than
    a silent fallback -- a typo must not quietly downgrade a contract to the
    weaker point-estimate rule."""
    if raw is None:
        return DEFAULT_EFFECT_EVIDENCE_POLICY
    if not isinstance(raw, str) or raw not in EFFECT_EVIDENCE_POLICIES:
        raise ExperimentContractError(
            f"{where} must be one of {list(EFFECT_EVIDENCE_POLICIES)} "
            f"({raw!r} given)"
        )
    return raw


def _min_paired_rounds(raw: object, where: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ExperimentContractError(f"{where} must be an integer")
    if raw < 1:
        raise ExperimentContractError(
            f"{where} must be >= 1 ({raw!r} given)"
        )
    return raw


def _non_negative_int(raw: object, where: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ExperimentContractError(
            f"{where} must be a non-negative integer, got {raw!r}"
        )
    return raw


def _optional_bool(raw: object, where: str) -> bool | None:
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise ExperimentContractError(f"{where} must be a boolean")
    return raw


@dataclass(frozen=True, order=True)
class DriverVersion:
    """Comparable, structured driver version; free-form version strings are
    intentionally not accepted by the contract schema."""

    major: int
    minor: int
    patch: int = 0

    def __post_init__(self) -> None:
        for name, value in (("major", self.major), ("minor", self.minor),
                            ("patch", self.patch)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExperimentContractError(
                    f"driver version {name} must be a non-negative integer, got {value!r}"
                )

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def _driver_version(raw: object, where: str) -> DriverVersion:
    if isinstance(raw, DriverVersion):
        return raw
    if not isinstance(raw, list) or len(raw) not in (2, 3):
        raise ExperimentContractError(
            f"{where} must be a [major, minor] or [major, minor, patch] integer list"
        )
    values = [_non_negative_int(value, f"{where}[{index}]")
              for index, value in enumerate(raw)]
    return DriverVersion(*values)


@dataclass(frozen=True)
class GpuCountConstraint:
    """Inclusive GPU-count range. At least one bound must be declared."""

    minimum: int | None = None
    maximum: int | None = None

    def __post_init__(self) -> None:
        if self.minimum is None and self.maximum is None:
            raise ExperimentContractError(
                "gpu_count must declare minimum and/or maximum"
            )
        for name, value in (("minimum", self.minimum), ("maximum", self.maximum)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ExperimentContractError(
                    f"gpu_count.{name} must be a positive integer, got {value!r}"
                )
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ExperimentContractError(
                "gpu_count.minimum must not exceed gpu_count.maximum"
            )

    def matches(self, actual: int) -> bool:
        return ((self.minimum is None or actual >= self.minimum)
                and (self.maximum is None or actual <= self.maximum))


def _gpu_count_constraint(raw: object, where: str) -> GpuCountConstraint | None:
    if raw is None:
        return None
    data = _table(raw, where)
    unknown = sorted(set(data) - {"minimum", "maximum"})
    if unknown:
        raise ExperimentContractError(
            f"{where} names unknown field(s): {', '.join(unknown)}"
        )
    minimum = data.get("minimum")
    maximum = data.get("maximum")
    if minimum is not None:
        minimum = _non_negative_int(minimum, f"{where}.minimum")
        if minimum == 0:
            raise ExperimentContractError(f"{where}.minimum must be >= 1")
    if maximum is not None:
        maximum = _non_negative_int(maximum, f"{where}.maximum")
        if maximum == 0:
            raise ExperimentContractError(f"{where}.maximum must be >= 1")
    return GpuCountConstraint(minimum=minimum, maximum=maximum)


@dataclass(frozen=True)
class DriverVersionConstraint:
    """Inclusive structured driver-version range."""

    minimum: DriverVersion | None = None
    maximum: DriverVersion | None = None

    def __post_init__(self) -> None:
        if self.minimum is None and self.maximum is None:
            raise ExperimentContractError(
                "driver must declare minimum and/or maximum"
            )
        for name, value in (("minimum", self.minimum), ("maximum", self.maximum)):
            if value is not None and not isinstance(value, DriverVersion):
                raise ExperimentContractError(
                    f"driver.{name} must be a DriverVersion or absent"
                )
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ExperimentContractError(
                "driver.minimum must not exceed driver.maximum"
            )

    def matches(self, actual: DriverVersion) -> bool:
        return ((self.minimum is None or actual >= self.minimum)
                and (self.maximum is None or actual <= self.maximum))


def _driver_constraint(raw: object, where: str) -> DriverVersionConstraint | None:
    if raw is None:
        return None
    data = _table(raw, where)
    unknown = sorted(set(data) - {"minimum", "maximum"})
    if unknown:
        raise ExperimentContractError(
            f"{where} names unknown field(s): {', '.join(unknown)}"
        )
    return DriverVersionConstraint(
        minimum=(_driver_version(data["minimum"], f"{where}.minimum")
                 if data.get("minimum") is not None else None),
        maximum=(_driver_version(data["maximum"], f"{where}.maximum")
                 if data.get("maximum") is not None else None),
    )


@dataclass(frozen=True)
class DeviceTraits:
    """Verified hardware facts supplied by an external detector or test.

    ``driver_version`` is optional because the current repository has no
    driver-detection implementation. A scope requiring ``driver`` therefore
    fails closed unless a detector supplies this field explicitly.
    """

    integrated: bool
    uma: bool
    peer_access: bool
    gpu_count: int
    driver_version: DriverVersion | None = None

    def __post_init__(self) -> None:
        for name, value in (("integrated", self.integrated), ("uma", self.uma),
                            ("peer_access", self.peer_access)):
            if not isinstance(value, bool):
                raise ExperimentContractError(f"hardware.{name} must be a boolean")
        if isinstance(self.gpu_count, bool) or not isinstance(self.gpu_count, int) or self.gpu_count < 1:
            raise ExperimentContractError(
                f"hardware.gpu_count must be a positive integer, got {self.gpu_count!r}"
            )
        if self.driver_version is not None and not isinstance(self.driver_version, DriverVersion):
            raise ExperimentContractError(
                "hardware.driver_version must be a DriverVersion or absent"
            )


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
    """Execution scope, including optional verified device-trait requirements.

    RD22's real requirement can now be expressed as
    ``Scope(backend="hip", architectures=("gfx1151",), weight_types=(),
    integrated=True, uma=True)``. The architecture remains useful for
    compilation coverage, while integrated/UMA is the correctness eligibility
    fact that distinguishes RD22 from discrete devices.

    In TOML, a multi-GPU requirement is ``[scope.gpu_count]`` with
    ``minimum = 2``; a driver requirement is ``[scope.driver]`` with a
    structured ``minimum = [major, minor, patch]``. Version strings are not
    accepted.
    """

    backend: str
    architectures: tuple[str, ...]
    weight_types: tuple[str, ...]
    integrated: bool | None = None
    uma: bool | None = None
    peer_access: bool | None = None
    gpu_count: GpuCountConstraint | None = None
    driver: DriverVersionConstraint | None = None


def evaluate_scope_eligibility(scope: Scope, hardware: DeviceTraits | None) -> bool:
    """Return whether verified hardware satisfies device-trait requirements.

    Trait-bearing scopes require an explicit ``DeviceTraits`` observation;
    missing observations, including an unavailable driver version, raise
    ``ExperimentContractError`` rather than passing silently. A legacy scope
    with no trait requirements remains eligible without hardware facts.
    """
    requirements_declared = any((scope.integrated is not None, scope.uma is not None,
                                scope.peer_access is not None, scope.gpu_count is not None,
                                scope.driver is not None))
    if not requirements_declared:
        return True
    if hardware is None:
        raise ExperimentContractError(
            "device-trait eligibility cannot be verified without hardware traits"
        )
    if scope.integrated is not None and scope.integrated != hardware.integrated:
        return False
    if scope.uma is not None and scope.uma != hardware.uma:
        return False
    if scope.peer_access is not None and scope.peer_access != hardware.peer_access:
        return False
    if scope.gpu_count is not None and not scope.gpu_count.matches(hardware.gpu_count):
        return False
    if scope.driver is not None:
        if hardware.driver_version is None:
            raise ExperimentContractError(
                "driver-version eligibility cannot be verified: hardware driver "
                "detection is unavailable"
            )
        if not scope.driver.matches(hardware.driver_version):
            return False
    return True


def _version_payload(version: DriverVersion | None) -> list[int] | None:
    if version is None:
        return None
    return [version.major, version.minor, version.patch]


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
class SourceEvidence:
    """EC17: what the EXTERNAL source actually reported -- kept structurally
    separate from `Acceptance` (BigCherry's OWN required thresholds).
    Purely documentary/provenance: does not drive any promotion-gate logic
    by itself (see source_evidence_mismatch_warning below for the one
    consistency check that DOES read it, which only ever flags, never
    blocks). `metric` is deliberately not a closed enum -- source reports
    genuinely vary (kernel_us, pp512, tg128, end_to_end_tps, ...) and
    forcing them into a fixed vocabulary would either lose real precision
    or invite a wrong-but-close mapping. This is the fix for a real
    external-review finding: a source-reported +7.4% end-to-end number is
    a reason to investigate, not a requirement that BigCherry's own kernel
    acceptance threshold be numerically identical -- they are different
    measurements, on different hardware, with different methodology."""
    metric: str
    value_pct: float
    hardware: str
    workload: str


@dataclass(frozen=True)
class ResourceLimit:
    """One declared resource-cost budget (VA12) -- e.g. RD73's graph-cache
    entry count/memory. At least one of max_value (an absolute ceiling on
    the subject's measured value) or max_increase_pct (a bound on growth
    over the paired control) must be declared -- a limit that checks
    neither is not a limit -- but both MAY be declared together (an
    absolute ceiling and a relative-growth bound are independent checks,
    not alternatives)."""
    metric: str
    unit: str
    max_value: float | None = None
    max_increase_pct: float | None = None


@dataclass(frozen=True)
class Acceptance:
    """BigCherry's OWN required thresholds -- a contract's promotion gate is
    evaluated against these, never against SourceEvidence directly. These
    may or may not numerically match what an external source reported
    (see SourceEvidence's docstring); source_evidence_mismatch_warning
    flags a large or backwards divergence for a human to look at, but a
    mismatch is never by itself a parse error or a promotion blocker."""
    target_kernel_gain_pct: float | None
    end_to_end_gain_pct: float | None
    max_control_regression_pct: float | None
    resource_limits: tuple[ResourceLimit, ...] = ()
    # VA24. Defaulted so every pre-VA24 contract keeps its exact behaviour
    # AND its exact contract_hash (both fields are omitted from the identity
    # payload when left at their defaults, following VA12's resource_limits
    # precedent). A contract that opts in gets a new hash, which is correct:
    # it has changed what it demands of evidence.
    effect_evidence_policy: str = DEFAULT_EFFECT_EVIDENCE_POLICY
    # Minimum paired rounds a lane must contribute before its interval is
    # trusted. run_paired_lane() accepts pairs=1, whose bootstrap produces a
    # degenerate interval that can look arbitrarily significant, so an
    # interval-based policy without a rounds floor is not actually stronger
    # than the point estimate it replaces.
    min_paired_rounds: int | None = None


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
    source_evidence: SourceEvidence | None = None

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
            "integrated": contract.scope.integrated,
            "uma": contract.scope.uma,
            "peer_access": contract.scope.peer_access,
            "gpu_count": (
                {
                    "minimum": contract.scope.gpu_count.minimum,
                    "maximum": contract.scope.gpu_count.maximum,
                }
                if contract.scope.gpu_count is not None else None
            ),
            "driver": (
                {
                    "minimum": _version_payload(contract.scope.driver.minimum),
                    "maximum": _version_payload(contract.scope.driver.maximum),
                }
                if contract.scope.driver is not None else None
            ),
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
            # VA12: omitted entirely (not an empty list) when no resource
            # limits are declared, so a contract predating this field keeps
            # its exact original contract_hash -- only a contract that
            # actually adds a resource limit gets a new hash, matching the
            # "an edit changes the hash" rule without a blanket digest-
            # domain bump for every existing contract.
            **(
                {
                    "resource_limits": [
                        {
                            "metric": limit.metric, "unit": limit.unit,
                            "max_value": limit.max_value,
                            "max_increase_pct": limit.max_increase_pct,
                        }
                        for limit in contract.acceptance.resource_limits
                    ]
                }
                if contract.acceptance.resource_limits else {}
            ),
            # VA24: same treatment as resource_limits above -- omitted while
            # left at the default, so every contract predating VA24 keeps its
            # exact original contract_hash. A contract that actually opts into
            # an interval-based policy (or declares a rounds floor) gets a new
            # hash, which is correct: it has changed what it demands of
            # evidence, so previously-recorded evidence should not silently
            # continue to satisfy it.
            **(
                {"effect_evidence_policy": contract.acceptance.effect_evidence_policy}
                if contract.acceptance.effect_evidence_policy != DEFAULT_EFFECT_EVIDENCE_POLICY
                else {}
            ),
            **(
                {"min_paired_rounds": contract.acceptance.min_paired_rounds}
                if contract.acceptance.min_paired_rounds is not None else {}
            ),
        },
        "source_evidence": (
            {
                "metric": contract.source_evidence.metric,
                "value_pct": contract.source_evidence.value_pct,
                "hardware": contract.source_evidence.hardware,
                "workload": contract.source_evidence.workload,
            }
            if contract.source_evidence is not None else None
        ),
    }


def _contract_digest(contract: ExperimentContract) -> str:
    encoded = json.dumps(
        _identity_payload(contract), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.blake2b(
        b"bigcherry/experiment-contract/v2\0" + encoded, digest_size=16
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
    unknown_scope = sorted(set(scope_data) - {
        "backend", "architectures", "weight_types", "integrated", "uma",
        "peer_access", "gpu_count", "driver",
    })
    if unknown_scope:
        raise ExperimentContractError(
            f"{where}.scope names unknown field(s): {', '.join(unknown_scope)}"
        )
    scope = Scope(
        backend=_required_string(scope_data.get("backend"), f"{where}.scope.backend"),
        architectures=_strings(scope_data.get("architectures"), f"{where}.scope.architectures",
                                required=True),
        weight_types=_strings(scope_data.get("weight_types"), f"{where}.scope.weight_types"),
        integrated=_optional_bool(scope_data.get("integrated"), f"{where}.scope.integrated"),
        uma=_optional_bool(scope_data.get("uma"), f"{where}.scope.uma"),
        peer_access=_optional_bool(
            scope_data.get("peer_access"), f"{where}.scope.peer_access"),
        gpu_count=_gpu_count_constraint(
            scope_data.get("gpu_count"), f"{where}.scope.gpu_count"),
        driver=_driver_constraint(scope_data.get("driver"), f"{where}.scope.driver"),
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

    # A lane is identified by (model, workload) -- EvaluationSet carries no
    # further axis, so a lane named by BOTH roles is one measurement asked to
    # satisfy two contradictory requirements at once: gain at least
    # target_kernel_gain_pct AND change by no more than
    # max_control_regression_pct. Worse, it makes the regression budget
    # self-referential: the "control" is the treatment, so it can never
    # detect the collateral damage a control exists to catch.
    shared_lanes = sorted(
        (model, workload)
        for model in set(positive.models) & set(controls.models)
        for workload in set(positive.workloads) & set(controls.workloads)
    )
    if shared_lanes:
        rendered = ", ".join(f"{model}/{workload}" for model, workload in shared_lanes)
        raise ExperimentContractError(
            f"{where}: lane(s) {rendered} appear in BOTH positive and controls -- "
            f"a lane cannot be both the thing that must improve and the thing that "
            f"must hold constant. A control must name a model/workload the "
            f"hypothesis does NOT claim to speed up, otherwise the regression "
            f"budget is measured against the treatment itself and is vacuous"
        )

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
    resource_limits_raw = acceptance_data.get("resource_limits")
    resource_limits: list[ResourceLimit] = []
    if resource_limits_raw is not None:
        rl_where = f"{where}.acceptance.resource_limits"
        if not isinstance(resource_limits_raw, list):
            raise ExperimentContractError(f"{rl_where} must be a list of tables")
        seen_metrics: set[str] = set()
        for index, entry in enumerate(resource_limits_raw):
            entry_where = f"{rl_where}[{index}]"
            entry_data = _table(entry, entry_where)
            unknown_fields = sorted(set(entry_data) - set(RESOURCE_LIMIT_FIELDS))
            if unknown_fields:
                raise ExperimentContractError(
                    f"{entry_where} names unknown field(s): {', '.join(unknown_fields)}"
                )
            metric = _required_string(entry_data.get("metric"), f"{entry_where}.metric")
            if metric in seen_metrics:
                raise ExperimentContractError(f"{rl_where} declares duplicate metric {metric!r}")
            seen_metrics.add(metric)
            unit = _required_string(
                entry_data.get("unit"), f"{entry_where}.unit", choices=RESOURCE_UNITS
            )
            max_value = _percent(entry_data.get("max_value"), f"{entry_where}.max_value")
            max_increase_pct = _percent(
                entry_data.get("max_increase_pct"), f"{entry_where}.max_increase_pct"
            )
            if max_value is None and max_increase_pct is None:
                raise ExperimentContractError(
                    f"{entry_where} must declare max_value or max_increase_pct "
                    "(a limit that checks neither is not a limit)"
                )
            resource_limits.append(ResourceLimit(
                metric=metric, unit=unit, max_value=max_value, max_increase_pct=max_increase_pct,
            ))
    resource_limits.sort(key=lambda limit: limit.metric)

    acceptance = Acceptance(
        target_kernel_gain_pct=_percent(
            acceptance_data.get("target_kernel_gain_pct"), f"{where}.acceptance.target_kernel_gain_pct"),
        end_to_end_gain_pct=_percent(
            acceptance_data.get("end_to_end_gain_pct"), f"{where}.acceptance.end_to_end_gain_pct"),
        max_control_regression_pct=_percent(
            acceptance_data.get("max_control_regression_pct"),
            f"{where}.acceptance.max_control_regression_pct"),
        resource_limits=tuple(resource_limits),
        effect_evidence_policy=_effect_evidence_policy(
            acceptance_data.get("effect_evidence_policy"),
            f"{where}.acceptance.effect_evidence_policy"),
        min_paired_rounds=_min_paired_rounds(
            acceptance_data.get("min_paired_rounds"),
            f"{where}.acceptance.min_paired_rounds"),
    )
    # VA24 P0 (dev-gpt-agent, req_d563bd481bcf4324): an interval policy
    # without a rounds floor is not actually stronger than the point estimate
    # it replaces -- run_paired_lane() accepts pairs=1, whose bootstrap yields
    # a degenerate interval that can look arbitrarily significant. Requiring
    # the floor makes "this contract uses intervals" mean "this contract
    # guarantees a minimum evidence depth", rather than leaving that to be
    # discovered per-contract.
    if (acceptance.effect_evidence_policy == "ci95_threshold_bound_v1"
            and acceptance.min_paired_rounds is None):
        raise ExperimentContractError(
            f"{where}.acceptance: effect_evidence_policy="
            f"'ci95_threshold_bound_v1' requires an explicit min_paired_rounds "
            f"-- an interval computed from a single paired round is degenerate "
            f"and can look arbitrarily significant, so a confidence policy with "
            f"no evidence-depth floor is weaker than it appears"
        )
    if acceptance.max_control_regression_pct is None:
        raise ExperimentContractError(
            f"{where}.acceptance.max_control_regression_pct is required -- every "
            f"contract must declare the regression budget its controls are held to, "
            f"even a correctness-only contract with no performance claim (see "
            f"docs/reference/experiments/EXPERIMENT_CONTRACT.md: every contract "
            f"needs an explicit control-regression budget)"
        )

    source_evidence: SourceEvidence | None = None
    source_evidence_raw = data.get("source-evidence")
    if source_evidence_raw is not None:
        se_where = f"{where}.source-evidence"
        se_data = _table(source_evidence_raw, se_where)
        se_metric = _required_string(se_data.get("metric"), f"{se_where}.metric")
        se_value = se_data.get("value_pct")
        if isinstance(se_value, bool) or not isinstance(se_value, (int, float)):
            raise ExperimentContractError(f"{se_where}.value_pct must be a number")
        se_value = float(se_value)
        if not math.isfinite(se_value):
            raise ExperimentContractError(
                f"{se_where}.value_pct must be a finite number -- {se_value!r} given"
            )
        se_hardware = _required_string(se_data.get("hardware"), f"{se_where}.hardware")
        se_workload = _required_string(se_data.get("workload"), f"{se_where}.workload")
        unknown_se = sorted(set(se_data) - {"metric", "value_pct", "hardware", "workload"})
        if unknown_se:
            raise ExperimentContractError(
                f"{se_where} names unknown field(s): {', '.join(unknown_se)}"
            )
        source_evidence = SourceEvidence(
            metric=se_metric, value_pct=se_value, hardware=se_hardware, workload=se_workload,
        )

    unknown_top = sorted(set(data) - {
        "title", "source", "hypothesis", "target", "prerequisites", "scope",
        "positive", "controls", "boundary", "correctness", "acceptance",
        "source-evidence",
    })
    if unknown_top:
        raise ExperimentContractError(f"{where}: unknown field(s): {', '.join(unknown_top)}")

    return ExperimentContract(
        id=contract_id, title=title, source=source, hypothesis=hypothesis, target=target,
        prerequisites=prerequisites, scope=scope, positive=positive, controls=controls,
        boundary=boundary, correctness=CorrectnessRequirements(required_checks),
        acceptance=acceptance, source_evidence=source_evidence,
    )


def source_evidence_mismatch_warning(contract: ExperimentContract) -> str | None:
    """EC17: non-fatal consistency check between a contract's SourceEvidence
    (what the external source reported) and its own Acceptance threshold
    (what BigCherry requires). Never raises, never blocks parsing or
    promotion -- returns a human-readable warning string when the numbers
    look structurally inconsistent, or None when there's nothing to flag
    (including when source_evidence is absent, or acceptance has no
    target_kernel_gain_pct to compare against -- a correctness-only
    contract has nothing numeric to check here).

    Flags two cases: (1) acceptance requires MORE gain than the source
    itself ever reported -- an impossible-to-meet bar, since BigCherry is
    asking to exceed the only evidence that the optimization works at all;
    (2) acceptance is set far below what the source reported (more than a
    2x gap) -- not wrong, but worth a human noticing the gate is much
    looser than the hypothesis being tested, in case that's accidental
    rather than a deliberate conservative choice."""
    if contract.source_evidence is None:
        return None
    target = contract.acceptance.target_kernel_gain_pct
    if target is None:
        return None
    source_value = contract.source_evidence.value_pct
    if target > source_value:
        return (
            f"acceptance.target_kernel_gain_pct ({target}%) exceeds the source's own "
            f"reported gain ({source_value}% {contract.source_evidence.metric} on "
            f"{contract.source_evidence.hardware}) -- this threshold can never be met "
            f"even if the port perfectly reproduces the source's result"
        )
    if source_value > 0 and target < source_value / 2:
        return (
            f"acceptance.target_kernel_gain_pct ({target}%) is less than half the "
            f"source's reported gain ({source_value}% {contract.source_evidence.metric} "
            f"on {contract.source_evidence.hardware}) -- confirm this conservative gate "
            f"is deliberate, not an oversight"
        )
    return None


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
        from ..core import paths
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


def known_model_ids_from_models_registry(
    path: str | Path | None = None,
) -> frozenset[str]:
    """Real cross-check input for ``load_contracts(known_model_ids=...)``:
    every ``[[models]] id`` registered in ``config/models.toml``
    (``paths.MODELS`` by default).

    Exists because ``_evaluation_set()`` constrains ``workloads`` to
    WORKLOAD_TAGS but constrained ``models`` to nothing -- an evaluation set
    could name a model that had never existed and validation would pass,
    deferring the failure to hardware time after a build and a model load had
    been spent. Worse, a silently-wrong model ref means recorded evidence
    CLAIMS a lane it did not measure, since the ref is a label and the actual
    gguf arrives separately via ``--model``.

    Kept as an explicit opt-in call for the same reason the source-id
    equivalent is: a caller wanting the real registry cross-check asks for
    it; a unit test building a contract in isolation is not forced to
    maintain a models.toml fixture."""
    import tomllib as _tomllib

    resolved = Path(path) if path is not None else None
    if resolved is None:
        from ..core import paths
        resolved = paths.MODELS
    try:
        raw = _tomllib.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ExperimentContractError(f"no models registry at {resolved}") from None
    except _tomllib.TOMLDecodeError as exc:
        raise ExperimentContractError(f"{resolved}: {exc}") from None
    return frozenset(
        entry["id"] for entry in raw.get("models", []) if isinstance(entry, dict) and entry.get("id")
    )


def load_contracts(path: str | Path, *,
                    known_source_ids: frozenset[str] | None = None,
                    known_model_ids: frozenset[str] | None = None) -> ContractRegistry:
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

    if known_model_ids is not None:
        for contract in contracts.values():
            for role, evaluation_set in (
                ("positive", contract.positive), ("controls", contract.controls),
            ):
                for model in evaluation_set.models:
                    if model not in known_model_ids:
                        raise ExperimentContractError(
                            f"contract.{contract.id}.{role}.models: {model!r} is not "
                            f"a known models.toml entry -- an evaluation set may only "
                            f"name a registered model, otherwise the recorded lane "
                            f"identifies a measurement nothing can resolve"
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
    # VA24. The producing report (comparisons.run_comparison() /
    # ab_benchmark.paired_summary()) already computes these; LaneEffect
    # previously discarded them, so the promotion gate never saw an interval.
    # Optional and defaulted so every existing construction is unchanged.
    ci95_low_pct: float | None = None
    ci95_high_pct: float | None = None
    paired_rounds: int | None = None
    # VA24: the ordered per-pair ratio vector this lane's interval was
    # bootstrapped from -- the sufficient statistic for recomputing an
    # AGGREGATE interval across several lanes without re-running benchmarks.
    pair_ratios: tuple[float, ...] = ()


def bootstrap_fixed_composite_mean(
    lanes: list[LaneEffect], *, seed: int = 0, resamples: int = 10_000,
) -> tuple[float, float] | None:
    """VA24: CI for the mean effect across a FIXED set of positive lanes.

    The estimand is the existing point-estimate semantic -- mean of the
    per-lane effects -- and this puts an interval on it without changing it.

    Critically it resamples ONLY WITHIN each lane, never the lanes
    themselves::

        for replicate b:
            E_j[b] = effect(resample lane j's own pair ratios)   for each lane j
            G[b]   = mean_j E_j[b]
        CI(G) = percentiles(G)

    Bootstrapping lane identity would be wrong: decode and prefill are fixed
    components the contract names, not IID draws from a population of
    workloads, and with two heterogeneous lanes "sampling a lane" is
    meaningless. What is uncertain is each lane's own measurement, so that is
    what gets resampled (dev-gpt-agent, req_a667633429fa4c9e).

    Mirrors block_bootstrap_effect()'s estimator exactly -- geometric mean of
    per-pair ratios, expressed as a percentage -- so a single-lane call here
    reproduces that function's own interval rather than a second, subtly
    different statistic.

    Returns None when any lane lacks its ratio vector, so the caller emits no
    interval and the gate reports "invalid" rather than guessing.
    """
    if not lanes:
        return None
    lane_logs: list[list[float]] = []
    for lane in lanes:
        if not lane.pair_ratios or any(
            not _finite_number(ratio) or ratio <= 0 for ratio in lane.pair_ratios
        ):
            return None
        lane_logs.append([math.log(ratio) for ratio in lane.pair_ratios])

    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(resamples):
        lane_effects_pct = []
        for logs in lane_logs:
            resampled = [rng.choice(logs) for _ in logs]
            lane_effects_pct.append(
                100.0 * (math.exp(statistics.mean(resampled)) - 1.0))
        replicates.append(statistics.mean(lane_effects_pct))
    replicates.sort()
    return (
        replicates[int(0.025 * resamples)],
        replicates[min(resamples - 1, int(0.975 * resamples))],
    )


#: Fewest measurement sessions ``bootstrap_session_effect()`` will estimate
#: from. A cluster bootstrap draws whole sessions with replacement, so with
#: very few clusters the resample space is tiny (S=2 admits 3 distinct
#: multisets, S=3 admits 10) and the resulting interval is badly
#: anticonservative -- it would report a narrow interval precisely when
#: between-session drift is least well characterised. Below this the function
#: returns None and the caller reports "not evaluable" rather than a number
#: nobody should act on, matching this module's existing failure style.
MIN_BOOTSTRAP_SESSIONS = 4


def bootstrap_session_effect(
    sessions: list[tuple[float, ...]], *, seed: int = 0, resamples: int = 10_000,
) -> dict[str, object] | None:
    """Effect and interval across repeated measurement SESSIONS of one lane.

    ``sessions`` is one per-pair ratio vector per session (the same
    ``pair_ratios`` sufficient statistic ``block_bootstrap_effect()`` already
    records, one entry per independent run).

    WHY THIS EXISTS. ``block_bootstrap_effect()`` resamples pairs within a
    single run, so its interval covers only within-session variation. RD73
    measured the same build three times and got +1.855%, +1.717% and +1.249%;
    the third run's point estimate fell BELOW the second run's ci95_low. The
    runs are not inconsistent measurements of different things -- they are
    honest measurements taken on different occasions, and the drift between
    occasions is real variance that a within-run interval structurally cannot
    see. Quoting a single run's interval therefore overstates precision.

    WHY RESAMPLING SESSION IDENTITY IS CORRECT HERE, where
    ``bootstrap_fixed_composite_mean()`` deliberately refuses to resample LANE
    identity: a contract's lanes are fixed components it names by hand
    (decode, prefill) -- "sampling a lane" is meaningless. Sessions are the
    opposite: they are exchangeable draws from the population of occasions on
    which this measurement could have been taken, and that population is
    exactly what a claim about the effect generalises over. Drawing sessions
    with replacement is what propagates between-occasion variance into the
    interval::

        for replicate b:
            draw S sessions with replacement from the S observed
            for each drawn session: resample its own pairs with replacement
            G[b] = mean of those sessions' geometric effects
        CI(G) = percentiles(G)

    Sessions are weighted EQUALLY rather than by pair count: the session is
    the unit of replication, so a run that happened to collect more pairs
    should not speak louder about where the true effect lies.

    Returns None -- never a guess -- when fewer than MIN_BOOTSTRAP_SESSIONS
    are supplied, or when any session's ratio vector is empty or non-finite.
    """
    if len(sessions) < MIN_BOOTSTRAP_SESSIONS:
        return None
    session_logs: list[list[float]] = []
    for ratios in sessions:
        if not ratios or any(not _finite_number(r) or r <= 0 for r in ratios):
            return None
        session_logs.append([math.log(r) for r in ratios])

    def _effect_pct(logs: list[float]) -> float:
        return 100.0 * (math.exp(statistics.mean(logs)) - 1.0)

    point = statistics.mean(_effect_pct(logs) for logs in session_logs)
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(resamples):
        drawn = [rng.choice(session_logs) for _ in session_logs]
        replicates.append(statistics.mean(
            _effect_pct([rng.choice(logs) for _ in logs]) for logs in drawn
        ))
    replicates.sort()
    per_session = [_effect_pct(logs) for logs in session_logs]
    return {
        "sessions": len(session_logs),
        "paired_rounds_total": sum(len(logs) for logs in session_logs),
        "geometric_effect_pct": point,
        "ci95_low_pct": replicates[int(0.025 * resamples)],
        "ci95_high_pct": replicates[min(resamples - 1, int(0.975 * resamples))],
        "per_session_effect_pct": tuple(per_session),
        # The variance component a within-run interval cannot see. Reported so
        # a reader can compare it against the single-run intervals directly.
        "between_session_sd_pct": (
            statistics.stdev(per_session) if len(per_session) > 1 else 0.0
        ),
        "resamples": resamples, "seed": seed,
    }


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
      direction, i.e. numerically most negative) effect across ALL
      CONTROL-role lanes, REGARDLESS of which metric each control lane
      reports -- never an average, and never filtered to target_metric.
      A real control lane very often measures a structurally different
      workload than the positive lane (e.g. RD08: positive=decode/tg128,
      control=prefill/pp512) and reports its own natural metric; requiring
      an exact target_metric match would either force a fabricated
      same-metric relabeling or silently drop real heterogeneous control
      evidence (GPT round-2 correction, req_240634997c1a4ee9, after this
      exact confusion showed up in VA14's first RD08 composition test). A
      contract that wins on average while quietly regressing one control
      lane, on that lane's own metric, must not pass. A contract with zero
      control-role effects raises rather than silently reporting 0.0
      (undefined is not the same as "no regression").
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
        if effect.role == "control"
    ]
    if not control_effects:
        raise ExperimentContractError(
            f"contract {contract.id!r}: no control-role effects at all -- every "
            f"contract must measure its declared controls, an empty set is not "
            f"evidence of no regression"
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

    aggregated: dict[str, object] = {
        "target_kernel_gain_pct": statistics.mean(positive_target),
        "end_to_end_gain_pct": statistics.mean(e2e_effects) if e2e_effects else None,
        "max_control_regression_pct": max_control_regression_pct,
    }

    # VA24: attach intervals ONLY where they can be carried through exactly.
    #
    # For a single contributing lane the aggregate IS that lane, so its
    # interval transfers unchanged. For several lanes it does NOT:
    # mean(lane ci95_lows) is not the ci95_low of the mean effect, and the
    # interval of the lane with the worst POINT estimate is not the interval
    # of the worst regression. Computing the aggregate interval properly means
    # bootstrapping the aggregate statistic from the per-round paired data,
    # which this function does not receive.
    #
    # So multi-lane contracts deliberately get NO interval here. Under
    # ci95_threshold_bound_v1 the gate then reports "invalid" (unevaluable)
    # rather than passing or failing -- which is the correct fail-closed
    # outcome, and far better than emitting a plausible-looking number from
    # invalid statistics. Removing this restriction requires giving this
    # function the raw paired observations, not a cleverer formula.
    # (dev-gpt-agent, req_cd86e5fd4a3b4328, P0.)
    def _usable_interval(effect: LaneEffect) -> bool:
        """VA24 P0: validate the SOURCE interval atomically, before any
        derivation (dev-gpt-agent, req_d563bd481bcf4324).

        Requires finite point/low/high and ``low <= point <= high`` -- which
        also proves ``low <= high``, so no separate ordering check is needed.

        Validating here rather than at the gate matters for the control side:
        the regression transform ``max(0, -effect)`` can otherwise HIDE a
        malformed source interval (e.g. an inverted low/high still yields a
        plausible-looking non-negative regression bound). A lane whose
        interval fails this check contributes no interval at all, so an
        interval policy reports "invalid" instead of trusting a derived
        number computed from nonsense.
        """
        return (
            _finite_number(effect.geometric_effect_pct)
            and _finite_number(effect.ci95_low_pct)
            and _finite_number(effect.ci95_high_pct)
            and effect.ci95_low_pct <= effect.geometric_effect_pct <= effect.ci95_high_pct
        )

    def _sole(role: str, metric: str) -> LaneEffect | None:
        matching = [
            effect for effect in lane_effects
            if effect.role == role and effect.metric == metric
        ]
        return matching[0] if len(matching) == 1 else None

    def _gain_interval(field: str, metric: str) -> None:
        """Attach the gain interval for one metric.

        One contributing lane: the aggregate IS that lane, so its own
        interval transfers unchanged. Several lanes: bootstrap the mean
        across the FIXED lane set, resampling only within each lane. The
        reported paired_rounds is the MINIMUM across contributing lanes,
        since the evidence floor must be met by the weakest contributor --
        an aggregate is not made trustworthy by one well-sampled lane
        carrying an under-sampled one.
        """
        lanes = [
            effect for effect in lane_effects
            if effect.role == "positive" and effect.metric == metric
        ]
        if not lanes:
            return
        if len(lanes) == 1:
            if _usable_interval(lanes[0]):
                aggregated[f"{field}_ci95_low"] = lanes[0].ci95_low_pct
                aggregated[f"{field}_paired_rounds"] = lanes[0].paired_rounds
            return
        if not all(_usable_interval(lane) for lane in lanes):
            return
        interval = bootstrap_fixed_composite_mean(lanes)
        if interval is None:
            return
        rounds = [lane.paired_rounds for lane in lanes]
        aggregated[f"{field}_ci95_low"] = interval[0]
        aggregated[f"{field}_paired_rounds"] = (
            min(rounds) if all(isinstance(r, int) and not isinstance(r, bool) for r in rounds)
            else None
        )

    _gain_interval("target_kernel_gain_pct", target_metric)
    _gain_interval("end_to_end_gain_pct", e2e_metric)

    control_lanes = [effect for effect in lane_effects if effect.role == "control"]
    if len(control_lanes) == 1 and _usable_interval(control_lanes[0]):
        # regression = max(0, -effect), so the interval ENDPOINTS REVERSE
        # under the negation:
        #     R_low  = max(0, -E_ci95_high)
        #     R_high = max(0, -E_ci95_low)
        # The interval does NOT transfer across unchanged. Taking the effect's
        # ci95_high as the regression's upper bound would be a correctness
        # bug -- it would report the most OPTIMISTIC case as the worst case.
        # Flagged HIGH risk in review (dev-gpt-agent, req_a667633429fa4c9e)
        # and pinned by test_regression_interval_endpoints_reverse.
        aggregated["max_control_regression_pct_ci95_high"] = max(
            0.0, -control_lanes[0].ci95_low_pct)
        aggregated["max_control_regression_pct_paired_rounds"] = control_lanes[0].paired_rounds
    # len(control_lanes) > 1 deliberately attaches NO interval. Independent
    # per-lane 95% bounds are not a 95% FAMILY guarantee, and the right
    # simultaneous procedure (Bonferroni vs a direct bootstrap of the fixed
    # max-regression statistic vs something else) depends on cross-lane
    # dependence that no contract in this registry currently exercises --
    # every one has exactly one control lane. Building an untested correction
    # with no consumer is worse than failing closed, so the gate reports
    # "invalid" until a real multi-control contract exists to design against.

    return aggregated


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


# ------------------------------------------------------------ resource cost (VA12)


@dataclass(frozen=True)
class ResourceResult:
    """One resource limit's measured evidence -- e.g. RD73's real
    graph-cache entry count after a validation run. control_value is only
    required when the limit checks max_increase_pct; a pure max_value
    limit needs only the subject's own measurement."""
    metric: str
    unit: str
    subject_value: float
    control_value: float | None = None

    def __post_init__(self) -> None:
        if not self.metric:
            raise ExperimentContractError("ResourceResult.metric must be a non-empty string")
        if self.unit not in RESOURCE_UNITS:
            raise ExperimentContractError(
                f"ResourceResult.unit={self.unit!r} is not one of {', '.join(RESOURCE_UNITS)}"
            )
        for field_name, value in (("subject_value", self.subject_value), ("control_value", self.control_value)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ExperimentContractError(f"ResourceResult.{field_name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ExperimentContractError(
                    f"ResourceResult.{field_name} must be finite and non-negative, got {value!r}"
                )


def evaluate_resource_gate(
    contract: ExperimentContract, results: dict[str, ResourceResult],
) -> dict[str, object]:
    """VA12: every limit in contract.acceptance.resource_limits must have a
    ResourceResult in ``results`` with matching metric AND unit, and must
    satisfy whichever bound(s) the limit declares. Missing evidence is a
    FAIL, never "not applicable" -- a declared resource budget with no
    measurement proves nothing. A max_increase_pct limit with no
    control_value evidence is also a FAIL (the bound cannot be evaluated).
    A control_value of 0 paired with any positive subject_value is
    unbounded relative growth and FAILS max_increase_pct rather than
    dividing by zero or silently passing."""
    missing: list[str] = []
    failed: list[str] = []
    detail: dict[str, object] = {}
    for limit in contract.acceptance.resource_limits:
        result = results.get(limit.metric)
        if result is None:
            missing.append(limit.metric)
            continue
        if result.metric != limit.metric:
            failed.append(limit.metric)
            detail[limit.metric] = (
                f"metric mismatch: dict key {limit.metric!r} maps to a ResourceResult "
                f"whose own metric field is {result.metric!r}"
            )
            continue
        if result.unit != limit.unit:
            failed.append(limit.metric)
            detail[limit.metric] = f"unit mismatch: limit={limit.unit!r} result={result.unit!r}"
            continue
        reasons: list[str] = []
        if limit.max_value is not None and result.subject_value > limit.max_value:
            reasons.append(f"subject_value {result.subject_value} exceeds max_value {limit.max_value}")
        if limit.max_increase_pct is not None:
            if result.control_value is None:
                reasons.append("max_increase_pct declared but no control_value evidence supplied")
            elif result.control_value == 0:
                if result.subject_value > 0:
                    reasons.append(
                        f"control_value is 0 but subject_value is {result.subject_value} "
                        "-- unbounded relative growth"
                    )
            else:
                increase_pct = (
                    (result.subject_value - result.control_value) / result.control_value
                ) * 100.0
                if increase_pct > limit.max_increase_pct:
                    reasons.append(
                        f"measured increase {increase_pct:.2f}% exceeds max_increase_pct "
                        f"{limit.max_increase_pct}"
                    )
        if reasons:
            failed.append(limit.metric)
            detail[limit.metric] = "; ".join(reasons)
    passed = not missing and not failed
    return {
        "passed": passed,
        "required_limits": [limit.metric for limit in contract.acceptance.resource_limits],
        "missing_metrics": missing,
        "failed_metrics": failed,
        "detail": detail,
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
    from .. import generalise
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


# ------------------------------------------------------------ trigger proof (EC18)


@dataclass(frozen=True)
class TriggerEvidence:
    """One lane's proof that the contract's target code path was actually
    exercised during measurement, not merely that a benchmark process
    completed. Sourced from BigCherry's existing dispatch/coverage
    telemetry (0700 coverage_counters, 0810 replay_hit_diagnostics, 0820
    measurement_signature_shapes, 0830 split_reduce_telemetry) -- this
    module does not produce trigger evidence itself, only evaluates it,
    matching EC07's CorrectnessResult precedent (the gate consumes facts,
    it does not measure them).

    At least one of candidate_launches or expected_route_selected must be
    an int (the other may be None when not applicable to this lane's
    target kind, e.g. a kernel-dispatch contract has no "route" concept
    and a routing/split contract may have no per-launch counter) -- a lane
    supplying neither is a caller error (raised, not silently ignored),
    since that would make trigger-proof unfalsifiable."""
    role: str
    lane_id: str
    candidate_launches: int | None = None
    expected_route_selected: int | None = None

    def __post_init__(self) -> None:
        if self.candidate_launches is None and self.expected_route_selected is None:
            raise ExperimentContractError(
                f"trigger evidence for lane {self.lane_id!r} supplies neither "
                "candidate_launches nor expected_route_selected -- at least one "
                "is required, or this lane's trigger proof is unfalsifiable"
            )
        for field_name, value in (
            ("candidate_launches", self.candidate_launches),
            ("expected_route_selected", self.expected_route_selected),
        ):
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ExperimentContractError(
                    f"trigger evidence for lane {self.lane_id!r}: {field_name} "
                    f"must be a non-negative integer, got {value!r}"
                )


def evaluate_trigger_proof(trigger_evidence: list[TriggerEvidence]) -> dict[str, object]:
    """EC18: fail closed unless every POSITIVE-role lane proves its target
    code path actually ran at least once. Scoped to positive-role lanes
    only, mirroring aggregate_contract_effects()'s own precedent (EC06):
    a contract's performance claim is about what its positive lanes
    measured, so those are exactly the lanes whose measurement is
    meaningless if the candidate path never triggered. Control lanes are
    NOT checked here -- a control lane deliberately should not trigger the
    candidate in most contracts, so requiring launches there would invert
    the check; boundary lanes are exploring the edge of eligibility and
    may legitimately not trigger on one side of that edge, so they are
    excluded too.

    An empty trigger_evidence list is itself a failure -- a contract
    supplying no trigger evidence for its positive lanes has not
    demonstrated anything was exercised, which is exactly the case this
    gate exists to catch, not a vacuously-true absence of counter-
    examples."""
    positive = [e for e in trigger_evidence if e.role == "positive"]
    if not positive:
        return {
            "passed": False,
            "reasons": ["no positive-role trigger evidence supplied -- cannot "
                        "prove the target code path was ever exercised"],
            "checked_lanes": 0,
            "untriggered_lanes": [],
        }

    untriggered: list[str] = []
    for evidence in positive:
        counts = [c for c in (evidence.candidate_launches, evidence.expected_route_selected)
                  if c is not None]
        if not any(c > 0 for c in counts):
            untriggered.append(evidence.lane_id)

    reasons = []
    if untriggered:
        reasons.append(
            f"{len(untriggered)} of {len(positive)} positive-role lane(s) show zero "
            f"candidate launches and zero expected-route selections: {', '.join(untriggered)} "
            "-- the target code path was never exercised, so any measured effect for "
            "these lanes is not evidence of anything"
        )
    return {
        "passed": not untriggered,
        "reasons": reasons,
        "checked_lanes": len(positive),
        "untriggered_lanes": untriggered,
    }


# --------------------------------------------------------------- promotion (EC09)


def _finite_number(value: object) -> bool:
    """True only for a real, finite, non-boolean number.

    ``isinstance(x, (int, float))`` alone is not sufficient for a fail-closed
    gate, for two reasons found in review (dev-gpt-agent, req_cd86e5fd4a3b4328):

      * ``bool`` is a subclass of ``int``, so ``True`` would be accepted and
        compared as ``1``.
      * NaN and infinity pass the isinstance check, and NaN silently defeats
        BOTH threshold comparisons in evaluate_promotion_gate(), because every
        ordered comparison against NaN is False:

            float("nan") < required_gain   -> False   (so the gain check passes)
            float("nan") > regression_budget -> False (so the regression check passes)

        i.e. a NaN effect produced a PASS on both gain and regression. That is
        a confidently wrong verdict from malformed evidence, which is exactly
        what a fail-closed gate must never do.

    Malformed evidence must therefore be rejected here and reported as a
    failure reason, never silently satisfy a bound.
    """
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def evaluate_promotion_gate(
    contract: ExperimentContract, *, correctness_gate: dict[str, object],
    aggregated_effects: dict[str, object], generalisation_result: dict[str, object] | None = None,
    trigger_proof: dict[str, object] | None = None,
    resource_gate: dict[str, object] | None = None,
) -> dict[str, object]:
    """EC09: the final pass/fail/invalid verdict, combining EC18's trigger
    proof, EC07's correctness gate, EC06's aggregated effects against
    contract.acceptance's own thresholds, and (when supplied) EC08's
    generalisation proof. This is ADDITIONAL to every existing BigCherry
    promotion gate (promotion.py/tune_promotion.py's production-class
    provenance, decision-grade report, replay coverage, etc.) -- a
    contract-backed promotion must still clear those, this function does
    not replace or re-check them.

    trigger_proof (EC18) is checked FIRST and short-circuits every other
    check when it fails: a benchmark whose target code path never ran
    cannot be evidence of pass OR fail, so its verdict is "invalid", a
    third state distinct from both -- never silently folded into "fail"
    (which would look like a real negative result) or "pass" (which is
    the exact danger EC18 exists to prevent). trigger_proof is optional
    (None) only for backward compatibility with callers/contracts that
    predate EC18 and have not been updated to supply it yet; a caller that
    can supply it and doesn't is choosing not to use this gate, not
    proving anything.

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
    if trigger_proof is not None and not trigger_proof.get("passed"):
        return {
            "status": "invalid",
            "passed": False,
            "reasons": list(trigger_proof.get("reasons") or []),
            "contract_id": contract.id,
            "contract_hash": contract.contract_hash,
        }

    reasons: list[str] = []

    if not correctness_gate.get("passed"):
        missing = correctness_gate.get("missing_checks") or []
        failed = correctness_gate.get("failed_checks") or []
        reasons.append(
            f"correctness gate failed (missing={list(missing)}, failed={list(failed)})"
        )

    acceptance = contract.acceptance
    interval_policy = acceptance.effect_evidence_policy == "ci95_threshold_bound_v1"
    # VA24: evidence that is missing or malformed under an interval policy is
    # neither a pass nor a measured negative -- it is unevaluable, and must
    # surface as "invalid" exactly like an unproven trigger_proof does. These
    # are collected separately from `reasons` so a genuine below-threshold
    # result stays an ordinary "fail".
    invalid_reasons: list[str] = []

    def _gain_reasons(field: str, threshold: float) -> None:
        measured = aggregated_effects.get(field)
        if not interval_policy:
            if not _finite_number(measured) or measured < threshold:
                reasons.append(f"{field} {measured} below required {threshold}")
            return
        ci_low = aggregated_effects.get(f"{field}_ci95_low")
        rounds = aggregated_effects.get(f"{field}_paired_rounds")
        if not _finite_number(measured) or not _finite_number(ci_low):
            invalid_reasons.append(
                f"{field}: ci95_threshold_bound_v1 requires a finite point estimate "
                f"and ci95 lower bound (got {measured!r} / {ci_low!r})"
            )
            return
        if ci_low > measured:
            invalid_reasons.append(
                f"{field}: incoherent interval -- ci95_low {ci_low} exceeds the "
                f"point estimate {measured}"
            )
            return
        required_rounds = acceptance.min_paired_rounds
        if required_rounds is not None:
            if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < required_rounds:
                invalid_reasons.append(
                    f"{field}: {rounds!r} paired rounds is below the required "
                    f"minimum {required_rounds} -- the interval is not trustworthy"
                )
                return
        # The BOUND, not merely positivity, must be established by the
        # interval. "CI excludes zero" would only show "probably positive".
        if ci_low < threshold:
            reasons.append(
                f"{field} ci95_low {ci_low} below required {threshold} "
                f"(point estimate {measured})"
            )

    if acceptance.target_kernel_gain_pct is not None:
        _gain_reasons("target_kernel_gain_pct", acceptance.target_kernel_gain_pct)
    if acceptance.end_to_end_gain_pct is not None:
        _gain_reasons("end_to_end_gain_pct", acceptance.end_to_end_gain_pct)

    measured_regression = aggregated_effects.get("max_control_regression_pct")
    if not interval_policy:
        if (not _finite_number(measured_regression)
                or measured_regression > acceptance.max_control_regression_pct):
            reasons.append(
                f"max_control_regression_pct {measured_regression} exceeds budget "
                f"{acceptance.max_control_regression_pct}"
            )
    else:
        # VA24, regression side. The symmetric-looking rule "a regression only
        # fails when it is SIGNIFICANTLY negative" was explicitly rejected in
        # review (dev-gpt-agent, req_cd86e5fd4a3b4328) as fail-OPEN: it would
        # let an uncertain, genuinely-over-budget regression through simply
        # because the interval was wide.
        #
        # The requirement is the mirror of the gain rule: the budget must be
        # ESTABLISHED by the interval, i.e. the UPPER bound on the regression
        # must sit inside the budget. With a 1.0 budget:
        #     point -0.2, ci95_high 0.4  -> passes (noise absorbed)
        #     point -0.2, ci95_high 1.2  -> fails  (could really exceed 1.0)
        # This matters most for correctness-first contracts, where the
        # regression budget is the ONLY performance gate they have.
        ci_high = aggregated_effects.get("max_control_regression_pct_ci95_high")
        control_rounds = aggregated_effects.get("max_control_regression_pct_paired_rounds")
        required_rounds = acceptance.min_paired_rounds
        if not _finite_number(measured_regression) or not _finite_number(ci_high):
            invalid_reasons.append(
                f"max_control_regression_pct: ci95_threshold_bound_v1 requires a "
                f"finite point estimate and ci95 upper bound "
                f"(got {measured_regression!r} / {ci_high!r})"
            )
        # VA24 P0: the rounds floor applies to the CONTROL lane too. A
        # regression budget established from a single usable pair is exactly
        # as untrustworthy as a gain established from one, and leaving the
        # floor off here meant a contract could demand N rounds of evidence
        # for its claim while accepting n=1 evidence that it broke nothing.
        elif (required_rounds is not None
              and (not isinstance(control_rounds, int)
                   or isinstance(control_rounds, bool)
                   or control_rounds < required_rounds)):
            invalid_reasons.append(
                f"max_control_regression_pct: {control_rounds!r} paired rounds is "
                f"below the required minimum {required_rounds} -- the control "
                f"interval is not trustworthy"
            )
        elif ci_high < measured_regression:
            invalid_reasons.append(
                f"max_control_regression_pct: incoherent interval -- ci95_high "
                f"{ci_high} is below the point estimate {measured_regression}"
            )
        elif ci_high > acceptance.max_control_regression_pct:
            reasons.append(
                f"max_control_regression_pct ci95_high {ci_high} exceeds budget "
                f"{acceptance.max_control_regression_pct} "
                f"(point estimate {measured_regression})"
            )

    if generalisation_result is not None and not generalisation_result.get("passed"):
        reasons.append("generalisation proof did not pass")

    # VA12: a contract that declares resource limits at all must have them
    # checked -- missing or failing resource evidence blocks promotion,
    # exactly like a missing/failed correctness check. A contract that
    # declares NO resource limits is unaffected regardless of whether a
    # resource_gate happens to be supplied.
    if contract.acceptance.resource_limits:
        if resource_gate is None:
            reasons.append(
                "contract declares resource_limits but no resource_gate evidence was supplied"
            )
        elif not resource_gate.get("passed"):
            missing_r = resource_gate.get("missing_metrics") or []
            failed_r = resource_gate.get("failed_metrics") or []
            reasons.append(
                f"resource gate failed (missing={list(missing_r)}, failed={list(failed_r)})"
            )

    # VA24: unevaluable evidence is "invalid", never "fail" and never "pass" --
    # the same three-state distinction trigger_proof already establishes. A
    # missing/malformed/degenerate interval means we could not measure the
    # claim, which is not the same finding as having measured it and found it
    # wanting. Reported alongside any ordinary failures so a reader sees both.
    if invalid_reasons:
        return {
            "status": "invalid",
            "passed": False,
            "reasons": invalid_reasons + reasons,
            "contract_id": contract.id,
            "contract_hash": contract.contract_hash,
        }

    return {
        "status": "pass" if not reasons else "fail",
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
    resource_gate: dict[str, object] | None = None,
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

    if contract.source_evidence is not None:
        lines.append("## Source evidence (EC17 -- what the external source reported)")
        lines.append(f"- metric: {contract.source_evidence.metric}")
        lines.append(f"- value_pct: {contract.source_evidence.value_pct}")
        lines.append(f"- hardware: {contract.source_evidence.hardware}")
        lines.append(f"- workload: {contract.source_evidence.workload}")
        mismatch = source_evidence_mismatch_warning(contract)
        if mismatch is not None:
            lines.append(f"- WARNING: {mismatch}")
        lines.append("")

    lines.append("## Scope")
    lines.append(f"- backend: {contract.scope.backend}")
    lines.append(f"- architectures: {', '.join(contract.scope.architectures)}")
    if contract.scope.weight_types:
        lines.append(f"- weight_types: {', '.join(contract.scope.weight_types)}")
    for name in ("integrated", "uma", "peer_access"):
        value = getattr(contract.scope, name)
        if value is not None:
            lines.append(f"- {name}: {value}")
    if contract.scope.gpu_count is not None:
        lines.append(
            f"- gpu_count: {contract.scope.gpu_count.minimum or '*'}.."
            f"{contract.scope.gpu_count.maximum or '*'}"
        )
    if contract.scope.driver is not None:
        minimum = contract.scope.driver.minimum or "*"
        maximum = contract.scope.driver.maximum or "*"
        lines.append(f"- driver: {minimum}..{maximum}")
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

    if contract.acceptance.resource_limits:
        lines.append("## Resources")
        for limit in contract.acceptance.resource_limits:
            bound = []
            if limit.max_value is not None:
                bound.append(f"max_value={limit.max_value} {limit.unit}")
            if limit.max_increase_pct is not None:
                bound.append(f"max_increase_pct={limit.max_increase_pct}%")
            lines.append(f"- {limit.metric} ({', '.join(bound)})")
        if resource_gate is not None:
            lines.append(f"- gate passed: {resource_gate.get('passed')}")
            missing_r = resource_gate.get("missing_metrics") or []
            failed_r = resource_gate.get("failed_metrics") or []
            if missing_r:
                lines.append(f"- missing: {', '.join(missing_r)}")
            if failed_r:
                lines.append(f"- failed: {', '.join(failed_r)}")
        else:
            lines.append("- (no resource gate evidence supplied)")
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
    status = promotion_gate.get("status")
    if status is not None:
        lines.append(f"- status: {status}")
        if status == "invalid":
            lines.append("- INVALID: the target code path was not proven to have been "
                         "exercised (EC18 trigger proof) -- this is not a pass or a fail, "
                         "the measurement itself cannot be trusted as evidence")
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


# ------------------------------------------------- VA24 legacy migration lint


LEGACY_MANIFEST_PATH = "config/experiment-contract-legacy.toml"


@dataclass(frozen=True)
class LegacyWaiver:
    """One frozen pre-VA24 entry from config/experiment-contract-legacy.toml."""
    contract_id: str
    waiver_class: str
    baseline_contract_hash: str
    migration: str


def load_legacy_waivers(path: Path) -> dict[str, LegacyWaiver]:
    """Load the frozen legacy manifest. A missing file is an empty mapping --
    i.e. every gain-declaring contract must then use the interval policy --
    rather than an error, so the lint fails CLOSED if the manifest is lost."""
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    waivers: dict[str, LegacyWaiver] = {}
    for contract_id, entry in (raw.get("legacy") or {}).items():
        if not isinstance(entry, dict):
            raise ExperimentContractError(
                f"{path}: legacy.{contract_id} must be a table")
        for field in ("waiver_class", "baseline_contract_hash", "migration"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ExperimentContractError(
                    f"{path}: legacy.{contract_id}.{field} is required")
        waivers[contract_id] = LegacyWaiver(
            contract_id=contract_id, waiver_class=entry["waiver_class"],
            baseline_contract_hash=entry["baseline_contract_hash"],
            migration=entry["migration"],
        )
    return waivers


def declares_gain_threshold(contract: ExperimentContract) -> bool:
    return (contract.acceptance.target_kernel_gain_pct is not None
            or contract.acceptance.end_to_end_gain_pct is not None)


def lint_effect_evidence_policy(
    registry: ContractRegistry, waivers: dict[str, LegacyWaiver],
) -> list[str]:
    """VA24 REGISTRY LINT, keyed by contract ID (edit time).

    A contract declaring a gain threshold under the weaker
    ``point_estimate_v1`` is valid only if its ID appears in the frozen
    legacy manifest. New contracts must adopt ``ci95_threshold_bound_v1``.

    Deliberately keyed by ID, NOT by contract_hash. Hash keying was the
    first proposal and is wrong here: contract_hash covers rationale prose
    and source provenance, so a typo fix would evaporate the waiver and
    demand a fresh hardware qualification -- impossible for contracts
    targeting hardware this project does not own, which would make them
    permanently uneditable. Evidence identity and migration policy are
    different invariants; the exact-hash rule belongs at qualification time
    (see legacy_evidence_is_honoured) where old evidence must genuinely not
    satisfy an edited contract.

    Returns a list of human-readable problems; empty means clean.
    """
    problems: list[str] = []
    for contract in sorted(registry, key=lambda c: c.id):
        if not declares_gain_threshold(contract):
            continue
        if contract.acceptance.effect_evidence_policy != "point_estimate_v1":
            continue
        if contract.id not in waivers:
            problems.append(
                f"{contract.id}: declares a gain threshold under "
                f"point_estimate_v1 but is not in the frozen legacy manifest "
                f"({LEGACY_MANIFEST_PATH}) -- new performance contracts must "
                f"declare effect_evidence_policy = 'ci95_threshold_bound_v1' "
                f"(with a min_paired_rounds floor). The manifest is frozen: "
                f"adding an entry for a new contract is not the fix."
            )
    return problems


def legacy_evidence_is_honoured(
    contract: ExperimentContract, waivers: dict[str, LegacyWaiver],
) -> bool:
    """VA24 QUALIFICATION RULE, keyed by exact baseline_contract_hash.

    Legacy point-estimate EVIDENCE is honoured only for the exact contract
    version recorded in the manifest. Once any edit changes the hash, that
    version cannot acquire fresh legacy qualification and must migrate to
    the interval policy before it is next qualified.

    This is where exact-hash strictness genuinely belongs: it is the
    existing "old evidence must not silently satisfy a changed contract"
    invariant, not a restriction on editing the registry.
    """
    waiver = waivers.get(contract.id)
    return waiver is not None and waiver.baseline_contract_hash == contract.contract_hash
