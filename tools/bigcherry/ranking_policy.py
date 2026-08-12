"""Ranking-policy schema and prototype-policy registry (HI50).

The tuner (`hip-autotune-tuner.cu`) evaluates every installed ranking policy
live, at tune time, and records each one's full decision directly in each
result row's `ranking_decisions[]` -- see `select_latency_v1()` and the
compiled-in `g_policy_table` there. This module does not reimplement that
ranking: there is exactly one place the live algorithm exists, so there is no
risk of a Python mirror silently drifting from whatever ranking logic
actually shipped a given artifact.

What this module *does* own:

  * a typed, validated read of what the tuner already recorded
    (`CandidateMetrics`, `RankedCandidate`, `PolicyDecision`,
    `parse_ranking_decisions`) -- used by `rank_replay.py`'s report mode;
  * the schema and registry for a *prototype* policy: one not yet compiled
    into the C++ policy table, authored in Python so it can be replayed
    against archived finalist data (`rank_replay.py --policy-module`) before
    it is worth porting into `hip-autotune-tuner.cu`. This is the offline
    simulator HI44 and HI45 both require as a pre-hardware gate.

A prototype policy follows the same versioned-object idiom as
`generalise.py`'s generalisation policies: a plain JSON-able dict with a
`schema_version` and a `policy_hash` covering its own content, validated
before use rather than trusted by construction.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

from .tune_journal import canonical

POLICY_SCHEMA_VERSION = 1
PERSON = b"bc-rankpolicy-v1"

# Verdicts the tuner's PolicyCandidateVerdict enum (hip-autotune-tuner.cu)
# emits per candidate, per policy. Kept here as the one place both
# rank_replay.py and a prototype policy's own rank() should agree on the
# vocabulary, rather than each inventing its own strings.
VERDICTS = {
    "winner",
    "qualified",
    "near_tie_below_threshold",
    "outside_tie_band",
    "not_attempted",
    "rejected",
}


class RankingPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateMetrics:
    """One finalist's already-measured metrics, as recorded by the tuner.

    Built from a `candidates[]` entry in `*.measurements.jsonl` (or the
    equivalent SQLite `measurement` row). Deliberately excludes anything a
    policy shouldn't need to re-derive on its own -- raw sample arrays,
    reject-reason bookkeeping beyond `status` -- so a prototype policy's
    `rank()` signature stays small.
    """

    name: str
    is_native: bool
    status: str
    effective_us: float | None
    median_us: float | None
    p95_us: float | None
    host_median_us: float | None
    mad_us: float | None
    workspace_bytes: int
    nmse: float | None
    max_abs_err: float | None
    sign_p: float | None
    sign_wins: int
    sign_rounds: int

    @classmethod
    def from_json(cls, entry: dict[str, Any], native_name: str) -> "CandidateMetrics":
        name = entry.get("name", "")
        return cls(
            name=name,
            is_native=(name == native_name),
            status=entry.get("status", "ok"),
            effective_us=entry.get("effective_us"),
            median_us=entry.get("median_us"),
            p95_us=entry.get("p95_us"),
            host_median_us=entry.get("host_median_us"),
            mad_us=entry.get("mad_us"),
            workspace_bytes=entry.get("workspace", 0),
            nmse=entry.get("nmse"),
            max_abs_err=entry.get("max_abs"),
            sign_p=entry.get("sign_p"),
            sign_wins=entry.get("sign_wins", 0),
            sign_rounds=entry.get("sign_rounds", 0),
        )


@dataclass(frozen=True)
class RankedCandidate:
    """One candidate's outcome under one policy -- winner or not, and why."""

    name: str
    effective_us: float | None
    verdict: str
    rejection_reason: str

    @classmethod
    def from_json(cls, entry: dict[str, Any]) -> "RankedCandidate":
        return cls(
            name=entry.get("name", ""),
            effective_us=entry.get("effective_us"),
            verdict=entry.get("verdict", ""),
            rejection_reason=entry.get("rejection_reason", ""),
        )


@dataclass(frozen=True)
class PolicyDecision:
    """One policy's full decision for one dispatch: predicted_winner is the
    ranking-stage pick (before determinism recheck/confirmation holdout);
    `candidates` is every finalist that policy considered, winner and
    rejected alike -- what makes "what else was determined but rejected"
    inspectable instead of inferred.
    """

    policy_name: str
    policy_version: int
    is_production: bool
    predicted_winner: str
    candidates: list[RankedCandidate] = field(default_factory=list)

    @classmethod
    def from_json(cls, entry: dict[str, Any]) -> "PolicyDecision":
        return cls(
            policy_name=entry.get("policy_name", ""),
            policy_version=entry.get("policy_version", 0),
            is_production=bool(entry.get("is_production")),
            predicted_winner=entry.get("predicted_winner", ""),
            candidates=[RankedCandidate.from_json(c) for c in entry.get("candidates", [])],
        )


def parse_ranking_decisions(result: dict[str, Any]) -> list[PolicyDecision]:
    """Parse and lightly validate one result row's `ranking_decisions[]`.

    Returns an empty list for a measurements file that predates HI50 (the
    field is simply absent) -- callers should treat that as "no shadow data
    available", not an error; older artifacts are still valid for every
    other purpose.
    """
    raw = result.get("ranking_decisions")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise RankingPolicyError("ranking_decisions must be a list")
    decisions = [PolicyDecision.from_json(entry) for entry in raw]
    names = [d.policy_name for d in decisions]
    if len(names) != len(set(names)):
        raise RankingPolicyError("ranking_decisions has duplicate policy_name entries")
    production = [d for d in decisions if d.is_production]
    if len(production) > 1:
        raise RankingPolicyError("ranking_decisions has more than one production policy")
    return decisions


def finalist_metrics(result: dict[str, Any]) -> list[CandidateMetrics]:
    """Every finalist's metrics for one result row -- `schedule.candidates`
    restricted, excluding screening-only entries, matching what the tuner's
    own policies rank over. Native's own metrics are included (its
    `#twin`-suffixed repeatability check, if present, is not -- twins are
    excluded from ranking by every compiled-in policy and should be from a
    prototype policy too).
    """
    native_name = result.get("native", "")
    finalist_names = set(result.get("schedule", {}).get("candidates", []) or [])
    out = []
    for entry in result.get("candidates", []):
        name = entry.get("name", "")
        if name.endswith("#twin"):
            continue
        if name not in finalist_names:
            continue
        out.append(CandidateMetrics.from_json(entry, native_name))
    return out


def policy_hash(policy: dict[str, Any]) -> str:
    value = copy.deepcopy(policy)
    value.pop("policy_hash", None)
    return hashlib.blake2b(canonical(value), digest_size=16, person=PERSON).hexdigest()


def validate_policy_spec(policy: dict[str, Any]) -> dict[str, Any]:
    """Validate a *prototype* policy spec -- the SPEC a --policy-module
    exposes to rank_replay.py. Distinct from validating an already-recorded
    live decision (parse_ranking_decisions above): this is for a policy that
    has never run on hardware.
    """
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise RankingPolicyError("rerun_or_external_conversion_required: ranking policy spec")
    if not isinstance(policy.get("name"), str) or not policy["name"]:
        raise RankingPolicyError("ranking policy spec needs a name")
    if not isinstance(policy.get("version"), int):
        raise RankingPolicyError("ranking policy spec needs an integer version")
    if policy.get("policy_hash") != policy_hash(policy):
        raise RankingPolicyError("policy content hash mismatch")
    return policy


class PrototypePolicy(Protocol):
    """The interface a `--policy-module` passed to `rank_replay.py` must
    implement: a module-level `SPEC` dict (validated by
    validate_policy_spec) and a `rank()` function with this signature.
    """

    SPEC: dict[str, Any]

    def rank(
        self,
        native: CandidateMetrics,
        candidates: list[CandidateMetrics],
        config: dict[str, Any],
    ) -> PolicyDecision:
        ...


# Importable module paths for known prototype policies, keyed by name --
# rank_replay.py's --policy-module flag can name an entry here instead of a
# raw path. Adding an entry here does not compile a policy into the live
# tuner -- that is a separate step in hip-autotune-tuner.cu's
# g_policy_table, taken only after this module's offline gates pass.
# NOTE: the four pareto_policy_*.py modules below are not currently present
# in this tree (blocked on pareto_report.py, never restored -- see HI47's
# notes); an entry here that is never invoked costs nothing, but invoking one
# of these four names will fail with an ImportError until that lands.
PROTOTYPE_POLICIES: dict[str, str] = {
    "pareto-low-memory-v1": "bigcherry.pareto_policy_low_memory",
    "pareto-conservative-stability-v1": "bigcherry.pareto_policy_conservative_stability",
    "pareto-balanced-v1": "bigcherry.pareto_policy_balanced",
    "pareto-max-throughput-v1": "bigcherry.pareto_policy_max_throughput",
}
