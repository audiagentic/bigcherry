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

from .journal import canonical

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
    def from_json(cls, entry: Any) -> "RankedCandidate":
        """Strict-typed parse. gpt-dev-agent review round 2, 2026-08-31:
        PolicyDecision.from_json validates ITS OWN fields strictly but
        used to hand list elements straight to this constructor with no
        type check at all -- a malformed candidate entry (not a dict, or
        with a non-string name/verdict) raised AttributeError/TypeError
        from deep inside dict.get() instead of a clean RankingPolicyError,
        breaking the "every malformed ranking record fails through
        RankingPolicyError" contract tune_promotion.py's error handling
        relies on.
        """
        if not isinstance(entry, dict):
            raise RankingPolicyError(f"ranking candidate entry must be a dict, got {type(entry).__name__}")
        name = entry.get("name", "")
        if not isinstance(name, str) or not name:
            raise RankingPolicyError("ranking candidate entry: name must be a non-empty string")
        effective_us = entry.get("effective_us")
        if effective_us is not None and (
            isinstance(effective_us, bool) or not isinstance(effective_us, (int, float))
        ):
            raise RankingPolicyError("ranking candidate entry: effective_us must be numeric or null")
        verdict = entry.get("verdict", "")
        if not isinstance(verdict, str):
            raise RankingPolicyError("ranking candidate entry: verdict must be a string")
        rejection_reason = entry.get("rejection_reason", "")
        if not isinstance(rejection_reason, str):
            raise RankingPolicyError("ranking candidate entry: rejection_reason must be a string")
        return cls(
            name=name,
            effective_us=float(effective_us) if effective_us is not None else None,
            verdict=verdict,
            rejection_reason=rejection_reason,
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
        """Strict-typed parse -- this feeds tune_promotion.py's production-
        policy identity check directly, so a type-coerced field here is a
        real correctness-boundary bug, not just a display bug (gpt-dev-agent
        review, 2026-08-31). ``bool(entry.get("is_production"))`` used to
        coerce the STRING ``"false"`` to Python ``True`` (any non-empty
        string is truthy) -- a malformed decision record with
        ``"is_production": "false"`` would count as a real production
        decision in ``parse_ranking_decisions``'s production-count check and
        in ``_validate_policy_identity``'s coverage check. Likewise
        ``policy_version`` had no type check at all, so a boolean ``true``
        would satisfy ``prod.policy_version != row_policy["version"]`` for
        any ``row_policy["version"] == 1`` (Python: ``True == 1``).
        """
        policy_name = entry.get("policy_name", "")
        if not isinstance(policy_name, str) or not policy_name:
            raise RankingPolicyError("ranking_decisions entry: policy_name must be a non-empty string")
        policy_version = entry.get("policy_version", 0)
        if isinstance(policy_version, bool) or not isinstance(policy_version, int):
            raise RankingPolicyError("ranking_decisions entry: policy_version must be an integer")
        is_production = entry.get("is_production", False)
        if not isinstance(is_production, bool):
            raise RankingPolicyError("ranking_decisions entry: is_production must be a boolean")
        predicted_winner = entry.get("predicted_winner", "")
        if not isinstance(predicted_winner, str):
            raise RankingPolicyError("ranking_decisions entry: predicted_winner must be a string")
        candidates_raw = entry.get("candidates", [])
        if not isinstance(candidates_raw, list):
            raise RankingPolicyError("ranking_decisions entry: candidates must be a list")
        return cls(
            policy_name=policy_name,
            policy_version=policy_version,
            is_production=is_production,
            predicted_winner=predicted_winner,
            candidates=[RankedCandidate.from_json(c) for c in candidates_raw],
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
