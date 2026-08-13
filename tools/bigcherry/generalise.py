"""Current-schema generalized-key policy validation and holdout proof (HI36)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .tune_journal import atomic_write, canonical

SCHEMA_VERSION = 1
PERSON = b"bc-general-v1"
REQUIRED_THRESHOLDS = {
    "min_exact_signatures": 3,
    "min_holdout_calls": 100,
    "min_added_coverage_pct": 5.0,
    "max_median_regret_pct": 0.5,
    "max_upper_regret_pct": 1.0,
    "max_exact_regression_pct": 0.5,
}
FIELDS = {
    "alignment_class", "dst_type", "flags", "fusion", "glu_op",
    "n_expert", "n_expert_used", "nb0", "nb1", "nbd", "ne0", "ne1",
    "ned", "occupancy_bucket", "offset_modulo", "op", "prec",
    "schema_version", "src0_type", "src1_type",
}


class GeneralisationError(RuntimeError):
    pass


_DIGEST = re.compile(r"^[0-9a-f]{32}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_WINNER_REQUIRED = {
    "source_signature", "coverage_delta", "manifest_identity",
    "evidence_references",
}


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value.lower()):
        raise GeneralisationError(f"generalized winner has invalid {name}")
    return value.lower()


def _require_percent(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeneralisationError(f"generalized winner has invalid {name}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise GeneralisationError(f"generalized winner has invalid {name}")
    return value


def validate_generalised_winner(winner: dict[str, Any]) -> dict[str, Any]:
    """Validate the portable evidence attached to one generalized winner.

    This is intentionally independent of replay-cache publication.  A caller
    may construct or inspect a generalized winner offline, but it cannot call
    it complete evidence unless the exact source and coverage claim can be
    traced to a build and to durable evidence artifacts.
    """
    if not isinstance(winner, dict) or winner.get("kind") != "generalised_winner":
        raise GeneralisationError("generalized winner record kind is required")
    missing = _WINNER_REQUIRED - winner.keys()
    if missing:
        raise GeneralisationError(
            "generalized winner provenance is incomplete: " + ", ".join(sorted(missing))
        )
    _require_digest(winner["source_signature"], "source_signature")

    delta = winner["coverage_delta"]
    if not isinstance(delta, dict):
        raise GeneralisationError("generalized winner coverage_delta must be an object")
    for field in ("baseline_coverage_pct", "generalised_coverage_pct", "added_coverage_pct"):
        if field not in delta:
            raise GeneralisationError(f"generalized winner coverage_delta lacks {field}")
        _require_percent(delta[field], f"coverage_delta.{field}")
    if delta["generalised_coverage_pct"] < delta["baseline_coverage_pct"]:
        raise GeneralisationError("generalized winner coverage delta goes backwards")
    expected = delta["generalised_coverage_pct"] - delta["baseline_coverage_pct"]
    if not math.isclose(float(delta["added_coverage_pct"]), expected, abs_tol=1e-9):
        raise GeneralisationError("generalized winner coverage delta is inconsistent")
    if isinstance(delta.get("added_calls", 0), bool) or not isinstance(delta.get("added_calls", 0), int) or delta.get("added_calls", 0) < 0:
        raise GeneralisationError("generalized winner has invalid coverage_delta.added_calls")

    identity = winner["manifest_identity"]
    if not isinstance(identity, dict):
        raise GeneralisationError("generalized winner manifest_identity must be an object")
    if not isinstance(identity.get("source_revision"), str) or not _REVISION.fullmatch(identity["source_revision"].lower()):
        raise GeneralisationError("generalized winner has invalid source_revision")
    _require_digest(identity.get("manifest_hash"), "manifest_hash")
    _require_digest(identity.get("build_descriptor_hash"), "build_descriptor_hash")

    refs = winner["evidence_references"]
    if not isinstance(refs, list) or not refs:
        raise GeneralisationError("generalized winner requires evidence_references")
    seen: set[str] = set()
    for ref in refs:
        if not isinstance(ref, dict) or not isinstance(ref.get("kind"), str) or not isinstance(ref.get("ref"), str):
            raise GeneralisationError("generalized winner evidence reference is malformed")
        key = ref["kind"] + "\0" + ref["ref"]
        if not ref["kind"].strip() or not ref["ref"].strip() or key in seen:
            raise GeneralisationError("generalized winner evidence references must be unique and non-empty")
        seen.add(key)
    return winner


def build_generalised_winner(
    *, source_signature: str, coverage_delta: dict[str, Any],
    source_revision: str, manifest_hash: str, build_descriptor_hash: str,
    evidence_references: list[dict[str, str]], **winner: Any,
) -> dict[str, Any]:
    """Construct and validate a generalized winner evidence record."""
    record = dict(winner)
    record.update(
        kind="generalised_winner", source_signature=source_signature,
        coverage_delta=copy.deepcopy(coverage_delta),
        manifest_identity={
            "source_revision": source_revision,
            "manifest_hash": manifest_hash,
            "build_descriptor_hash": build_descriptor_hash,
        }, evidence_references=copy.deepcopy(evidence_references),
    )
    return validate_generalised_winner(record)


def policy_hash(policy: dict[str, Any]) -> str:
    value = copy.deepcopy(policy)
    value.pop("policy_hash", None)
    return hashlib.blake2b(canonical(value), digest_size=16, person=PERSON).hexdigest()


def validate_policy(policy: dict[str, Any], *, require_promoted: bool = False) -> dict[str, Any]:
    if policy.get("schema_version") != SCHEMA_VERSION or policy.get("key_version") != 1:
        raise GeneralisationError("rerun_or_external_conversion_required: generalisation policy")
    if policy.get("family") not in {"mmq", "mmvq", "mmvf", "mmf", "blas"}:
        raise GeneralisationError("unknown kernel family")
    if policy.get("eligibility_predicate_version") != 1:
        raise GeneralisationError("unknown eligibility predicate")
    included = policy.get("included_fields")
    excluded = policy.get("excluded_fields")
    if (not isinstance(included, list) or not isinstance(excluded, list) or
            set(included) | set(excluded) != FIELDS or set(included) & set(excluded)):
        raise GeneralisationError("policy fields must be a complete disjoint partition")
    for transform in policy.get("transforms", []):
        if (transform.get("op") not in {"zero", "bucket_pow2"} or
                transform.get("field") not in FIELDS):
            raise GeneralisationError("unknown policy transform")
        if transform.get("field") in included:
            raise GeneralisationError("transformed field cannot also be included verbatim")
    thresholds = policy.get("thresholds")
    if not isinstance(thresholds, dict) or any(name not in thresholds for name in REQUIRED_THRESHOLDS):
        raise GeneralisationError("predeclared thresholds are incomplete")
    if policy.get("policy_hash") != policy_hash(policy):
        raise GeneralisationError("policy content hash mismatch")
    if require_promoted and policy.get("status") != "promoted":
        raise GeneralisationError("generalized lookup policy lacks holdout promotion")
    return policy


def _transform_value(value: Any, transform: dict[str, Any]) -> Any:
    op = transform["op"]
    indices = transform.get("indices")
    if indices is not None:
        if not isinstance(value, list):
            raise GeneralisationError("indexed transform requires an array field")
        output = list(value)
        for index in indices:
            if not isinstance(index, int) or not 0 <= index < len(output):
                raise GeneralisationError("policy transform index is invalid")
            output[index] = 0 if op == "zero" else (1 << (max(1, int(output[index])) - 1).bit_length())
        return output
    if op == "zero":
        return [0] * len(value) if isinstance(value, list) else 0
    number = max(1, int(value))
    return 1 << (number - 1).bit_length()


def generalised_canonical(signature: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    validate_policy(policy)
    missing = FIELDS - set(signature)
    if missing:
        raise GeneralisationError("signature lacks current fields: " + ", ".join(sorted(missing)))
    output = {field: copy.deepcopy(signature[field]) for field in policy["included_fields"]}
    for transform in policy.get("transforms", []):
        output[transform["field"]] = _transform_value(signature[transform["field"]], transform)
    output["generalisation_key_version"] = policy["key_version"]
    output["generalisation_policy_hash"] = policy["policy_hash"]
    return output


def generalised_digest(signature: dict[str, Any], policy: dict[str, Any]) -> str:
    return hashlib.blake2b(
        canonical(generalised_canonical(signature, policy)),
        digest_size=16, person=PERSON,
    ).hexdigest()


def _candidate_time(row: dict[str, Any], name: str) -> float | None:
    for candidate in row.get("candidates", []):
        if candidate.get("name") == name and candidate.get("status") == "ok":
            value = candidate.get("effective_us", candidate.get("median_us"))
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return None


def prove(policy: dict[str, Any], results: list[dict[str, Any]],
          holdout_calls: dict[str, int], *, exact_regression_pct: float = 0.0) -> dict[str, Any]:
    """Prove a policy from exact results plus independently observed call weights."""
    validate_policy(policy)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        if row.get("kind") != "result" or not isinstance(row.get("canonical"), dict):
            raise GeneralisationError("current result/canonical signature required")
        groups[generalised_digest(row["canonical"], policy)].append(row)

    regrets: list[tuple[float, int]] = []
    proven_groups = 0
    converted_calls = 0
    correctness_failures = 0
    minimum = int(policy["thresholds"]["min_exact_signatures"])
    for members in groups.values():
        if len(members) < minimum:
            continue
        # Call-weighted representative; it must have timing evidence in every
        # exact member, otherwise the group is not safe to export.
        representative = max(
            members, key=lambda row: holdout_calls.get(str(row.get("signature")), 0)
        ).get("winner")
        group_regrets: list[tuple[float, int]] = []
        valid = True
        for row in members:
            if row.get("promotion_blocked") or row.get("canary_state") == "unresolved":
                valid = False
                break
            exact = _candidate_time(row, str(row.get("winner")))
            generalized = _candidate_time(row, str(representative))
            if exact is None or generalized is None:
                valid = False
                break
            calls = int(holdout_calls.get(str(row.get("signature")), 0))
            group_regrets.append((100.0 * (generalized - exact) / exact, calls))
        if not valid:
            correctness_failures += 1
            continue
        proven_groups += 1
        regrets.extend(group_regrets)
        converted_calls += sum(calls for _, calls in group_regrets)

    total_calls = sum(max(0, int(value)) for value in holdout_calls.values())
    ordered = sorted(regrets, key=lambda item: item[0])
    weighted_total = sum(weight for _, weight in ordered)
    middle = weighted_total / 2
    cumulative = 0
    median_regret = math.inf
    for regret, weight in ordered:
        cumulative += weight
        if cumulative >= middle:
            median_regret = regret
            break
    if ordered and weighted_total > 0:
        bootstrap_seed = int(policy["policy_hash"][:16], 16)
        rng = random.Random(bootstrap_seed)
        values = [item[0] for item in ordered]
        weights = [item[1] for item in ordered]
        bootstrap = sorted(
            statistics_mean(rng.choices(values, weights=weights, k=len(values)))
            for _ in range(10_000)
        )
        upper_regret = bootstrap[9750]
    else:
        bootstrap_seed = 0
        upper_regret = math.inf
    coverage = 100.0 * converted_calls / total_calls if total_calls else 0.0
    thresholds = policy["thresholds"]
    passed = (
        proven_groups > 0 and converted_calls >= thresholds["min_holdout_calls"] and
        coverage >= thresholds["min_added_coverage_pct"] and correctness_failures == 0 and
        median_regret <= thresholds["max_median_regret_pct"] and
        upper_regret <= thresholds["max_upper_regret_pct"] and
        exact_regression_pct <= thresholds["max_exact_regression_pct"]
    )
    return {
        "proven_groups": proven_groups, "holdout_calls": total_calls,
        "converted_calls": converted_calls, "added_coverage_pct": coverage,
        "median_regret_pct": median_regret, "upper95_regret_pct": upper_regret,
        "regret_bootstrap_resamples": 10_000, "regret_bootstrap_seed": bootstrap_seed,
        "exact_regression_pct": exact_regression_pct,
        "correctness_failures": correctness_failures, "passed": passed,
    }


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values)


# --------------------------------------------------------------------------
# HI36 steps 1 and 3: offline "which grouping should we even propose" analysis.
#
# This is deliberately separate from prove()/validate_policy() above, which
# score a policy someone has already committed to. regret_table() answers the
# prior question -- for a *proposed* grouping, what would generalisation cost,
# computed entirely from medians already on disk, with no GPU and no policy
# object required. The criterion is regret, not winner agreement (HI36's own
# governing rule): a group whose winners alternate 0.3% apart is a good
# generalisation and a bad unanimity score.
# --------------------------------------------------------------------------

# HI36 detailed_solution's field mapping: named dimensions onto the raw
# canonical signature. K/M/ncols_dst are not signature fields themselves --
# they are positions inside the ne0/ned extent arrays.
_NAMED_FIELD_PATHS = {
    "K": ("ne0", 0),
    "M": ("ne0", 1),
    "ncols_dst": ("ned", 1),
}


def _grouping_value(canonical: dict[str, Any], field: str) -> Any:
    """Resolve one --group-by field name against a canonical signature dict.

    Plain FIELDS members (src0_type, prec, ...) are read directly; K/M/
    ncols_dst are derived via _NAMED_FIELD_PATHS, since they are extent-array
    positions, not top-level fields.
    """
    if field in _NAMED_FIELD_PATHS:
        array_name, index = _NAMED_FIELD_PATHS[field]
        array = canonical.get(array_name)
        if not isinstance(array, list) or index >= len(array):
            return None
        return array[index]
    return canonical.get(field)


def _row_family(row: dict[str, Any]) -> str | None:
    """Family from the native candidate's stable name, e.g. 'mmq:q8_0:...' -> 'mmq'.

    New-format result rows name the native candidate in "native"; a row
    without one (old-format, not yet joined to its record) cannot be grouped
    by family and is reported rather than silently dropped.
    """
    native = row.get("native")
    if isinstance(native, str) and ":" in native:
        return native.split(":", 1)[0]
    return None


def group_key(row: dict[str, Any], group_by: list[str]) -> tuple[Any, ...] | None:
    """The proposed group key for one result row, or None if unresolvable.

    None (rather than raising) is deliberate: an unresolvable row is a
    finding about the input data, not a bug, and regret_table() reports it
    as such instead of aborting the whole analysis.
    """
    canonical = row.get("canonical")
    if not isinstance(canonical, dict):
        return None
    values = []
    for field in group_by:
        if field == "family":
            value = _row_family(row)
        else:
            value = _grouping_value(canonical, field)
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def _candidate_status(row: dict[str, Any], name: str) -> dict[str, Any] | None:
    for candidate in row.get("candidates", []):
        if candidate.get("name") == name and candidate.get("status") == "ok":
            value = candidate.get("median_us")
            if isinstance(value, (int, float)) and value > 0:
                return candidate
    return None


def _native_median(row: dict[str, Any]) -> float:
    native = row.get("native")
    if isinstance(native, str):
        candidate = _candidate_status(row, native)
        if candidate is not None:
            return float(candidate["median_us"])
    winner = _candidate_status(row, str(row.get("winner", "")))
    return float(winner["median_us"]) if winner else 0.0


def group_representatives(
    results: list[dict[str, Any]], calls: dict[str, int], group_by: list[str],
) -> dict[tuple[Any, ...], str]:
    """One representative winner per group: the winner of its highest call-weighted member.

    This mirrors what an exporter ends up storing -- one entry per group,
    picked from whichever signature in the group carries the most weight.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        key = group_key(row, group_by)
        if key is not None:
            groups[key].append(row)

    representatives: dict[tuple[Any, ...], str] = {}
    for key, members in groups.items():
        representative = max(
            members,
            key=lambda row: calls.get(str(row.get("signature")), 0) * _native_median(row),
        )
        representatives[key] = str(representative.get("winner", ""))
    return representatives


def regret_table(
    results: list[dict[str, Any]], calls: dict[str, int], group_by: list[str],
) -> list[dict[str, Any]]:
    """Cost of serving every signature in a proposed group with one winner.

    Computable entirely offline from data already on disk: every candidate's
    median for every signature is already in the measurements file, so any
    proposed grouping can be priced without running anything.

    The criterion is regret, not agreement (HI36's governing rule): two
    winners alternating within a group at 0.3% apart is a *good*
    generalisation, and requiring identical winners would reject it and most
    of the value with it.
    """
    representatives = group_representatives(results, calls, group_by)
    rows: list[dict[str, Any]] = []
    for row in results:
        key = group_key(row, group_by)
        if key is None:
            continue
        representative = representatives[key]
        exact = _candidate_status(row, str(row.get("winner", "")))
        if exact is None:
            continue  # rejected signature: no claim to make
        generalised = _candidate_status(row, representative)
        signature = row.get("signature")
        entry = {"key": key, "signature": signature, "calls": int(calls.get(str(signature), 0))}
        if generalised is None:
            # The representative was not even eligible here -- the finding is
            # that the grouping is too coarse for this family, not a bug.
            entry.update(regret_pct=None, status="ineligible")
        else:
            entry.update(
                regret_pct=100.0 * (float(generalised["median_us"]) - float(exact["median_us"]))
                / float(exact["median_us"]),
                status="ok",
            )
        rows.append(entry)
    return rows


def _weighted_median(pairs: list[tuple[float, int]]) -> float:
    ordered = sorted(pairs, key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return math.inf
    middle = total / 2
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= middle:
            return value
    return ordered[-1][0]


def summarise_regret(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a regret_table() output: unweighted spread plus call-weighted median.

    Both numbers are reported deliberately (HI36 detailed_solution): the
    unweighted distribution bounds the worst case, the call-weighted median
    is the expected cost, and averaging a 15%-regret-at-52-calls signature
    with a 0.1%-regret-at-110,160-calls one unweighted would misstate both.
    """
    ok = [row for row in rows if row["status"] == "ok"]
    ineligible = sum(1 for row in rows if row["status"] == "ineligible")
    if not ok:
        return {
            "signatures": len(rows), "ok": 0, "ineligible": ineligible,
            "median_regret_pct": None, "p95_regret_pct": None,
            "worst_regret_pct": None, "call_weighted_median_regret_pct": None,
        }
    unweighted = sorted(row["regret_pct"] for row in ok)

    def percentile(values: list[float], pct: float) -> float:
        index = min(len(values) - 1, max(0, int(round(pct / 100.0 * (len(values) - 1)))))
        return values[index]

    return {
        "signatures": len(rows), "ok": len(ok), "ineligible": ineligible,
        "median_regret_pct": percentile(unweighted, 50.0),
        "p95_regret_pct": percentile(unweighted, 95.0),
        "worst_regret_pct": unweighted[-1],
        "call_weighted_median_regret_pct": _weighted_median(
            [(row["regret_pct"], row["calls"]) for row in ok]
        ),
    }


def what_if(
    miss_rows: list[dict[str, Any]], results: list[dict[str, Any]],
    calls: dict[str, int], group_by: list[str],
) -> dict[str, Any]:
    """Replay real miss-log entries against a proposed grouping (HI36 step 3).

    This is the item's own go/no-go gate: "if it converts fewer than half,
    the item should stop there." Only counts a conversion when the proposed
    group has a tuned representative *and* the miss row's own candidate list
    shows that representative eligible -- can_execute is not re-derived
    offline, so "ineligible" here is inferred from measured-set absence and
    is confirmed for real only by the runtime guard at step 4.
    """
    representatives = group_representatives(results, calls, group_by)
    converted = no_member_tuned = ineligible_here = 0
    conversions: list[float] = []
    for miss in miss_rows:
        key = group_key(miss, group_by)
        representative = representatives.get(key) if key is not None else None
        if representative is None:
            no_member_tuned += 1
            continue
        candidate = _candidate_status(miss, representative)
        fallback = _candidate_status(miss, str(miss.get("fallback", "")))
        if candidate is None or fallback is None:
            ineligible_here += 1
            continue
        converted += 1
        conversions.append(
            100.0 * (float(candidate["median_us"]) - float(fallback["median_us"]))
            / float(fallback["median_us"])
        )
    total = len(miss_rows)
    return {
        "misses": total, "converted": converted, "no_member_tuned": no_member_tuned,
        "ineligible_here": ineligible_here,
        "median_regret_pct": statistics.median(conversions) if conversions else None,
        "converted_fraction": converted / total if total else 0.0,
    }


def _main_analyse(argv: list[str]) -> int:
    """`bigcherry generalise analyse ...` -- HI36 steps 1/3 offline regret analysis.

    Deliberately a separate leading-token dispatch rather than argparse
    subparsers, so the existing `bigcherry generalise <policy> --measurements
    ... --output ...` invocation wired in __main__.py (and documented in
    HI36) keeps working unchanged -- this command only ever wins when the
    first token is literally "analyse".
    """
    parser = argparse.ArgumentParser(prog="bigcherry generalise analyse")
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--holdout-calls", type=Path)
    parser.add_argument(
        "--group-by", type=str, required=True,
        help="comma-separated field list, e.g. family,src0_type,K,ncols_dst",
    )
    parser.add_argument("--what-if", type=Path, dest="miss_log", default=None,
                        help="replay a miss log (JSONL with 'canonical' and 'fallback') against the grouping")
    args = parser.parse_args(argv)

    group_by = [field.strip() for field in args.group_by.split(",") if field.strip()]
    rows = [json.loads(line) for line in args.measurements.read_text(encoding="utf-8").splitlines()]
    results = [row for row in rows if row.get("kind") == "result"]
    calls = json.loads(args.holdout_calls.read_text(encoding="utf-8")) if args.holdout_calls else {}
    if args.miss_log is not None:
        misses = [json.loads(line) for line in args.miss_log.read_text(encoding="utf-8").splitlines()]
        summary = what_if(misses, results, calls, group_by)
    else:
        summary = summarise_regret(regret_table(results, calls, group_by))
    print(json.dumps(summary, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is not None and argv and argv[0] == "analyse":
        return _main_analyse(argv[1:])

    parser = argparse.ArgumentParser(prog="bigcherry generalise")
    parser.add_argument("policy", type=Path)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--holdout-calls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        validate_policy(policy)
        rows = [json.loads(line) for line in args.measurements.read_text(encoding="utf-8").splitlines()]
        results = [row for row in rows if row.get("kind") == "result"]
        calls = json.loads(args.holdout_calls.read_text(encoding="utf-8"))
        evidence = prove(policy, results, calls)
        output = copy.deepcopy(policy)
        output["evidence"] = evidence
        output["status"] = "promoted" if evidence["passed"] else "rejected"
        output["policy_hash"] = policy_hash(output)
        atomic_write(args.output, json.dumps(output, sort_keys=True, indent=2).encode("ascii") + b"\n")
    except (OSError, ValueError, GeneralisationError) as exc:
        print(f"invalid: {exc}")
        return 1
    print(json.dumps(evidence, sort_keys=True))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
