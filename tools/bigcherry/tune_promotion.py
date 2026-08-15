"""Current-only experiment-wide promotion for fresh confirmation evidence (HI34).

Recovered from commit 2c2fe7c (external git-reset incident, 2026-08-11/12) and
reconciled against current tuner output. One change from the original: the
tuner does not yet emit ``variant_set``/``hardware_key``/``config`` in the
measurements header (HI37, provenance capture, tracked separately and still
pending) -- ``required_header`` below checks only the fields the tuner
actually emits today.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from .tune_journal import atomic_write, canonical
from .identity_separation import IdentitySeparationError, validate_measurement_identity
from . import ranking_policy

SCHEMA_VERSION = 1
MIN_PAIRED_ROUNDS = 8


class PromotionError(RuntimeError):
    pass


def production_policy_hash(name: str, version: int) -> str:
    """Return the stable identity hash for the production ranking policy."""
    spec = {
        "schema_version": ranking_policy.POLICY_SCHEMA_VERSION,
        "name": name,
        "version": version,
    }
    return ranking_policy.policy_hash(spec)


def _validate_policy_identity(row: dict[str, Any], header: dict[str, Any]) -> None:
    """Validate policy identity and ranking coverage when HI50 data is present.

    Pre-HI50 artifacts omit these fields and remain valid.  Once an artifact
    claims ranking evidence, every decision must be attributable to the same
    production policy and cover every non-twin scheduled finalist.
    """
    row_policy = row.get("production_policy")
    header_policy = header.get("production_policy")
    if row_policy is not None:
        if (not isinstance(row_policy, dict) or
                not isinstance(row_policy.get("name"), str) or
                not row_policy["name"] or
                isinstance(row_policy.get("version"), bool) or
                not isinstance(row_policy.get("version"), int) or
                row_policy["version"] < 1):
            raise PromotionError("invalid production policy identity")
        if not isinstance(header_policy, str) or header_policy != row_policy["name"]:
            raise PromotionError("production policy identity does not match header")
        expected_hash = production_policy_hash(row_policy["name"], row_policy["version"])
        supplied_hash = row_policy.get("policy_hash")
        if supplied_hash is not None and supplied_hash != expected_hash:
            raise PromotionError("production policy hash mismatch")

    if "ranking_decisions" not in row:
        return
    try:
        decisions = ranking_policy.parse_ranking_decisions(row)
    except ranking_policy.RankingPolicyError as exc:
        raise PromotionError(str(exc)) from exc
    if not decisions:
        raise PromotionError("ranking decision coverage is empty")
    production = [decision for decision in decisions if decision.is_production]
    if len(production) != 1:
        raise PromotionError("ranking decision production policy coverage is invalid")
    if row_policy is None:
        raise PromotionError("ranking decision production policy identity is missing")
    prod = production[0]
    if (prod.policy_name != row_policy["name"] or
            prod.policy_version != row_policy["version"]):
        raise PromotionError("ranking decision production policy identity does not match")
    expected_names = {
        name for name in row.get("schedule", {}).get("candidates", [])
        if isinstance(name, str) and not name.endswith("#twin")
    }
    if not expected_names:
        # Native-only results have no selection schedule: there was no
        # challenger eligible for the signature.  They still carry a
        # one-candidate ranking decision for provenance, so validate that
        # decision against the native identity instead of treating the absent
        # schedule as malformed evidence.
        native = row.get("native")
        if (not isinstance(native, str) or
                row.get("provisional_winner") != native):
            raise PromotionError("ranking decision coverage has no scheduled finalists")
        expected_names = {native}
    for decision in decisions:
        names = [candidate.name for candidate in decision.candidates]
        if set(names) != expected_names or len(names) != len(set(names)):
            raise PromotionError("ranking decision candidate coverage is incomplete")
        if decision.predicted_winner not in expected_names:
            raise PromotionError("ranking decision winner is outside candidate coverage")
        if any(candidate.verdict not in ranking_policy.VERDICTS
               for candidate in decision.candidates):
            raise PromotionError("ranking decision has unknown candidate verdict")
        winners = [candidate.name for candidate in decision.candidates
                   if candidate.verdict == "winner"]
        if winners != [decision.predicted_winner]:
            raise PromotionError("ranking decision winner verdict is inconsistent")
    if prod.predicted_winner != row.get("provisional_winner"):
        raise PromotionError("production ranking decision does not match provisional winner")


def _validate_provisional_status(row: dict[str, Any]) -> None:
    provisional = row.get("provisional_winner")
    status = row.get("promotion_status")
    native = row.get("native")
    if not isinstance(provisional, str) or not provisional:
        raise PromotionError("provisional winner identity is missing")
    # Keep the legacy promotion-stage diagnostic for artifacts that predate
    # the persisted status field; the caller applies the current-state gate.
    if status is None:
        return
    if status not in {"native", "pending_bh", "confirmation_rejected", "promoted",
                      "rejected_effect", "rejected_ci", "rejected_bh"}:
        raise PromotionError("unknown promotion status")
    if provisional == native and status != "native":
        raise PromotionError("native provisional winner has inconsistent promotion status")
    if provisional != native and status == "native":
        raise PromotionError("challenger provisional winner has native promotion status")


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else \
        0.5 * (ordered[middle - 1] + ordered[middle])


def _validated_effect(confirmation: dict[str, Any]) -> float:
    native, winner = _paired_rounds(confirmation)
    if len(native) < MIN_PAIRED_ROUNDS:
        raise PromotionError("confirmation has insufficient paired rounds")
    if len(native) != len(winner):
        raise PromotionError("confirmation paired samples are missing or misaligned")
    observed = 100.0 * (_median(native) - _median(winner)) / _median(native)
    persisted = confirmation.get("effect_pct")
    if not isinstance(persisted, (int, float)) or not math.isfinite(float(persisted)):
        raise PromotionError("confirmation effect_pct is missing or non-finite")
    if not math.isclose(observed, float(persisted), rel_tol=1e-6, abs_tol=1e-5):
        raise PromotionError("confirmation effect_pct does not match paired samples")
    return observed


def _paired_rounds(confirmation: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Return strict, round-aligned confirmation samples.

    The tuner preserves failed rounds as non-finite sentinels.  Those are not
    evidence, but silently dropping malformed values would let an artifact
    claim more usable rounds than it actually contains.  Promotion therefore
    validates the persisted count and only then computes from the usable
    pairs.
    """
    native_raw = confirmation.get("native_us")
    winner_raw = confirmation.get("winner_us")
    if not isinstance(native_raw, list) or not isinstance(winner_raw, list):
        raise PromotionError("confirmation paired samples are missing")
    if len(native_raw) != len(winner_raw) or not native_raw:
        raise PromotionError("confirmation paired samples are missing or misaligned")
    native: list[float] = []
    winner: list[float] = []
    for left, right in zip(native_raw, winner_raw):
        if (not isinstance(left, (int, float)) or
                not isinstance(right, (int, float)) or
                not math.isfinite(float(left)) or not math.isfinite(float(right))):
            continue
        if float(left) <= 0.0 or float(right) <= 0.0:
            raise PromotionError("confirmation contains non-positive timing evidence")
        # The C++ paired sign test excludes exact ties from its declared
        # round count. Mirror that rule while reducing the aligned arrays;
        # persisted arrays retain the tie for positional auditability.
        if float(left) == float(right):
            continue
        native.append(float(left))
        winner.append(float(right))
    declared = confirmation.get("rounds")
    if not isinstance(declared, int) or declared != len(native):
        raise PromotionError("confirmation rounds do not match paired samples")
    wins = sum(b < a for a, b in zip(native, winner))
    declared_wins = confirmation.get("wins")
    if not isinstance(declared_wins, int) or declared_wins != wins:
        raise PromotionError("confirmation wins do not match paired samples")
    return native, winner


def validate_adaptive_evidence(row: dict[str, Any], header: dict[str, Any], *,
                               min_paired_rounds: int = MIN_PAIRED_ROUNDS) -> None:
    """Validate screen/final/confirmation and HI24 canary evidence.

    This is deliberately an offline artifact check.  It does not alter the
    live tuner or ranking policy; it only prevents promotion from trusting a
    partial or internally contradictory adaptive run.
    """
    if min_paired_rounds < 1:
        raise PromotionError("invalid minimum paired-round policy")
    for field in ("screen_samples", "final_samples", "confirmation_samples"):
        value = header.get(field)
        if value is not None and (not isinstance(value, int) or value < 1):
            raise PromotionError(f"invalid {field} evidence count")

    stage_counts = [row.get(field) for field in
                    ("generated", "applicable", "eligible", "measured")]
    if any(value is not None and (not isinstance(value, int) or value < 0)
           for value in stage_counts):
        raise PromotionError("invalid adaptive stage counts")
    present_counts = [value for value in stage_counts if value is not None]
    if present_counts != sorted(present_counts, reverse=True):
        raise PromotionError("adaptive stage counts are inconsistent")

    canary_state = row.get("canary_state")
    if canary_state is not None:
        if canary_state not in {"not_available", "pass", "retried_pass", "unresolved"}:
            raise PromotionError("unknown noise-canary state")
        retries = row.get("canary_retries", 0)
        if not isinstance(retries, int) or retries < 0:
            raise PromotionError("invalid noise-canary retry count")
        pair = row.get("canary_pair", "")
        pct = row.get("canary_pct", -1.0)
        if not isinstance(pair, str) or not isinstance(pct, (int, float)) or not math.isfinite(float(pct)):
            raise PromotionError("invalid noise-canary evidence")
        if canary_state == "not_available" and (pair or float(pct) >= 0.0):
            raise PromotionError("noise-canary not_available evidence is inconsistent")
        if canary_state != "not_available" and (not pair or float(pct) < 0.0):
            raise PromotionError("noise-canary evidence is incomplete")
        if canary_state == "unresolved" and retries < 1:
            raise PromotionError("unresolved noise-canary lacks retry evidence")
        if canary_state == "unresolved" and row.get("provisional_winner") not in (None, row.get("native")):
            raise PromotionError("unresolved noise-canary cannot claim a challenger winner")

    candidates = row.get("candidates")
    if candidates is not None:
        if not isinstance(candidates, list):
            raise PromotionError("candidate evidence must be a list")
        final_limit = header.get("final_samples")
        seen: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("name"), str):
                raise PromotionError("malformed candidate evidence")
            name = candidate["name"]
            if name in seen:
                raise PromotionError("duplicate candidate evidence")
            seen.add(name)
            samples = candidate.get("samples")
            if not isinstance(samples, int) or samples < 0:
                raise PromotionError("invalid candidate final sample count")
            samples_us = candidate.get("samples_us")
            if samples_us is not None:
                if not isinstance(samples_us, list):
                    raise PromotionError("candidate final samples must be a list")
                usable = [v for v in samples_us if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0]
                if samples != len(usable):
                    raise PromotionError("candidate final samples do not match samples_us")
            if final_limit is not None and samples > final_limit:
                raise PromotionError("candidate final samples exceed final_samples")

    confirmation = row.get("confirmation")
    if confirmation is not None:
        if not isinstance(confirmation, dict):
            raise PromotionError("malformed confirmation evidence")
        native, _ = _paired_rounds(confirmation)
        if len(native) < min_paired_rounds:
            raise PromotionError("confirmation has insufficient paired rounds")
        configured = header.get("confirmation_samples")
        if configured is not None and len(confirmation["native_us"]) < max(configured, min_paired_rounds):
            raise PromotionError("confirmation round payload is shorter than configured evidence")


def _read(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PromotionError(f"malformed current measurements line {number}") from exc
        if row.get("kind") == "header":
            if header is not None:
                raise PromotionError("duplicate header")
            header = row
        elif row.get("kind") == "result":
            try:
                validate_measurement_identity(row, header=header,
                                              where=f"measurements line {number}")
            except IdentitySeparationError as exc:
                raise PromotionError(str(exc)) from exc
            results.append(row)
        else:
            raise PromotionError(f"unknown current record kind at line {number}")
    if header is None or not results:
        raise PromotionError("current measurements header/results required")
    if "production_policy" in header and not isinstance(header["production_policy"], str):
        raise PromotionError("invalid production policy header identity")
    return header, results


def paired_bootstrap(native: list[Any], winner: list[Any], *, seed: int,
                     resamples: int = 10_000) -> tuple[float, float]:
    pairs = [
        (float(a), float(b)) for a, b in zip(native, winner)
        if a is not None and b is not None and float(a) > 0 and float(b) > 0 and
        math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if not pairs:
        raise PromotionError("confirmation has no usable paired rounds")
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [rng.choice(pairs) for _ in pairs]
        native_median = _median([a for a, _ in sample])
        winner_median = _median([b for _, b in sample])
        draws.append(100.0 * (native_median - winner_median) / native_median)
    draws.sort()
    return draws[int(0.025 * resamples)], draws[min(resamples - 1, int(0.975 * resamples))]


def validate_schedule(row: dict[str, Any]) -> None:
    """Verify current C++ schedule provenance before using confirmation data."""
    signature = row.get("signature")
    seed = row.get("schedule_seed")
    schedule = row.get("schedule")
    if (not isinstance(signature, str) or len(signature) != 32 or
            not isinstance(seed, int) or not isinstance(schedule, dict)):
        raise PromotionError("current schedule identity missing; rerun")
    expected_seed = int.from_bytes(bytes.fromhex(signature)[:4], "little")
    if seed != expected_seed:
        raise PromotionError("schedule seed drift detected")
    if (schedule.get("schema_version") != 1 or
            schedule.get("selection_algorithm") != "seeded-rotation-v1" or
            schedule.get("confirmation_algorithm") != "seeded-alternation-v1"):
        raise PromotionError("unknown current schedule algorithm; rerun")
    candidates = schedule.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2 or any(
            not isinstance(name, str) or not name for name in candidates):
        raise PromotionError("current selection schedule candidates missing; rerun")
    expected_order = sorted(
        candidates, key=lambda name: (name.removesuffix("#twin"), name.endswith("#twin"))
    )
    if candidates != expected_order or len(set(candidates)) != len(candidates):
        raise PromotionError("selection schedule position drift detected")


def benjamini_hochberg(rows: list[tuple[str, float]]) -> dict[str, float]:
    """Return monotone BH-adjusted q-values keyed by hypothesis identity."""
    ordered = sorted(rows, key=lambda item: (item[1], item[0]))
    if len({digest for digest, _ in ordered}) != len(ordered):
        raise PromotionError("duplicate hypothesis identity")
    count = len(ordered)
    adjusted = [min(1.0, p_value * count / rank)
                for rank, (_, p_value) in enumerate(ordered, 1)]
    for index in range(count - 2, -1, -1):
        adjusted[index] = min(adjusted[index], adjusted[index + 1])
    return {ordered[index][0]: adjusted[index] for index in range(count)}


def simulate_null_fdr(*, experiments: int, hypotheses: int, q: float,
                      seed: int) -> dict[str, Any]:
    """Exercise the shipped BH implementation under independent global nulls."""
    if experiments < 100 or hypotheses < 1 or not 0 < q < 1:
        raise PromotionError("invalid null-FDR simulation policy")
    root = random.Random(seed)
    runs: list[dict[str, Any]] = []
    false_discoveries = 0
    for experiment in range(experiments):
        experiment_seed = root.getrandbits(64)
        rng = random.Random(experiment_seed)
        rows = [(f"h{index:06d}", rng.random()) for index in range(hypotheses)]
        adjusted = benjamini_hochberg(rows)
        rejected = sorted(digest for digest, value in adjusted.items() if value <= q)
        false_discoveries += bool(rejected)
        runs.append({
            "experiment": experiment, "seed": experiment_seed,
            "p_values": [value for _, value in rows], "rejected": rejected,
        })
    observed = false_discoveries / experiments
    return {
        "schema_version": SCHEMA_VERSION, "method": "benjamini-hochberg-global-null",
        "seed": seed, "experiments": experiments, "hypotheses": hypotheses, "q": q,
        "experiments_with_false_discovery": false_discoveries,
        "empirical_fdr": observed, "runs": runs,
    }


def promote(measurements: Path, output: Path, *, q: float = 0.05,
            threshold_pct: float = 1.0, resamples: int = 10_000) -> dict[str, Any]:
    if not 0 < q < 1 or threshold_pct < 0 or resamples < 100:
        raise PromotionError("invalid predeclared promotion policy")
    header, results = _read(measurements)
    # HI37 (pending, tracked separately) will add variant_set/hardware_key/
    # config to this header; until then, only what the tuner actually emits
    # today is required.
    required_header = {"artifact_version", "source_revision", "manifest_hash"}
    if required_header - header.keys():
        raise PromotionError("current measurements header metadata required; rerun")
    if (header.get("artifact_version") != 1 or
            not isinstance(header.get("manifest_hash"), str) or
            len(header["manifest_hash"]) != 32):
        raise PromotionError("invalid current measurements header; rerun")
    hypotheses: list[tuple[str, float]] = []
    for row in results:
        _validate_policy_identity(row, header)
        if "provisional_winner" in row or "promotion_status" in row:
            _validate_provisional_status(row)
        validate_adaptive_evidence(row, header)
        provisional = row.get("provisional_winner")
        if not isinstance(provisional, str) or provisional == row.get("native"):
            continue
        original_status = row.get("promotion_status")
        if original_status not in {"pending_bh", "confirmation_rejected"}:
            raise PromotionError("non-native result lacks current pending_bh/confirmation evidence")
        validate_schedule(row)
        dispatch = row.get("dispatch")
        p_value = row.get("confirmation", {}).get("p_value")
        if (not isinstance(dispatch, str) or
                not isinstance(p_value, (int, float)) or
                not math.isfinite(float(p_value)) or not 0.0 <= float(p_value) <= 1.0):
            raise PromotionError("confirmation identity/p-value missing")
        hypotheses.append((dispatch, float(p_value)))
    adjusted = benjamini_hochberg(hypotheses) if hypotheses else {}
    accepted = {digest for digest, value in adjusted.items() if value <= q}
    promoted_count = 0
    for row in results:
        provisional = row.get("provisional_winner")
        if not isinstance(provisional, str) or provisional == row.get("native"):
            row["promotion_status"] = "native"
            continue
        confirmation = row["confirmation"]
        observed_effect = _validated_effect(confirmation)
        seed_material = (str(header.get("manifest_hash", "")) + row["dispatch"]).encode("ascii")
        seed = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "little")
        low, high = paired_bootstrap(
            confirmation.get("native_us", []), confirmation.get("winner_us", []),
            seed=seed, resamples=resamples,
        )
        original_status = row["promotion_status"]
        passed = (original_status == "pending_bh" and
                  row["dispatch"] in accepted and
                  observed_effect >= threshold_pct and low > 0.0)
        row["promotion"] = {
            "schema_version": SCHEMA_VERSION, "q": q,
            "threshold_pct": threshold_pct, "bootstrap_resamples": resamples,
            "bootstrap_seed": seed, "ci95_low_pct": low, "ci95_high_pct": high,
            "p_value": float(confirmation["p_value"]),
            "q_value": adjusted[row["dispatch"]],
            "bh_accepted": row["dispatch"] in accepted,
        }
        if passed:
            row["promotion_status"] = "promoted"
            promoted_count += 1
        else:
            if original_status == "confirmation_rejected":
                rejection_status = "confirmation_rejected"
            elif observed_effect < threshold_pct:
                rejection_status = "rejected_effect"
            elif low <= 0.0:
                rejection_status = "rejected_ci"
            else:
                rejection_status = "rejected_bh"
            row["promotion_status"] = rejection_status
            row["winner"] = row["native"]
            row["improvement_pct"] = 0.0
            row["reason"] = {
                "confirmation_rejected": "native retained by fresh confirmation",
                "rejected_effect": "native retained below the declared effect threshold",
                "rejected_ci": "native retained because the confidence interval crosses zero",
                "rejected_bh": "native retained by experiment-wide promotion policy",
            }[rejection_status]
    promoted_header = dict(header)
    promoted_header["promotion_policy"] = {
        "schema_version": SCHEMA_VERSION, "method": "benjamini-hochberg",
        "q": q, "threshold_pct": threshold_pct,
        "bootstrap_resamples": resamples, "hypotheses": len(hypotheses),
    }
    promoted_header["promotion_policy"]["policy_hash"] = hashlib.blake2b(
        canonical(promoted_header["promotion_policy"]), digest_size=16,
        person=b"bc-promotion-v1",
    ).hexdigest()
    data = b"\n".join([canonical(promoted_header), *(canonical(row) for row in results)]) + b"\n"
    atomic_write(output, data)
    return {
        "schema_version": SCHEMA_VERSION, "hypotheses": len(hypotheses),
        "promoted": promoted_count,
        "content_hash": hashlib.blake2b(data, digest_size=16).hexdigest(),
        "output": str(output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry tune-promote")
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q", type=float, default=0.05)
    parser.add_argument("--threshold-pct", type=float, default=1.0)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args(argv)
    try:
        result = promote(args.measurements, args.output, q=args.q,
                         threshold_pct=args.threshold_pct, resamples=args.resamples)
    except (OSError, ValueError, PromotionError) as exc:
        print(f"invalid: {exc}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


def null_fdr_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry tune-null-fdr")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", type=int, default=5000)
    parser.add_argument("--hypotheses", type=int, required=True)
    parser.add_argument("--q", type=float, default=0.05)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        result = simulate_null_fdr(experiments=args.experiments, hypotheses=args.hypotheses,
                                   q=args.q, seed=args.seed)
    except PromotionError as exc:
        print(f"invalid: {exc}")
        return 1
    atomic_write(args.output, canonical(result))
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
