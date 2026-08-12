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

SCHEMA_VERSION = 1


class PromotionError(RuntimeError):
    pass


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
            results.append(row)
        else:
            raise PromotionError(f"unknown current record kind at line {number}")
    if header is None or not results:
        raise PromotionError("current measurements header/results required")
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
        native_sorted = sorted(a for a, _ in sample)
        winner_sorted = sorted(b for _, b in sample)
        native_median = native_sorted[len(native_sorted) // 2]
        winner_median = winner_sorted[len(winner_sorted) // 2]
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
        seed_material = (str(header.get("manifest_hash", "")) + row["dispatch"]).encode("ascii")
        seed = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "little")
        low, high = paired_bootstrap(
            confirmation.get("native_us", []), confirmation.get("winner_us", []),
            seed=seed, resamples=resamples,
        )
        original_status = row["promotion_status"]
        passed = (original_status == "pending_bh" and
                  row["dispatch"] in accepted and
                  float(confirmation.get("effect_pct", 0.0)) >= threshold_pct and low > 0.0)
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
            row["promotion_status"] = (
                "confirmation_rejected" if original_status == "confirmation_rejected"
                else "rejected_bh"
            )
            row["winner"] = row["native"]
            row["improvement_pct"] = 0.0
            row["reason"] = (
                "native retained by fresh confirmation"
                if original_status == "confirmation_rejected"
                else "native retained by experiment-wide promotion policy"
            )
    promoted_header = dict(header)
    promoted_header["promotion_policy"] = {
        "schema_version": SCHEMA_VERSION, "method": "benjamini-hochberg",
        "q": q, "threshold_pct": threshold_pct,
        "bootstrap_resamples": resamples, "hypotheses": len(hypotheses),
    }
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
