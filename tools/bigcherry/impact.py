"""Offline, call-weighted tuning impact calculations (HI35).

This module deliberately consumes existing record and measurements JSONL.  It
does not alter either artifact or the replay/database wire formats.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any


class ImpactError(ValueError):
    pass


@dataclass
class Coverage:
    matched: int = 0
    record_only: list[str] = field(default_factory=list)
    measurement_only: list[str] = field(default_factory=list)
    calls_total: int = 0
    calls_covered: int = 0


def _finite(value: Any, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImpactError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (not positive and value < 0):
        raise ImpactError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    return value


def _candidate(result: dict[str, Any], name: str) -> dict[str, Any] | None:
    for candidate in result.get("candidates", []):
        if candidate.get("name") == name and candidate.get("status") == "ok":
            if "median_us" not in candidate:
                raise ImpactError(f"candidate {name!r} has no median_us")
            return candidate
    return None


def predicted_saving(observations: list[dict[str, Any]],
                     results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate explicit coverage and call-weighted native-vs-winner saving."""
    calls: dict[str, int] = {}
    native_names: dict[str, str] = {}
    for row in observations:
        signature = row.get("signature")
        count = row.get("calls")
        if not isinstance(signature, str) or not signature:
            raise ImpactError("observation signature is required")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ImpactError(f"calls for {signature} must be a non-negative integer")
        if signature in calls:
            raise ImpactError(f"duplicate observation signature {signature}")
        calls[signature] = count
        native_names[signature] = row.get("native", "")

    coverage = Coverage(calls_total=sum(calls.values()))
    coverage.record_only = sorted(calls)
    measurement_signatures: set[str] = set()
    rows: list[dict[str, Any]] = []
    by_family: dict[str, dict[str, float | int]] = collections.defaultdict(
        lambda: {"calls": 0, "native_us": 0.0, "tuned_us": 0.0}
    )
    native_total = tuned_total = 0.0
    for result in results:
        signature = result.get("signature")
        if not isinstance(signature, str) or not signature:
            raise ImpactError("measurement signature is required")
        if signature in measurement_signatures:
            raise ImpactError(f"duplicate measurement signature {signature}")
        measurement_signatures.add(signature)
        if signature not in calls:
            continue
        coverage.record_only.remove(signature)
        native_name = result.get("native") or native_names[signature]
        winner_name = result.get("winner")
        if not isinstance(native_name, str) or not isinstance(winner_name, str):
            raise ImpactError(f"native and winner are required for {signature}")
        native = _candidate(result, native_name)
        winner = _candidate(result, winner_name)
        if native is None or winner is None:
            continue
        count = calls[signature]
        native_us_each = _finite(native["median_us"], "native median_us")
        winner_us_each = _finite(winner["median_us"], "winner median_us")
        native_us = count * native_us_each
        tuned_us = count * winner_us_each
        native_total += native_us
        tuned_total += tuned_us
        coverage.matched += 1
        coverage.calls_covered += count
        family = winner_name.split(":", 1)[0]
        bucket = by_family[family]
        bucket["calls"] += count
        bucket["native_us"] += native_us
        bucket["tuned_us"] += tuned_us
        rows.append({"signature": signature, "calls": count,
                     "saved_us_each": native_us_each - winner_us_each,
                     "saved_us": native_us - tuned_us, "winner": winner_name})
    coverage.measurement_only = sorted(measurement_signatures - set(calls))
    rows.sort(key=lambda row: (-row["saved_us"], row["signature"]))
    return {
        "coverage": coverage,
        "native_total_us": native_total,
        "tuned_total_us": tuned_total,
        "saving_pct": (100.0 * (native_total - tuned_total) / native_total
                        if native_total else 0.0),
        "by_family": dict(by_family), "rows": rows,
        "slower": [row for row in rows if row["saved_us"] < 0],
    }


def expected_decode_effect(saving_pct: Any, decode_matmul_fraction: Any) -> float:
    """Return the predicted decode-wall effect, with explicit unit checks."""
    saving = _finite(saving_pct, "predicted saving", positive=False)
    fraction = _finite(decode_matmul_fraction, "decode matmul fraction", positive=False)
    if fraction > 1.0:
        raise ImpactError("decode matmul fraction must be between 0 and 1")
    if saving > 100.0:
        raise ImpactError("predicted saving cannot exceed 100 percent")
    return (saving / 100.0) * fraction * 100.0
