"""Strict current-run comparison for cache-residency and harness experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class CompareError(RuntimeError):
    pass


def _read(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    header: dict[str, Any] | None = None
    results: dict[str, dict[str, Any]] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CompareError(f"{path}: malformed line {number}") from exc
        if row.get("kind") == "header":
            if header is not None:
                raise CompareError(f"{path}: duplicate header")
            header = row
        elif row.get("kind") == "result":
            signature = row.get("signature")
            if not isinstance(signature, str) or signature in results:
                raise CompareError(f"{path}: missing or duplicate signature")
            results[signature] = row
        else:
            raise CompareError(f"{path}: unknown current record kind at line {number}")
    if header is None or not results:
        raise CompareError(f"{path}: current header and results required")
    return header, results


def _midranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank = ((i + 1) + j) / 2.0
        for key, _ in ordered[i:j]:
            ranks[key] = rank
        i = j
    return ranks


def spearman(left: dict[str, float], right: dict[str, float]) -> float | None:
    keys = sorted(set(left) & set(right))
    if len(keys) < 2:
        return None
    a, b = _midranks({key: left[key] for key in keys}), _midranks({key: right[key] for key in keys})
    am = sum(a.values()) / len(keys)
    bm = sum(b.values()) / len(keys)
    numerator = sum((a[k] - am) * (b[k] - bm) for k in keys)
    ad = sum((a[k] - am) ** 2 for k in keys)
    bd = sum((b[k] - bm) ** 2 for k in keys)
    return numerator / math.sqrt(ad * bd) if ad > 0 and bd > 0 else None


def _timings(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for candidate in row.get("candidates", []):
        value = candidate.get("effective_us")
        if (candidate.get("status") == "ok" and isinstance(value, (int, float)) and
                math.isfinite(float(value)) and float(value) > 0):
            out[str(candidate.get("name"))] = float(value)
    return out


def _calls(record: Path | None) -> dict[str, int]:
    if record is None:
        return {}
    calls: dict[str, int] = {}
    for raw in record.read_text(encoding="utf-8").splitlines():
        row = json.loads(raw)
        if row.get("kind") == "observation":
            calls[str(row["signature"])] = int(row["calls"])
    return calls


def compare(before: Path, after: Path, *, record: Path | None = None) -> dict[str, Any]:
    bh, br = _read(before)
    ah, ar = _read(after)
    for field in ("source_revision", "manifest_hash"):
        if bh.get(field) != ah.get(field):
            raise CompareError(f"runs differ in {field}; rerun under one current build/hardware")
    before_hardware = {row.get("hardware") for row in br.values()}
    after_hardware = {row.get("hardware") for row in ar.values()}
    if (len(before_hardware) != 1 or len(after_hardware) != 1 or
            None in before_hardware or before_hardware != after_hardware):
        raise CompareError("runs differ in hardware; rerun both arms on one device")
    common = sorted(set(br) & set(ar))
    if not common:
        raise CompareError("runs have no common signatures")
    counts = _calls(record)
    details: list[dict[str, Any]] = []
    before_total = after_total = 0.0
    weighted_calls = 0
    comparable_calls = 0
    missing_weighted_signatures = 0
    for signature in common:
        left, right = _timings(br[signature]), _timings(ar[signature])
        winner_before, winner_after = br[signature].get("winner"), ar[signature].get("winner")
        left_us, right_us = left.get(str(winner_before)), right.get(str(winner_after))
        calls = counts.get(signature, 0)
        if left_us is not None and right_us is not None and calls > 0:
            before_total += calls * left_us
            after_total += calls * right_us
            comparable_calls += calls
        elif calls > 0:
            missing_weighted_signatures += 1
        weighted_calls += calls
        details.append({
            "signature": signature, "calls": calls,
            "winner_before": winner_before, "winner_after": winner_after,
            "winner_agree": winner_before == winner_after,
            "common_candidates": len(set(left) & set(right)),
            "rank_correlation": spearman(left, right),
            "winner_before_us": left_us, "winner_after_us": right_us,
        })
    correlations = [d["rank_correlation"] for d in details if d["rank_correlation"] is not None]
    agreeing_calls = sum(d["calls"] for d in details if d["winner_agree"])
    return {
        "schema_version": 1, "before": str(before), "after": str(after),
        "common_signatures": len(common),
        "before_only": len(set(br) - set(ar)), "after_only": len(set(ar) - set(br)),
        "winner_agreement": sum(d["winner_agree"] for d in details),
        "winner_agreement_pct": 100.0 * sum(d["winner_agree"] for d in details) / len(details),
        "winner_agreement_calls": agreeing_calls,
        "winner_agreement_calls_pct": (
            100.0 * agreeing_calls / weighted_calls if weighted_calls else None
        ),
        "mean_rank_correlation": sum(correlations) / len(correlations) if correlations else None,
        "ranked_signatures": len(correlations), "weighted_calls": weighted_calls,
        "comparable_calls": comparable_calls,
        "call_coverage_pct": 100.0 * comparable_calls / weighted_calls if weighted_calls else None,
        "missing_weighted_signatures": missing_weighted_signatures,
        "call_weighted_time_change_pct": (
            100.0 * (after_total - before_total) / before_total
            if before_total > 0 and missing_weighted_signatures == 0 else None
        ),
        "details": details,
    }
