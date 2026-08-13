"""Deterministic, offline analysis of routing-transform gaps (HI32).

This module consumes the validated transform-record boundary. It deliberately
does not propose, generate, register, or enable runtime transformations, and
does not read or write SQLite, replay, or release artifacts.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .transform_records import TransformRecordError, load_transform_records, validate_transform_record


class GapAnalysisError(ValueError):
    """The input set cannot produce a trustworthy gap report."""


def _provenance_key(record: dict[str, Any]) -> tuple[str, ...]:
    hardware = record["hardware_provenance"]
    build = record["build_provenance"]
    return (hardware["digest"].casefold(), hardware["architecture"].casefold(),
            build["source_revision"].casefold(), build["manifest_hash"].casefold(),
            build["build_descriptor_hash"].casefold())


def _provenance_document(record: dict[str, Any]) -> dict[str, Any]:
    return {"hardware": dict(record["hardware_provenance"]),
            "build": dict(record["build_provenance"])}


def analyze_gaps(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate transform-gap records by pattern and native family.

    Every record is normalized through ``validate_transform_record``. Attempts
    are ignored, while gaps must share one provenance namespace and must not
    reuse an evidence reference (case-insensitively). Output ordering is stable.
    """
    gaps: list[dict[str, Any]] = []
    provenance: tuple[str, ...] | None = None
    provenance_record: dict[str, Any] | None = None
    evidence_seen: set[str] = set()
    for index, raw in enumerate(records):
        try:
            record = validate_transform_record(raw, where=f"records[{index}]")
        except TransformRecordError as exc:
            raise GapAnalysisError(str(exc)) from exc
        if record["kind"] != "transform-gap":
            continue
        current = _provenance_key(record)
        if provenance is None:
            provenance, provenance_record = current, record
        elif current != provenance:
            raise GapAnalysisError("transform-gap records have mixed hardware/build provenance")
        for reference in record["evidence_references"]:
            folded = reference.casefold()
            if folded in evidence_seen:
                raise GapAnalysisError(
                    f"transform-gap evidence reference is duplicated across records: {reference}")
            evidence_seen.add(folded)
        gaps.append(record)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in gaps:
        key = (record["pattern"], record.get("native_family", "unknown"))
        grouped.setdefault(key, []).append(record)
    groups = []
    for (pattern, native_family), members in sorted(grouped.items()):
        members.sort(key=lambda item: item["source_signature"])
        groups.append({
            "pattern": pattern,
            "native_family": native_family,
            "count": len(members),
            "calls": sum(item.get("calls", 0) for item in members),
            "estimated_bytes": sum(item.get("est_bytes", 0) for item in members),
            "source_signatures": [item["source_signature"] for item in members],
        })
    return {
        "schema_version": 1,
        "record_count": len(gaps),
        "group_count": len(groups),
        "provenance": (_provenance_document(provenance_record)
                       if provenance_record is not None else None),
        "groups": groups,
    }


def analyze_gap_file(path: str | Path) -> dict[str, Any]:
    """Load a transform JSONL file and return its deterministic gap report."""
    try:
        records = load_transform_records(path)
    except TransformRecordError as exc:
        raise GapAnalysisError(str(exc)) from exc
    return analyze_gaps(records)


__all__ = ["GapAnalysisError", "analyze_gap_file", "analyze_gaps"]
