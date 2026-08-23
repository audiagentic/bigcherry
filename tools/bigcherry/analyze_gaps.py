"""Deterministic, offline analysis of routing-transform gaps (HI32).

This module consumes the validated transform-record boundary. It deliberately
does not propose, generate, register, or enable runtime transformations, and
does not read or write SQLite, replay, or release artifacts.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .transform_records import (
    TransformRecordError,
    load_artifact_records,
    validate_transform_record,
)


class GapAnalysisError(ValueError):
    """The input set cannot produce a trustworthy gap report."""


def _provenance_key(record: dict[str, Any]) -> tuple[str | None, ...]:
    # Runtime-flat records (HI29) carry None for the provenance dimensions the
    # runtime does not emit (architecture, build_descriptor_hash).  Use actual
    # None in the key -- no sentinel string, and no schema marker (the
    # serialization format is not provenance); a flat record can never be
    # mixed with a complete nested one because the None vs digest dimensions
    # differ.
    hardware = record["hardware_provenance"]
    build = record["build_provenance"]
    return (
        hardware["digest"].casefold(),
        hardware["architecture"].casefold() if hardware["architecture"] is not None else None,
        build["source_revision"].casefold(),
        build["manifest_hash"].casefold(),
        build["build_descriptor_hash"].casefold()
        if build["build_descriptor_hash"] is not None else None,
    )


def _provenance_document(record: dict[str, Any]) -> dict[str, Any]:
    return {"hardware": dict(record["hardware_provenance"]),
            "build": dict(record["build_provenance"])}


def analyze_gaps(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate transform-gap records by pattern and native family.

    Every record is normalized through ``validate_transform_record`` (the
    strict nested evidence contract).  Attempts are ignored, while gaps must
    share one provenance namespace and must not reuse an evidence reference
    (case-insensitively).  Output ordering is stable.  Artifact-level loading
    of either schema goes through ``analyze_gap_file`` instead.
    """
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        try:
            normalized.append(validate_transform_record(raw, where=f"records[{index}]"))
        except TransformRecordError as exc:
            raise GapAnalysisError(str(exc)) from exc
    return _analyze_normalized(normalized)


def _analyze_normalized(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate already-normalized transform records.

    This is the aggregation half of ``analyze_gaps`` for records that were
    validated at the artifact boundary by ``load_artifact_records`` (which
    may be either the nested schema or the runtime-flat HI29 schema).
    Callers must not feed arbitrary dicts in here; the normalization boundary
    is the only place where record shape is established.
    """
    gaps: list[dict[str, Any]] = []
    provenance: tuple[str | None, ...] | None = None
    provenance_record: dict[str, Any] | None = None
    evidence_seen: set[str] = set()
    for record in records:
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
    """Load a transform JSONL file and return its deterministic gap report.

    The header dispatches the schema (nested evidence or runtime-flat HI29);
    records are validated once at this boundary and the analyzer consumes the
    normalized result.
    """
    try:
        records = load_artifact_records(path)
    except TransformRecordError as exc:
        raise GapAnalysisError(str(exc)) from exc
    return _analyze_normalized(records)


__all__ = ["GapAnalysisError", "analyze_gap_file", "analyze_gaps"]
