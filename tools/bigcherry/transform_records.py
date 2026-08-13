"""Offline validation and loading for routing-transform JSONL records.

This module is deliberately independent of the runtime transform feature.  It
defines the evidence boundary HI29/HI33 need before a future runtime writer or
SQLite loader can consume transform-attempt and transform-gap records.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class TransformRecordError(ValueError):
    """A transform JSONL header or record is malformed."""


_HEX32 = re.compile(r"^[0-9a-fA-F]{32}$")
_HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_KINDS = frozenset(("transform-attempt", "transform-gap"))
_SOURCES = frozenset(("predefined", "discovered"))
_OUTCOMES = frozenset(("success", "rejected", "gap"))


def _object(value: Any, field: str, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TransformRecordError(f"{where}.{field} must be an object")
    return value


def _text(value: Any, field: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TransformRecordError(f"{where}.{field} must be a non-empty string")
    return value.strip()


def _digest(value: Any, field: str, where: str, pattern: re.Pattern[str]) -> str:
    value = _text(value, field, where)
    if pattern.fullmatch(value) is None:
        raise TransformRecordError(f"{where}.{field} must be hexadecimal digest")
    return value.lower()


def _positive_int(value: Any, field: str, where: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (value < 0 if zero else value <= 0):
        bound = "non-negative" if zero else "positive"
        raise TransformRecordError(f"{where}.{field} must be a {bound} integer")
    return value


def _evidence(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise TransformRecordError(f"{where}.evidence_references must be non-empty")
    refs = [_text(item, f"evidence_references[{index}]", where)
            for index, item in enumerate(value)]
    folded = [item.casefold() for item in refs]
    if len(set(folded)) != len(folded):
        raise TransformRecordError(
            f"{where}.evidence_references must contain unique references")
    return refs


def _provenance(record: dict[str, Any], where: str) -> dict[str, Any]:
    hardware = _object(record.get("hardware_provenance"), "hardware_provenance", where)
    build = _object(record.get("build_provenance"), "build_provenance", where)
    normalized_hardware = {
        "digest": _digest(hardware.get("digest"), "hardware_provenance.digest", where, _HEX32),
        "architecture": _text(
            hardware.get("architecture"), "hardware_provenance.architecture", where),
    }
    normalized_build = {
        "source_revision": _digest(
            build.get("source_revision"), "build_provenance.source_revision", where, _HEX40),
        "manifest_hash": _digest(
            build.get("manifest_hash"), "build_provenance.manifest_hash", where, _HEX64),
        "build_descriptor_hash": _digest(
            build.get("build_descriptor_hash"),
            "build_provenance.build_descriptor_hash", where, _HEX64),
    }
    return {"hardware": normalized_hardware, "build": normalized_build}


def _transformation(record: dict[str, Any], where: str) -> dict[str, Any]:
    transformation = _object(record.get("transformation"), "transformation", where)
    ident = _positive_int(transformation.get("id"), "transformation.id", where, zero=True)
    name = _text(transformation.get("name"), "transformation.name", where)
    source = _text(transformation.get("source"), "transformation.source", where).casefold()
    if source not in _SOURCES:
        raise TransformRecordError(
            f"{where}.transformation.source must be predefined or discovered")
    return {"id": ident, "name": name, "source": source}


def _validate_attempt(record: dict[str, Any], where: str) -> dict[str, Any]:
    outcome = _text(record.get("outcome"), "outcome", where).casefold()
    if outcome not in ("success", "rejected"):
        raise TransformRecordError(f"{where}.outcome must be success or rejected")
    normalized = {
        "kind": "transform-attempt",
        "source_signature": _digest(record.get("source_signature"), "source_signature", where, _HEX32),
        "transformation": _transformation(record, where),
        "outcome": outcome,
        "reason": _text(record.get("reason"), "reason", where),
        "evidence_references": _evidence(record.get("evidence_references"), where),
    }
    provenance = _provenance(record, where)
    normalized["hardware_provenance"] = provenance["hardware"]
    normalized["build_provenance"] = provenance["build"]
    if "transformed_signature" in record and record["transformed_signature"] is not None:
        normalized["transformed_signature"] = _digest(
            record["transformed_signature"], "transformed_signature", where, _HEX32)
    return normalized


def _validate_gap(record: dict[str, Any], where: str) -> dict[str, Any]:
    tried = record.get("transformations_tried")
    if tried is None:
        transformation = record.get("transformation")
        if isinstance(transformation, dict):
            tried = transformation.get("tried")
    if not isinstance(tried, list) or not tried:
        raise TransformRecordError(f"{where}.transformations_tried must be non-empty")
    normalized_tried = []
    seen_ids: set[int] = set()
    for index, item in enumerate(tried):
        item_where = f"{where}.transformations_tried[{index}]"
        item = _object(item, "record", item_where)
        ident = _positive_int(item.get("id"), "id", item_where, zero=True)
        if ident in seen_ids:
            raise TransformRecordError(f"{item_where}.id duplicates another transformation")
        seen_ids.add(ident)
        normalized_tried.append({
            "id": ident,
            "name": _text(item.get("name"), "name", item_where),
            "reason": _text(item.get("reason"), "reason", item_where),
        })
    normalized = {
        "kind": "transform-gap",
        "source_signature": _digest(record.get("source_signature"), "source_signature", where, _HEX32),
        "transformation": {"tried": normalized_tried},
        "outcome": "gap",
        "reason": _text(record.get("reason"), "reason", where),
        "evidence_references": _evidence(record.get("evidence_references"), where),
        "pattern": _text(record.get("pattern"), "pattern", where),
        # Native-family was added after the initial JSONL schema.  Keep old
        # records readable while making the grouping dimension explicit.
        "native_family": "unknown",
    }
    if "native_family" in record and record["native_family"] is not None:
        normalized["native_family"] = _text(
            record["native_family"], "native_family", where).casefold()
    provenance = _provenance(record, where)
    normalized["hardware_provenance"] = provenance["hardware"]
    normalized["build_provenance"] = provenance["build"]
    if "calls" in record:
        normalized["calls"] = _positive_int(record["calls"], "calls", where, zero=True)
    if "est_bytes" in record:
        normalized["est_bytes"] = _positive_int(record["est_bytes"], "est_bytes", where, zero=True)
    return normalized


def validate_transform_record(record: Any, *, where: str = "record") -> dict[str, Any]:
    """Validate and normalize one transform-attempt or transform-gap record."""
    record = _object(record, "record", where)
    kind = _text(record.get("kind"), "kind", where)
    if kind == "transform-attempt":
        return _validate_attempt(record, where)
    if kind == "transform-gap":
        return _validate_gap(record, where)
    raise TransformRecordError(f"{where}.kind must be transform-attempt or transform-gap")


def _validate_header(record: Any, where: str) -> dict[str, Any]:
    header = _object(record, "record", where)
    if header.get("kind") != "header":
        raise TransformRecordError(f"{where}.kind must be header")
    version = header.get("transform_schema_version")
    if version != 1:
        raise TransformRecordError(f"{where}.transform_schema_version must be 1")
    return header


def load_transform_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a transform JSONL file, requiring one valid schema header.

    Blank lines are ignored.  Header and record failures include the source
    line number.  Runtime writing, SQLite, and replay formats are intentionally
    outside this loader's scope.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []
    header_seen = False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TransformRecordError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TransformRecordError(f"{path}:{line_number}: invalid JSON") from exc
        where = f"{path}:{line_number}"
        if isinstance(value, dict) and value.get("kind") == "header":
            if header_seen:
                raise TransformRecordError(f"{where}: duplicate header")
            _validate_header(value, where)
            header_seen = True
            continue
        if not header_seen:
            raise TransformRecordError(f"{where}: header must be first non-blank record")
        records.append(validate_transform_record(value, where=where))
    if not header_seen:
        raise TransformRecordError(f"{path}: missing transform JSONL header")
    return records
