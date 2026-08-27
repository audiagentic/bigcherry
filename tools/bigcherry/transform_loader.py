"""Offline SQLite loader for validated routing-transform evidence (HI33).

This module intentionally has no runtime-transform or replay dependencies.
Records must resolve to the exact existing build, hardware, and source
signature namespace; the loader never creates incomplete identity rows.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .tuning.inventory import RecordError, _require_current_schema
from .transform_records import TransformRecordError, load_artifact_records


def _blob(value: str, field: str) -> bytes:
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise RecordError(f"transform {field} is not hexadecimal") from exc
    if len(result) != 16:
        raise RecordError(f"transform {field} must be 16 bytes")
    return result


def _schema_supports_transforms(connection: sqlite3.Connection) -> None:
    _require_current_schema(connection)
    row = connection.execute(
        "SELECT value FROM schema_meta WHERE key = 'transform_schema'"
    ).fetchone()
    if row is None or row[0] != "1":
        raise RecordError("dispatch database does not support transform_schema '1'")
    for table in ("transform_attempt", "transform_gap"):
        if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone():
            raise RecordError(f"dispatch database is missing {table} table")


def _resolve_build_id(connection: sqlite3.Connection, build: dict[str, Any]) -> int:
    descriptor = build.get("build_descriptor_hash")
    if descriptor is None:
        # Runtime-flat record (HI29): the artifact asserts only
        # source_revision + manifest_hash as producer identity, so resolve the
        # build row by exactly those.  hardware/signature digests are GLOBAL
        # namespaces in the dispatch DB, not per-build evidence, so they must
        # never be used to disambiguate a build (an observation row only says
        # a build saw a pair, not that it produced this artifact).
        rows = connection.execute(
            "SELECT build_id FROM build WHERE source_revision=? AND manifest_hash=?",
            (build["source_revision"], build["manifest_hash"]),
        ).fetchall()
        if len(rows) == 0:
            raise RecordError("transform build provenance does not match an existing build")
        if len(rows) > 1:
            raise RecordError(
                "ambiguous build identity for runtime-flat transform record: "
                f"{len(rows)} builds share source_revision+manifest_hash; "
                "refusing to bind one (fail closed)")
        return rows[0][0]
    # Intrinsically-complete nested records bind by all three build identity
    # columns.  The source_revision+manifest_hash-only branch above remains
    # solely for the transitional runtime-flat schema (HI97).
    build_row = connection.execute(
        "SELECT build_id FROM build WHERE source_revision=? AND manifest_hash=? "
        "AND build_descriptor_hash=?",
        (build["source_revision"], build["manifest_hash"], descriptor),
    ).fetchone()
    if build_row is None:
        raise RecordError("transform build provenance does not match an existing build")
    return build_row[0]


def _identity_ids(connection: sqlite3.Connection, record: dict[str, Any]) -> tuple[int, int, bytes]:
    build = record["build_provenance"]
    hardware = record["hardware_provenance"]
    build_id = _resolve_build_id(connection, build)
    hardware_blob = _blob(hardware["digest"], "hardware_provenance.digest")
    hardware_row = connection.execute(
        "SELECT hardware_id FROM hardware WHERE hardware_digest=?", (hardware_blob,)
    ).fetchone()
    if hardware_row is None:
        raise RecordError("transform hardware provenance does not match existing hardware")
    signature_blob = _blob(record["source_signature"], "source_signature")
    if connection.execute(
        "SELECT 1 FROM signature WHERE signature_digest=?", (signature_blob,)
    ).fetchone() is None:
        raise RecordError("transform source_signature does not match an existing signature")
    return build_id, hardware_row[0], signature_blob


def load_transforms(
    transforms_path: Path,
    database_path: Path,
    schema_path: Path,
) -> dict[str, int]:
    """Load validated transform records into an existing dispatch database.

    Loading is idempotent.  All records are validated before any transaction
    is committed, and provenance mismatches fail closed.
    """
    # Header-dispatching boundary (HI97): a nested evidence artifact and a
    # flat runtime (HI29) artifact both load here; each is validated by its
    # own schema at the load boundary.
    try:
        records = load_artifact_records(transforms_path)
    except TransformRecordError as exc:
        raise RecordError(str(exc)) from exc
    connection = sqlite3.connect(str(database_path))
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        if not database_path.exists() or not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_meta'"
        ).fetchone():
            connection.executescript(schema_path.read_text(encoding="utf-8"))
        _schema_supports_transforms(connection)
        attempts = gaps = 0
        for record in records:
            build_id, hardware_id, signature_blob = _identity_ids(connection, record)
            evidence = json.dumps(record["evidence_references"], separators=(",", ":"))
            if record["kind"] == "transform-attempt":
                transformation = record["transformation"]
                connection.execute(
                    "INSERT OR IGNORE INTO transform_attempt "
                    "(build_id, hardware_id, signature_digest, transformation_id, "
                    "transformation_name, source, result, reason, transformed_sig_digest, "
                    "evidence_references) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (build_id, hardware_id, signature_blob, transformation["id"],
                     transformation["name"], transformation["source"], record["outcome"],
                     record["reason"],
                     _blob(record["transformed_signature"], "transformed_signature")
                     if record.get("transformed_signature") else None, evidence),
                )
                attempts += connection.execute("SELECT changes()").fetchone()[0]
            else:
                observation = connection.execute(
                    "SELECT calls, est_bytes FROM observation WHERE build_id=? "
                    "AND hardware_id=? AND signature_id=(SELECT signature_id FROM signature "
                    "WHERE signature_digest=?)",
                    (build_id, hardware_id, signature_blob),
                ).fetchone()
                connection.execute(
                    "INSERT OR IGNORE INTO transform_gap "
                    "(build_id, hardware_id, signature_digest, pattern_description, "
                    "native_family, transformations_tried, calls, est_bytes, "
                    "evidence_references) VALUES (?,?,?,?,?,?,?,?,?)",
                    (build_id, hardware_id, signature_blob, record["pattern"],
                     record["native_family"], json.dumps(record["transformation"]["tried"],
                     separators=(",", ":")), observation[0] if observation else record.get("calls", 0),
                     observation[1] if observation else record.get("est_bytes", 0), evidence),
                )
                gaps += connection.execute("SELECT changes()").fetchone()[0]
        connection.commit()
        return {"attempts": attempts, "gaps": gaps}
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
