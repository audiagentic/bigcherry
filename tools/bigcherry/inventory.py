"""Turn a record-mode JSONL file into an inventory and a SQLite database.

Record mode writes JSON Lines from C++ (see `hip-autotune-record.cpp` for why
it is not SQLite). This module is the other half: it reads that file and
produces the two things downstream needs.

    inventory JSON  ->  `generate --variant-set workload-max`
    SQLite database ->  querying, and the tuner's measurement store

`sqlite3` is in the Python standard library, so the database is built with no
dependency on either machine. The schema is `sql/dispatch-db.sql` unchanged --
moving the writer offline did not change what is stored, only who stores it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths
from .identity_separation import IdentitySeparationError, validate_measurement_identity


class RecordError(RuntimeError):
    pass


CURRENT_DB_SCHEMA_VERSION = "2"

_RESULT_STATUSES = {
    "ok", "architecture", "ineligible", "workspace", "launch_failed",
    "nan_inf", "tolerance", "unstable", "noisy",
}


def _validate_measurement_header(header: dict[str, Any], line: int) -> None:
    """Require the sampling policy needed to interpret timing results."""
    for field in ("final_samples", "warmup_launches", "screen_samples",
                  "confirmation_samples"):
        if field not in header:
            continue  # retain compatibility with pre-HI34 artifacts
        value = header[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RecordError(
                f"measurements line {line}: {field} must be a non-negative integer"
            )


def _finite_number(value: Any, field: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordError(f"measurement result field {field!r} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise RecordError(f"measurement result field {field!r} is invalid")
    return number


def _validate_measurement_result(row: Any, line: int) -> dict[str, Any]:
    """Validate one complete tuner result before it can affect the DB.

    The C++ tuner emits one result per dispatch.  Treating malformed rows as
    absent makes a partial run look like a smaller successful run, so this
    boundary is deliberately fail-closed.
    """
    if not isinstance(row, dict):
        raise RecordError(f"measurements line {line}: result must be an object")
    try:
        validate_measurement_identity(row, where=f"measurements line {line}")
    except IdentitySeparationError as exc:
        raise RecordError(str(exc)) from exc
    dispatch = row.get("dispatch")
    if (not isinstance(dispatch, str) or len(dispatch) != 32 or
            any(c not in "0123456789abcdefABCDEF" for c in dispatch)):
        raise RecordError(f"measurements line {line}: invalid dispatch digest")
    winner = row.get("winner")
    if not isinstance(winner, str) or not winner:
        raise RecordError(f"measurements line {line}: result requires winner")
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RecordError(f"measurements line {line}: result requires candidates")
    names: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise RecordError(f"measurements line {line}: candidate must be an object")
        name = candidate.get("name")
        status = candidate.get("status", "ok")
        if not isinstance(name, str) or not name or name in names:
            raise RecordError(f"measurements line {line}: invalid candidate name")
        if status not in _RESULT_STATUSES:
            raise RecordError(f"measurements line {line}: unknown candidate status {status!r}")
        names.add(name)
        for field in ("median_us", "mad_us", "p95_us", "host_median_us",
                      "nmse", "max_abs", "workspace", "samples"):
            if field in candidate:
                _finite_number(candidate[field], field)
        samples = candidate.get("samples", 0)
        if int(samples) != samples:
            raise RecordError(f"measurements line {line}: samples must be an integer")
        if "samples_us" in candidate:
            samples_us = candidate["samples_us"]
            if not isinstance(samples_us, list):
                raise RecordError(f"measurements line {line}: samples_us must be an array")
            for sample in samples_us:
                if sample is not None:
                    _finite_number(sample, "samples_us")
            reported = candidate.get("samples", 0)
            usable = sum(sample is not None for sample in samples_us)
            if reported != usable:
                raise RecordError(
                    f"measurements line {line}: samples does not match samples_us"
                )
    if winner not in names:
        raise RecordError(f"measurements line {line}: winner is not a candidate")
    for field in ("improvement_pct", "confidence"):
        if field in row:
            _finite_number(row[field], field, nonnegative=False)
    for field in ("generated", "applicable", "eligible", "measured"):
        if field in row:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RecordError(f"measurements line {line}: {field} must be non-negative integer")
    if "launches_per_sample" in row:
        value = row["launches_per_sample"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RecordError(
                f"measurements line {line}: launches_per_sample must be positive integer"
            )
    return row


def _require_current_schema(connection: sqlite3.Connection) -> None:
    """Reject databases whose shape this reader has not explicitly audited."""
    try:
        row = connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.Error as exc:
        raise RecordError(
            "dispatch database is missing schema metadata; refusing to infer "
            "an unverified schema"
        ) from exc

    actual = row[0] if row is not None else None
    if actual != CURRENT_DB_SCHEMA_VERSION:
        raise RecordError(
            "unsupported dispatch database schema_version "
            f"{actual!r}; expected {CURRENT_DB_SCHEMA_VERSION!r}"
        )


@dataclass
class Record:
    header: dict[str, Any]
    observations: list[dict[str, Any]] = field(default_factory=list)


def read_jsonl(path: Path) -> Record:
    """Parse a record file, tolerating a truncated final line.

    A run killed mid-flush leaves a partial last line. That is exactly the case
    JSON Lines exists to survive, so it is a warning rather than an error --
    everything before it is still valid data.
    """
    header: dict[str, Any] | None = None
    observations: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"warning: {path.name} line {number} is truncated or "
                    f"malformed; ignoring it and everything after",
                    file=sys.stderr,
                )
                break
            kind = row.get("kind")
            if kind == "header":
                header = row
            elif kind == "observation":
                observations.append(row)

    if header is None:
        raise RecordError(
            f"{path}: no header line. Either the file is not a bigcherry "
            f"record, or the run died before its first flush."
        )
    return Record(header=header, observations=observations)


# ------------------------------------------------------------------ inventory

# ggml_type values that matter to the catalog, by numeric id. Recorded
# signatures carry the enum value, and the catalog speaks in short names.
_GGML_TYPE_NAMES = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    9: "q8_1",
    10: "q2_k",
    11: "q3_k",
    12: "q4_k",
    13: "q5_k",
    14: "q6_k",
    15: "q8_k",
    16: "iq2_xxs",
    17: "iq2_xs",
    18: "iq3_xxs",
    19: "iq1_s",
    20: "iq4_nl",
    21: "iq3_s",
    22: "iq2_s",
    23: "iq4_xs",
    24: "i8",
    25: "i16",
    26: "i32",
    27: "i64",
    28: "f64",
    29: "iq1_m",
    30: "bf16",
    39: "mxfp4",
}


def type_name(value: int) -> str | None:
    return _GGML_TYPE_NAMES.get(value)


def build_inventory(record: Record) -> dict[str, Any]:
    """Derive the observed type and width sets the catalog restricts to.

    The family a signature ran under comes from its native candidate's stable
    name, which is the only place that information exists -- the signature
    describes the *operation*, deliberately not the implementation chosen for
    it (standards 5.1).
    """
    families: dict[str, set[str]] = {
        "mmq": set(),
        "mmvq": set(),
        "mmvf": set(),
        "mmf": set(),
    }
    widths: set[int] = set()
    uses_blas = False
    unknown_types: Counter[int] = Counter()

    for observation in record.observations:
        native = observation.get("native", "")
        family = native.split(":", 1)[0] if ":" in native else ""
        canonical = observation.get("canonical", {})

        src0 = canonical.get("src0_type")
        if src0 is not None:
            name = type_name(int(src0))
            if name is None:
                unknown_types[int(src0)] += 1
            elif family in families:
                families[family].add(name)

        # The dst width is operation-semantic: MUL_MAT_ID uses the expert
        # token dimension (ned[2]), while dense MUL_MAT uses ned[1].
        ned = canonical.get("ned")
        has_ids = bool(int(canonical.get("flags", 0)) & (1 << 3))
        width_index = 2 if has_ids else 1
        if isinstance(ned, list) and len(ned) > width_index:
            width = int(ned[width_index])
            if 1 <= width <= 16:
                widths.add(width)

        if family == "blas":
            uses_blas = True

    if unknown_types:
        print(
            f"warning: {sum(unknown_types.values())} observation(s) use "
            f"ggml_type ids not in the name table: "
            f"{sorted(unknown_types)}. Add them to _GGML_TYPE_NAMES or the "
            f"catalog will silently omit those types.",
            file=sys.stderr,
        )

    return {
        "source_revision": record.header.get("source_revision", ""),
        "manifest_hash": record.header.get("manifest_hash", ""),
        "signatures_observed": len(record.observations),
        "mmq_types": sorted(families["mmq"]),
        "mmvq_types": sorted(families["mmvq"]),
        "mmvf_types": sorted(families["mmvf"]),
        "mmf_types": sorted(families["mmf"]),
        "widths": sorted(widths),
        "uses_blas": uses_blas,
    }


# -------------------------------------------------------------------- sqlite


def build_database(record: Record, target: Path, schema: Path) -> dict[str, int]:
    """Populate a fresh SQLite database from a record file."""
    if target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(target))
    cursor = None
    try:
        connection.executescript(schema.read_text(encoding="utf-8"))
        _require_current_schema(connection)
        build_columns = {row[1] for row in connection.execute("PRAGMA table_info(build)")}
        if "build_descriptor_hash" not in build_columns:
            connection.execute("ALTER TABLE build ADD COLUMN build_descriptor_hash TEXT")
        header = record.header

        cursor = connection.execute(
            "INSERT INTO build (source_revision, source_dirty, manifest_hash, "
            "signature_schema, hardware_schema, variant_set, build_descriptor_hash) "
            "VALUES (?, 0, ?, ?, ?, ?, ?)",
            (
                header.get("source_revision", ""),
                header.get("manifest_hash", ""),
                header.get("signature_schema", 1),
                header.get("hardware_schema", 1),
                header.get("variant_set", "inventory"),
                header.get("build_descriptor_hash"),
            ),
        )
        build_id = cursor.lastrowid
        cursor.close()

        hardware_ids: dict[str, int] = {}
        signature_ids: dict[str, int] = {}

        for observation in record.observations:
            hardware_hex = observation["hardware"]
            if hardware_hex not in hardware_ids:
                key = observation.get("hardware_key", {})
                cursor = connection.execute(
                    "INSERT INTO hardware (hardware_digest, architecture, "
                    "architecture_code, wave_size, compute_units, "
                    "feature_flags, canonical_json) VALUES (?,?,?,?,?,?,?)",
                    (
                        bytes.fromhex(hardware_hex),
                        str(key.get("architecture_code", "")),
                        key.get("architecture_code", 0),
                        key.get("wave_size", 0),
                        key.get("compute_units", 0),
                        key.get("feature_flags", 0),
                        json.dumps(key, sort_keys=True, separators=(",", ":")),
                    ),
                )
                hardware_ids[hardware_hex] = cursor.lastrowid

            signature_hex = observation["signature"]
            if signature_hex not in signature_ids:
                canonical = observation.get("canonical", {})
                ned = canonical.get("ned", [0, 0, 0, 0])
                ne0 = canonical.get("ne0", [0, 0, 0, 0])
                has_ids = bool(int(canonical.get("flags", 0)) & (1 << 3))
                cursor = connection.execute(
                    "INSERT INTO signature (signature_digest, base_digest, "
                    "schema_version, op, src0_type, src1_type, dst_type, "
                    "m, n, k, canonical_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        bytes.fromhex(signature_hex),
                        bytes.fromhex(signature_hex),
                        canonical.get("schema_version", 1),
                        str(canonical.get("op", "")),
                        str(canonical.get("src0_type", "")),
                        str(canonical.get("src1_type", "")),
                        str(canonical.get("dst_type", "")),
                        ne0[1] if len(ne0) > 1 else 0,
                        ned[2] if has_ids and len(ned) > 2 else ned[1] if len(ned) > 1 else 0,
                        ne0[0] if ne0 else 0,
                        json.dumps(canonical, sort_keys=True, separators=(",", ":")),
                    ),
                )
                signature_ids[signature_hex] = cursor.lastrowid

            connection.execute(
                "INSERT OR REPLACE INTO observation (build_id, hardware_id, "
                "signature_id, native_stable_name, calls, est_bytes, "
                "sites_json) VALUES (?,?,?,?,?,?,?)",
                (
                    build_id,
                    hardware_ids[hardware_hex],
                    signature_ids[signature_hex],
                    observation.get("native", ""),
                    observation.get("calls", 0),
                    observation.get("est_bytes", 0),
                    json.dumps(observation.get("devices", [])),
                ),
            )

        connection.commit()
        return {
            "builds": 1,
            "hardware": len(hardware_ids),
            "signatures": len(signature_ids),
            "observations": len(record.observations),
        }
    finally:
        connection.close()


def load_measurements(
    measurements_path: Path,
    database_path: Path,
    schema_path: Path,
    *,
    manifest_path: Path | None = None,
    signature_source_paths: list[Path] | None = None,
) -> dict[str, int]:
    """Load tuning measurements JSONL into SQLite.

    Reads the .measurements.jsonl written by the tuning engine (HI12)
    and populates build, candidate, measurement, and winner tables.

    The dispatch_digest is the primary lookup key. signature_id is NULL
    when only the dispatch digest is available — it can be linked later
    if the DB was also built from record-mode observations.

    Duplicate measurements and winners (same ``build_id``, ``hardware_id``,
    ``candidate_id``, ``objective``, ``stage``, ``dispatch_digest``) are
    replaced by the newer data via ``INSERT OR REPLACE``, making reloads
    idempotent. This supports incremental tuning where new runs supersede
    old results for the same signatures.

    If ``manifest_path`` is provided, candidate rows are populated from
    the manifest's full descriptor data; otherwise only the stable_name
    and family (derived from the name prefix) are recorded.
    """
    # Read measurements JSONL
    results: list[dict[str, Any]] = []
    header: dict[str, Any] | None = None

    with measurements_path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecordError(
                    f"{measurements_path}: line {number} is malformed JSON"
                ) from exc
            if not isinstance(row, dict):
                raise RecordError(f"{measurements_path}: line {number} must be an object")
            kind = row.get("kind")
            if kind == "header":
                if header is not None:
                    raise RecordError(f"{measurements_path}: duplicate header at line {number}")
                header = row
                _validate_measurement_header(row, number)
            elif kind == "result":
                results.append(_validate_measurement_result(row, number))
            else:
                raise RecordError(
                    f"{measurements_path}: line {number} has unknown record kind {kind!r}"
                )

    if header is None:
        raise RecordError(
            f"{measurements_path}: no header line. Either the file is not a "
            f"bigcherry measurements file, or the run died before its first "
            f"flush."
        )

    # Recover canonical shapes from record/replay diagnostics when older
    # measurement artifacts predate inline `canonical` metadata.
    signature_shapes: dict[str, dict[str, Any]] = {}
    for source_path in signature_source_paths or []:
        if not source_path.is_file():
            continue
        with source_path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                signature_hex = row.get("signature", "")
                canonical = row.get("canonical")
                if len(signature_hex) == 32 and isinstance(canonical, dict):
                    signature_shapes.setdefault(signature_hex, canonical)

    # Resolve manifest → candidate lookup if provided
    manifest_by_name: dict[str, dict[str, Any]] = {}
    if manifest_path and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as e:
            raise RecordError(f"{manifest_path}: not valid JSON: {e}") from e
        manifest_by_name = {c["stable_name"]: c for c in manifest.get("candidates", [])}

    # Derive family from stable name prefix (for minimal insertion without manifest)
    def _family_from_name(name: str) -> str:
        if ":" not in name:
            return "unknown"
        return name.split(":", 1)[0]

    # Build the database (or open existing)
    db_path = Path(database_path)
    if db_path.exists():
        connection = sqlite3.connect(str(db_path))
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(db_path))
        # Initialize schema
        connection.executescript(schema_path.read_text(encoding="utf-8"))

    try:
        _require_current_schema(connection)
        connection.execute("PRAGMA foreign_keys = ON")
        for table in ("measurement", "winner"):
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            if "run_id" not in columns:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN run_id INTEGER")
            if "pool_peak_bytes" not in columns:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN pool_peak_bytes INTEGER")
        build_columns = {row[1] for row in connection.execute("PRAGMA table_info(build)")}
        if "build_descriptor_hash" not in build_columns:
            connection.execute("ALTER TABLE build ADD COLUMN build_descriptor_hash TEXT")

        # Find or create build row
        source_revision = header.get("source_revision", "")
        manifest_hash = header.get("manifest_hash", "")
        variant_set = header.get("variant_set")
        if not isinstance(variant_set, str) or not variant_set:
            raise ValueError("measurements header requires variant_set")
        build_descriptor_hash = header.get("build_descriptor_hash")
        artifact_version = header.get("artifact_version", 1)

        cursor = connection.execute(
            "SELECT build_id FROM build WHERE source_revision = ? "
            "AND manifest_hash = ? AND variant_set = ? "
            "AND (build_descriptor_hash = ? OR (build_descriptor_hash IS NULL AND ? IS NULL))",
            (source_revision, manifest_hash, variant_set,
             build_descriptor_hash, build_descriptor_hash),
        )
        build_row = cursor.fetchone()
        if build_row:
            build_id = build_row[0]
        else:
            # `compiler` is HI12 E6 -- omitted here for a while even after the
            # tuner started writing it in the header, which is exactly the
            # class of defect E6 exists to close.
            #
            cursor = connection.execute(
                "INSERT INTO build (source_revision, source_dirty, "
                "manifest_hash, signature_schema, hardware_schema, variant_set, "
                "dispatch_abi, compiler, hip_version, build_descriptor_hash) "
                "VALUES (?, 0, ?, 1, 1, ?, ?, ?, ?, ?)",
                (
                    source_revision,
                    manifest_hash,
                    variant_set,
                    str(artifact_version),
                    header.get("compiler"),
                    header.get("hip_version"),
                    build_descriptor_hash,
                ),
            )
            build_id = cursor.lastrowid

        signatures = sorted({str(row.get("signature", "")) for row in results
                             if isinstance(row.get("signature"), str)})
        workload_digest = hashlib.blake2b(
            "\n".join(signatures).encode("ascii", "ignore"), digest_size=16
        ).digest()
        run_material = json.dumps({
            "source_revision": source_revision,
            "manifest_hash": manifest_hash,
            "workload_digest": workload_digest.hex(),
            "header": header,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        run_digest = hashlib.blake2b(run_material, digest_size=16).digest()
        cursor = connection.execute(
            "SELECT run_id FROM tuning_run WHERE run_digest = ?", (run_digest,))
        run_row = cursor.fetchone()
        if run_row:
            run_id = run_row[0]
        else:
            cursor = connection.execute(
                "INSERT INTO tuning_run (build_id, run_digest, workload_digest, "
                "workload_label, host_sync_overhead_us, config_json, machine_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (build_id, run_digest, workload_digest, header.get("workload_label"),
                 header.get("host_sync_overhead_us"),
                 json.dumps(header, sort_keys=True, separators=(",", ":")),
                 json.dumps({"compiler": header.get("compiler"),
                             "hip_version": header.get("hip_version")},
                            sort_keys=True, separators=(",", ":"))),
            )
            run_id = cursor.lastrowid

        # Find hardware row (use placeholder if not from record mode)
        # The measurements JSONL doesn't include hardware info directly.
        # We look for any existing hardware row or create a default.
        #
        # KNOWN GAP (HI37/HI48, not fixed here): this binds to whichever
        # hardware row happens to be first, not the row matching the
        # digest actually recorded in each result's "hardware" field. Wrong
        # on any database that already holds more than one architecture's
        # rows. Needs resolving by result["hardware"] digest, with an
        # explicit stub row keyed by that digest when the descriptor is
        # incomplete -- never a placeholder zero digest pretending to be
        # measured hardware.
        hardware_ids: dict[str, int] = {}
        legacy_hardware_id: int | None = None

        def _resolve_hardware(result: dict[str, Any]) -> int:
            nonlocal legacy_hardware_id
            digest_hex = result.get("hardware")
            if not isinstance(digest_hex, str) or len(digest_hex) != 32:
                if legacy_hardware_id is None:
                    cursor = connection.execute(
                        "INSERT INTO hardware (hardware_digest, architecture, "
                        "architecture_code, wave_size, compute_units, feature_flags, "
                        "canonical_json) VALUES (X'00000000000000000000000000000000', "
                        "'unknown-incomplete', 0, 0, 0, 0, '{\"complete\":false}')"
                    )
                    legacy_hardware_id = cursor.lastrowid
                return legacy_hardware_id
            try:
                raw = bytes.fromhex(digest_hex)
            except ValueError as exc:
                raise RecordError(f"invalid tuning hardware digest: {digest_hex!r}") from exc
            if len(raw) != 16:
                raise RecordError("tuning hardware digest must be 16 bytes")
            if digest_hex not in hardware_ids:
                cursor = connection.execute(
                    "SELECT hardware_id FROM hardware WHERE hardware_digest = ?", (raw,))
                row = cursor.fetchone()
                if row:
                    hardware_ids[digest_hex] = row[0]
                else:
                    cursor = connection.execute(
                        "INSERT INTO hardware (hardware_digest, architecture, "
                        "architecture_code, wave_size, compute_units, feature_flags, "
                        "canonical_json) VALUES (?, 'unknown-incomplete', 0, 0, 0, 0, ?)",
                        (raw, json.dumps({"digest": digest_hex, "complete": False}, sort_keys=True)),
                    )
                    hardware_ids[digest_hex] = cursor.lastrowid
            return hardware_ids[digest_hex]

        # Resolve candidate and signature names → IDs (cache lookups)
        candidate_cache: dict[str, int] = {}
        signature_cache: dict[str, int | None] = {}

        def _resolve_signature(result: dict[str, Any]) -> int | None:
            signature_hex = result.get("signature", "")
            if len(signature_hex) != 32:
                return None
            if signature_hex in signature_cache:
                return signature_cache[signature_hex]
            canonical = result.get("canonical") or signature_shapes.get(
                signature_hex, {}
            )
            if not isinstance(canonical, dict):
                canonical = {}
            ned = canonical.get("ned", [0, 0, 0, 0])
            ne0 = canonical.get("ne0", [0, 0, 0, 0])
            has_ids = bool(int(canonical.get("flags", 0)) & (1 << 3))
            cursor = connection.execute(
                "SELECT signature_id FROM signature WHERE signature_digest = ?",
                (bytes.fromhex(signature_hex),),
            )
            row = cursor.fetchone()
            if row:
                signature_id = row[0]
            else:
                cursor = connection.execute(
                    "INSERT INTO signature (signature_digest, base_digest, "
                    "schema_version, op, src0_type, src1_type, dst_type, "
                    "m, n, k, canonical_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        bytes.fromhex(signature_hex),
                        bytes.fromhex(signature_hex),
                        canonical.get("schema_version", 1),
                        str(canonical.get("op", "")),
                        str(canonical.get("src0_type", "")),
                        str(canonical.get("src1_type", "")),
                        str(canonical.get("dst_type", "")),
                        ne0[1] if len(ne0) > 1 else 0,
                        ned[2] if has_ids and len(ned) > 2 else ned[1] if len(ned) > 1 else 0,
                        ne0[0] if ne0 else 0,
                        json.dumps(canonical, sort_keys=True, separators=(",", ":")),
                    ),
                )
                signature_id = cursor.lastrowid
            signature_cache[signature_hex] = signature_id
            return signature_id

        def _resolve_candidate(name: str) -> int:
            if name in candidate_cache:
                return candidate_cache[name]

            # Look for existing row
            cursor = connection.execute(
                "SELECT candidate_id FROM candidate WHERE build_id = ? "
                "AND stable_name = ?",
                (build_id, name),
            )
            row = cursor.fetchone()
            if row:
                candidate_cache[name] = row[0]
                return row[0]

            # Insert from manifest or minimal data
            cdata = manifest_by_name.get(name)
            if cdata:
                cursor = connection.execute(
                    "INSERT INTO candidate (build_id, stable_name, family, "
                    "source_class, implementation_version, architectures, "
                    "architecture_mask, graph_safe, deterministic, config_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        build_id,
                        name,
                        cdata["family"],
                        cdata["source_class"],
                        cdata.get("implementation_version", 1),
                        json.dumps(cdata.get("architectures", [])),
                        cdata.get("architecture_mask", 0),
                        int(cdata.get("graph_safe", False)),
                        int(cdata.get("deterministic", True)),
                        json.dumps(cdata.get("config", {})),
                    ),
                )
            else:
                family = _family_from_name(name)
                # Use 'existing_runtime' as fallback — the CHECK constraint
                # rejects 'unknown'. This is a placeholder that signals
                # "no manifest data available" without violating schema.
                cursor = connection.execute(
                    "INSERT INTO candidate (build_id, stable_name, family, "
                    "source_class, implementation_version, architectures, "
                    "architecture_mask, graph_safe, deterministic, config_json) "
                    "VALUES (?, ?, ?, 'existing_runtime', 1, '[]', 0, 0, 1, '{}')",
                    (build_id, name, family),
                )
            candidate_cache[name] = cursor.lastrowid
            return cursor.lastrowid

        # Process each tuning result
        results_inserted = 0
        measurements_inserted = 0

        for result in results:
            hardware_id = _resolve_hardware(result)
            dispatch_hex = result.get("dispatch", "")
            if not dispatch_hex or len(dispatch_hex) != 32:
                raise RecordError("invalid tuning dispatch digest")
            try:
                dispatch_bytes = bytes.fromhex(dispatch_hex)
            except ValueError as exc:
                raise RecordError(f"invalid tuning dispatch digest: {dispatch_hex!r}") from exc
            if len(dispatch_bytes) != 16:
                raise RecordError("tuning dispatch digest must be 16 bytes")

            winner_name = result.get("winner", "")
            improvement_pct = result.get("improvement_pct", 0.0)
            reason = result.get("reason", "")
            confidence = result.get("confidence")  # HI12 E1
            launches_per_sample = result.get("launches_per_sample")  # HI34
            promotion_status = result.get("promotion_status")  # HI34/HI50
            # HI34: only present once tune-promote has run over this row;
            # a raw (unpromoted) measurements file has no q_value yet.
            q_value = result.get("promotion", {}).get("q_value")

            # Map status names to reject reasons
            status_to_reason = {
                "ok": None,
                "architecture": "GGML_HIP_REJECT_ARCHITECTURE",
                "ineligible": "GGML_HIP_REJECT_INELIGIBLE",
                "workspace": "GGML_HIP_REJECT_WORKSPACE",
                "launch_failed": "GGML_HIP_REJECT_LAUNCH_FAILED",
                "nan_inf": "GGML_HIP_REJECT_NAN_INF",
                "tolerance": "GGML_HIP_REJECT_TOLERANCE",
                "unstable": "GGML_HIP_REJECT_UNSTABLE",
                "noisy": "GGML_HIP_REJECT_NOISY",
            }

            signature_id = _resolve_signature(result)

            # Insert measurement rows for each candidate
            for cand in result.get("candidates", []):
                cand_name = cand.get("name", "")
                status = cand.get("status", "ok")
                reject_reason = status_to_reason.get(status)
                accepted = 1 if reject_reason is None else 0

                candidate_id = _resolve_candidate(cand_name)
                median_us = cand.get("median_us")
                gpu_mad_us = cand.get("mad_us")
                p95_us = cand.get("p95_us")
                host_median_us = cand.get("host_median_us")
                nmse = cand.get("nmse")
                max_abs_err = cand.get("max_abs")
                workspace_bytes = cand.get("workspace", 0)
                pool_peak_bytes = cand.get("pool_peak_bytes")
                samples = cand.get("samples", 0)
                effective_us = cand.get("effective_us")  # HI50 ranking metric
                # HI12 E2: raw finalist samples, so a winner is recomputable
                # offline. Absent for screened-only candidates (never a
                # finalist) and when the tuner ran with emit_samples=0.
                samples_us = cand.get("samples_us")
                samples_json = (
                    json.dumps(samples_us, separators=(",", ":"))
                    if samples_us is not None
                    else None
                )

                # Convert hex string to bytes for BLOB column.
                # SQLite's X? syntax is not supported by the Python bindings;
                # pass raw bytes through a regular ? parameter instead.
                connection.execute(
                    "INSERT OR REPLACE INTO measurement (build_id, hardware_id, "
                    "signature_id, dispatch_digest, candidate_id, run_id, objective, stage, "
                    "accepted, reject_reason, samples, launches_per_sample, "
                    "median_us, gpu_mad_us, "
                    "p95_us, host_median_us, workspace_bytes, pool_peak_bytes, nmse, "
                    "max_abs_err, samples_json, effective_us) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'latency', 'final', ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        build_id,
                        hardware_id,
                        signature_id,
                        dispatch_bytes,
                        candidate_id,
                        run_id,
                        accepted,
                        reject_reason,
                        samples,
                        launches_per_sample,
                        median_us,
                        gpu_mad_us,
                        p95_us,
                        host_median_us,
                        workspace_bytes,
                        pool_peak_bytes,
                        nmse,
                        max_abs_err,
                        samples_json,
                        effective_us,
                    ),
                )
                measurements_inserted += 1

            # Insert winner row
            if winner_name:
                winner_id = _resolve_candidate(winner_name)
                # HI34: the tuner now records which candidate was native
                # directly (result["native"]) -- prefer that over guessing
                # from a ":native:vN" name-suffix pattern, which a fresh
                # candidate-naming convention could silently stop matching.
                native_name = result.get("native") or next(
                    (c.get("name", "") for c in result.get("candidates", [])
                     if c.get("name", "").endswith((":native:v1", ":native:v0"))),
                    None,
                )
                is_native = 1 if winner_name == native_name else 0

                # Find winner's measurement for median/p95
                winner_median = None
                winner_p95 = None
                winner_ws = 0
                winner_pool_peak = None
                for cand in result.get("candidates", []):
                    if cand.get("name") == winner_name:
                        winner_median = cand.get("median_us")
                        winner_p95 = cand.get("p95_us")
                        winner_ws = cand.get("workspace", 0)
                        winner_pool_peak = cand.get("pool_peak_bytes")
                        break

                dispatch_bytes = bytes.fromhex(dispatch_hex)

                connection.execute(
                    "INSERT OR REPLACE INTO winner (build_id, hardware_id, signature_id, run_id, objective, "
                    "dispatch_digest, candidate_id, stable_name, "
                    "native_stable_name, is_native, improvement_pct, "
                    "median_us, p95_us, workspace_bytes, pool_peak_bytes, reason, confidence, "
                    "promotion_status, q_value) "
                    "VALUES (?, ?, ?, ?, 'latency', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        build_id,
                        hardware_id,
                        signature_id,
                        run_id,
                        dispatch_bytes,
                        winner_id,
                        winner_name,
                        native_name or "",
                        is_native,
                        improvement_pct,
                        winner_median,
                        winner_p95,
                        winner_ws,
                        winner_pool_peak,
                        reason,
                        confidence,
                        promotion_status,
                        q_value,
                    ),
                )
                results_inserted += 1

        connection.commit()
        return {
            "build_id": build_id,
            "results": results_inserted,
            "measurements": measurements_inserted,
            "candidates": len(candidate_cache),
        }
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bigcherry inventory",
        description="Convert a record-mode JSONL file into an inventory JSON "
        "and a SQLite database, or load tuning measurements.",
    )
    sub = parser.add_subparsers(dest="subcommand")

    # Record mode subcommand (existing behavior)
    rec = sub.add_parser("record", help="Convert record-mode JSONL to inventory + DB")
    rec.add_argument("record", help="JSONL written by GGML_HIP_DISPATCH_DB")
    rec.add_argument(
        "--inventory", default=None, help="inventory JSON to write (default: alongside)"
    )
    rec.add_argument(
        "--database", default=None, help="SQLite database to write (default: alongside)"
    )

    # Tuning mode subcommand
    tune = sub.add_parser("tuning", help="Load tuning measurements into SQLite")
    tune.add_argument(
        "measurements",
        help="JSONL written by GGML_HIP_DISPATCH_DB (the .measurements.jsonl file)",
    )
    tune.add_argument(
        "--database",
        default=None,
        help="SQLite database path (default: alongside measurements)",
    )
    tune.add_argument(
        "--manifest", default=None, help="Manifest JSON for full candidate data"
    )
    tune.add_argument(
        "--signature-source",
        action="append",
        default=[],
        help="JSONL record/replay diagnostics file containing canonical shapes; may be repeated",
    )

    args = parser.parse_args(argv)

    if args.subcommand == "tuning":
        return _cmd_tuning(args)
    elif args.subcommand == "record":
        return _cmd_record(args)
    else:
        # Backward compat: positional arg means record mode
        args.record = args.subcommand  # treat subcommand as record path
        return _cmd_record(args)


def _cmd_record(args) -> int:
    """Existing record-mode command (unchanged behavior)."""
    record_path = Path(args.record)
    if not record_path.is_file():
        print(f"no such record file: {record_path}", file=sys.stderr)
        return 2

    try:
        record = read_jsonl(record_path)
    except RecordError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    inventory = build_inventory(record)
    inventory_path = (
        Path(args.inventory)
        if args.inventory
        else record_path.with_suffix(".inventory.json")
    )
    inventory_path.write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8", newline=""
    )

    database_path = (
        Path(args.database) if args.database else record_path.with_suffix(".sqlite")
    )
    counts = build_database(record, database_path, paths.SQL / "dispatch-db.sql")

    print(f"read {len(record.observations)} observation(s) from {record_path}")
    print(f"  types: mmq={inventory['mmq_types']} mmvq={inventory['mmvq_types']}")
    print(f"         mmvf={inventory['mmvf_types']} mmf={inventory['mmf_types']}")
    print(f"  widths: {inventory['widths']}")
    print(f"  blas observed: {inventory['uses_blas']}")
    print(f"  inventory: {inventory_path}")
    print(
        f"  database:  {database_path} "
        f"({counts['signatures']} signatures, {counts['hardware']} hardware)"
    )
    return 0


def _cmd_tuning(args) -> int:
    """Tuning measurements loader."""
    meas_path = Path(args.measurements)
    if not meas_path.is_file():
        print(f"no such measurements file: {meas_path}", file=sys.stderr)
        return 2

    db_path = Path(args.database) if args.database else meas_path.with_suffix(".sqlite")
    manifest_path = Path(args.manifest) if args.manifest else None

    counts = load_measurements(
        meas_path,
        db_path,
        paths.SQL / "dispatch-db.sql",
        manifest_path=manifest_path,
        signature_source_paths=[Path(p) for p in args.signature_source],
    )

    print(
        f"loaded {counts['results']} result(s) with "
        f"{counts['measurements']} measurement(s) and "
        f"{counts['candidates']} candidate(s) into {db_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
