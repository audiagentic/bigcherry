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
import json
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths


class RecordError(RuntimeError):
    pass


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

        # The dst width is what the width-parameterised families are compiled
        # for; ned[1] is the device-local column count.
        ned = canonical.get("ned")
        if isinstance(ned, list) and len(ned) > 1:
            width = int(ned[1])
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
        header = record.header

        cursor = connection.execute(
            "INSERT INTO build (source_revision, source_dirty, manifest_hash, "
            "signature_schema, hardware_schema, variant_set) "
            "VALUES (?, 0, ?, ?, ?, ?)",
            (
                header.get("source_revision", ""),
                header.get("manifest_hash", ""),
                header.get("signature_schema", 1),
                header.get("hardware_schema", 1),
                header.get("variant_set", "inventory"),
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
                        ned[1] if len(ned) > 1 else 0,
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
) -> dict[str, int]:
    """Load tuning measurements JSONL into SQLite.

    Reads the .measurements.jsonl written by the tuning engine (HI12)
    and populates build, candidate, measurement, and winner tables.

    The dispatch_digest is the primary lookup key. signature_id is NULL
    when only the dispatch digest is available — it can be linked later
    if the DB was also built from record-mode observations.

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
            except json.JSONDecodeError:
                print(
                    f"warning: {measurements_path.name} line {number} is "
                    f"truncated or malformed; ignoring it",
                    file=sys.stderr,
                )
                break
            kind = row.get("kind")
            if kind == "header":
                header = row
            elif kind == "result":
                results.append(row)

    if header is None:
        raise RecordError(
            f"{measurements_path}: no header line. Either the file is not a "
            f"bigcherry measurements file, or the run died before its first "
            f"flush."
        )

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
        connection.execute("PRAGMA foreign_keys = ON")

        # Find or create build row
        source_revision = header.get("source_revision", "")
        manifest_hash = header.get("manifest_hash", "")
        artifact_version = header.get("artifact_version", 1)

        cursor = connection.execute(
            "SELECT build_id FROM build WHERE source_revision = ? "
            "AND manifest_hash = ?",
            (source_revision, manifest_hash),
        )
        build_row = cursor.fetchone()
        if build_row:
            build_id = build_row[0]
        else:
            cursor = connection.execute(
                "INSERT INTO build (source_revision, source_dirty, "
                "manifest_hash, signature_schema, hardware_schema, variant_set, "
                "dispatch_abi) VALUES (?, 0, ?, 1, 1, ?, ?)",
                (source_revision, manifest_hash, "tuning", str(artifact_version)),
            )
            build_id = cursor.lastrowid

        # Find hardware row (use placeholder if not from record mode)
        # The measurements JSONL doesn't include hardware info directly.
        # We look for any existing hardware row or create a default.
        cursor = connection.execute("SELECT hardware_id FROM hardware LIMIT 1")
        hw_row = cursor.fetchone()
        if hw_row:
            hardware_id = hw_row[0]
        else:
            # Create placeholder — user can update later or from record-mode DB
            cursor = connection.execute(
                "INSERT INTO hardware (hardware_digest, architecture, "
                "architecture_code, wave_size, compute_units, feature_flags, "
                "canonical_json) VALUES (X'00000000000000000000000000000000', "
                "'unknown', 0, 64, 1, 0, '{}')"
            )
            hardware_id = cursor.lastrowid

        # Resolve candidate name → ID (cache lookups)
        candidate_cache: dict[str, int] = {}

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
            dispatch_hex = result.get("dispatch", "")
            if not dispatch_hex or len(dispatch_hex) != 32:
                continue

            winner_name = result.get("winner", "")
            improvement_pct = result.get("improvement_pct", 0.0)
            reason = result.get("reason", "")

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
            }

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
                samples = cand.get("samples", 0)

                # Convert hex string to bytes for BLOB column.
                # SQLite's X? syntax is not supported by the Python bindings;
                # pass raw bytes through a regular ? parameter instead.
                dispatch_bytes = bytes.fromhex(dispatch_hex)

                connection.execute(
                    "INSERT INTO measurement (build_id, hardware_id, "
                    "dispatch_digest, candidate_id, objective, stage, "
                    "accepted, reject_reason, samples, median_us, gpu_mad_us, "
                    "p95_us, host_median_us, workspace_bytes, nmse, "
                    "max_abs_err) "
                    "VALUES (?, ?, ?, ?, 'latency', 'final', ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?)",
                    (
                        build_id,
                        hardware_id,
                        dispatch_bytes,
                        candidate_id,
                        accepted,
                        reject_reason,
                        samples,
                        median_us,
                        gpu_mad_us,
                        p95_us,
                        host_median_us,
                        workspace_bytes,
                        nmse,
                        max_abs_err,
                    ),
                )
                measurements_inserted += 1

            # Insert winner row
            if winner_name:
                winner_id = _resolve_candidate(winner_name)
                native_name = None
                for cand in result.get("candidates", []):
                    cname = cand.get("name", "")
                    if cname.endswith((":native:v1", ":native:v0")):
                        native_name = cname
                        break
                is_native = 1 if winner_name == native_name else 0

                # Find winner's measurement for median/p95
                winner_median = None
                winner_p95 = None
                winner_ws = 0
                for cand in result.get("candidates", []):
                    if cand.get("name") == winner_name:
                        winner_median = cand.get("median_us")
                        winner_p95 = cand.get("p95_us")
                        winner_ws = cand.get("workspace", 0)
                        break

                dispatch_bytes = bytes.fromhex(dispatch_hex)

                connection.execute(
                    "INSERT INTO winner (build_id, hardware_id, objective, "
                    "dispatch_digest, candidate_id, stable_name, "
                    "native_stable_name, is_native, improvement_pct, "
                    "median_us, p95_us, workspace_bytes, reason) "
                    "VALUES (?, ?, 'latency', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        build_id,
                        hardware_id,
                        dispatch_bytes,
                        winner_id,
                        winner_name,
                        native_name or "",
                        is_native,
                        improvement_pct,
                        winner_median,
                        winner_p95,
                        winner_ws,
                        reason,
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
    )

    print(
        f"loaded {counts['results']} result(s) with "
        f"{counts['measurements']} measurement(s) and "
        f"{counts['candidates']} candidate(s) into {db_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
