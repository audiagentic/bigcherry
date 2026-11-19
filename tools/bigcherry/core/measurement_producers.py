"""MRC06 shared producer-to-MRC01 adapter: content only, no legacy writes.

Engine-specific producers supply strictly measured identity, metrics and inputs.
This adapter neither infers missing provenance nor reads legacy result indexes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from . import engines, run_records, calibration_records
from . import host_facts as host_capture


def canonical_blob(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def utc_stamp(value: str) -> str:
    """Normalise known producer times; refuse unparseable dates rather than guess."""
    dt = datetime.fromisoformat(value)
    if dt.utcoffset() is None or dt.utcoffset().total_seconds() != 0:
        raise run_records.RecordError("recorded_at must be UTC")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def publish(
    *,
    work_root: Path | None,
    run_id: str,
    recorded_at: str,
    identity: dict[str, Any],
    workload: dict[str, Any],
    metrics: dict[str, dict[str, float | int]],
    effective_config: bytes,
    argv: list[str],
    environment: dict[str, Any],
    power_settings: dict[str, Any],
    inputs: dict[str, bytes | Path],
    host_facts: dict[str, Any],
    feature_states: list[dict],
    metric_specs: dict | None = None,
    validity: dict | None = None,
    session_id: str | None = None,
    calibration_selection: dict[str, Any] | None = None,
) -> str:
    """Publish one complete immutable record, verifying every source blob."""
    if not isinstance(effective_config, bytes) or not effective_config:
        raise run_records.RecordError("effective manifest/config bytes are required")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(a, str) for a in argv)
    ):
        raise run_records.RecordError("actual argv required")
    if any(not isinstance(v, (bytes, Path)) for v in inputs.values()):
        raise run_records.RecordError("provenance inputs must be exact bytes or readable files")
    # Radiance requires a sealed start receipt; other engines are MRC07 work.
    recorded_time = utc_stamp(recorded_at)
    if identity["engine"] == engines.RADIANCE:
        try:
            host_capture.validate_start_snapshot(host_facts)
            start_time = utc_stamp(host_facts["captured_at"])
        except (ValueError, TypeError, KeyError) as exc:
            raise run_records.RecordError(f"missing or invalid start host facts: {exc}") from exc
        if start_time > recorded_time:
            raise run_records.RecordError("start host-facts timestamp is after the measured record time")
    store = run_records.RunRecordStore(work_root)

    def source_blob(source: bytes | Path) -> dict:
        if isinstance(source, bytes):
            return store.put_blob(source)
        if source.is_symlink() or not source.is_file():
            raise run_records.RecordError(f"unsafe provenance source: {source}")
        return store.put_blob_file(source)
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "recorded_at": recorded_time,
        "identity": identity,
        "workload": workload,
        "metrics": metrics,
        "provenance": {
            "effective_config": store.put_blob(effective_config),
            "argv": store.put_blob(canonical_blob(argv)),
            "environment": store.put_blob(canonical_blob(environment)),
            "power_settings": store.put_blob(canonical_blob(power_settings)),
            "inputs": [
                {"role": key, "blob": source_blob(raw)}
                for key, raw in sorted(inputs.items())
            ],
        },
        "host_facts": host_facts,
        "feature_states": feature_states,
    }
    if calibration_selection is not None:
        if not isinstance(calibration_selection, dict):
            raise run_records.RecordError("calibration selection must be an object")
        status = calibration_selection.get("status")
        if status == "selected":
            if set(calibration_selection) != {"status", "record", "quantity_blobs"}:
                raise run_records.RecordError("calibration selected needs record and quantities")
            source = calibration_records.validate_record(calibration_selection["record"])
            quantities = calibration_selection["quantity_blobs"]
            if not isinstance(quantities, dict):
                raise run_records.RecordError("calibration quantities must be named")
            record["calibration_record"] = {
                "status": "selected", "kind": source["kind"],
                "key_sha256": source["key"]["sha256"],
                "record_blob": store.put_blob(canonical_blob(source)),
                "quantity_blobs": [{"name": name, "blob": source_blob(raw)}
                                   for name, raw in sorted(quantities.items())],
            }
        elif status == "none":
            record["calibration_record"] = dict(calibration_selection)
        else:
            raise run_records.RecordError("unknown calibration selection status")
    if metric_specs is not None:
        record["metric_specs"] = metric_specs
    if validity is not None:
        record["validity"] = validity
    if session_id:
        record["session_id"] = session_id
    digest = store.put_record(record)
    if store.get_record(digest) != record:
        raise run_records.RecordError("run-record content verification failed")
    return digest
