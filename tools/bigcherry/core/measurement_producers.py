"""MRC06 shared producer-to-MRC01 adapter: content only, no legacy writes.

Engine-specific producers supply strictly measured identity, metrics and inputs.
This adapter neither infers missing provenance nor reads legacy result indexes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from . import run_records


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
    inputs: dict[str, bytes],
    host_facts: dict[str, Any],
    feature_states: list[dict],
    metric_specs: dict | None = None,
    validity: dict | None = None,
    session_id: str | None = None,
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
    if any(not isinstance(v, bytes) for v in inputs.values()):
        raise run_records.RecordError("provenance input contents must be bytes")
    store = run_records.RunRecordStore(work_root)
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "recorded_at": utc_stamp(recorded_at),
        "identity": identity,
        "workload": workload,
        "metrics": metrics,
        "provenance": {
            "effective_config": store.put_blob(effective_config),
            "argv": store.put_blob(canonical_blob(argv)),
            "environment": store.put_blob(canonical_blob(environment)),
            "power_settings": store.put_blob(canonical_blob(power_settings)),
            "inputs": [
                {"role": key, "blob": store.put_blob(raw)}
                for key, raw in sorted(inputs.items())
            ],
        },
        "host_facts": host_facts,
        "feature_states": feature_states,
    }
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
