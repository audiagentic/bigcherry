"""Platform-owned MRC03 reader for native feature-states.v1.json files.

Engines emit this JSON file without Python. The platform validates its exact
contract and copies the immutable start snapshot into a v1 run record. No
producer migration or engine-specific parsing lives here.
"""

from __future__ import annotations

import json
import os
import stat
from copy import deepcopy
from pathlib import Path
from typing import Any

from . import run_records

SCHEMA_PATH = Path(__file__).parent / "schemas" / "feature-state-v1.schema.json"
FILE_NAME = "feature-states.v1.json"
MAX_BYTES = 1024 * 1024


class FeatureStateError(ValueError):
    """A native feature-state snapshot is invalid, unsafe, or inconsistent."""


def load_schema() -> dict[str, Any]:
    """Standalone portable JSON Schema for a native feature-state array."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_feature_states(states: Any) -> list[dict[str, Any]]:
    schema = load_schema()
    if schema["items"] != run_records.load_schema()["$defs"]["feature_state"]:
        raise FeatureStateError(
            "platform run-record and native feature-state schemas diverge"
        )
    try:
        run_records._check_schema(states, schema, schema, "feature_states")
        run_records._json_bytes(states)  # exclude NaN/Infinity and non-JSON objects
    except run_records.RecordError as exc:
        raise FeatureStateError(str(exc)) from exc
    seen: set[tuple[str, str]] = set()
    for item in states:
        key = (item["engine"], item["feature_id"])
        if key in seen:
            raise FeatureStateError(f"duplicate engine feature: {key!r}")
        seen.add(key)
    return states


def read_feature_states(path: Path | str) -> list[dict[str, Any]]:
    """Read a single, regular, non-symlinked feature-state file, with a size bound."""
    path = Path(path)
    fd = -1
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise FeatureStateError("feature-state file is not a regular file")
        if os.fstat(fd).st_size > MAX_BYTES:
            raise FeatureStateError("feature-state file exceeds the v1 size limit")
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise FeatureStateError("feature-state file exceeds the v1 size limit")
    except OSError as exc:
        raise FeatureStateError(
            f"feature-state file unreadable or unsafe: {exc}"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
    try:
        states = json.loads(
            raw,
            object_pairs_hook=run_records._pairs_no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FeatureStateError(f"invalid JSON constant {value}")
            ),
        )
    except (json.JSONDecodeError, UnicodeError, run_records.RecordError) as exc:
        raise FeatureStateError(f"invalid feature-state JSON: {exc}") from exc
    return validate_feature_states(states)


def with_feature_states(record: dict[str, Any], path: Path | str) -> dict[str, Any]:
    """Copy the native snapshot into a run record; do not modify the input.

    Engine identity is checked before persistence. The run's other immutable
    measured values and blob references are untouched.
    """
    run_records.validate_record(record)
    states = read_feature_states(path)
    engine = record["identity"]["engine"]
    if any(item["engine"] != engine for item in states):
        raise FeatureStateError(
            f"feature-state engine differs from run-record engine {engine}"
        )
    result = deepcopy(record)
    result["feature_states"] = deepcopy(states)
    run_records.validate_record(result)
    return result
