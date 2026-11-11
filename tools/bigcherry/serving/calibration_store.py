"""Inactive S13 P2: scoped, immutable S8 calibration evidence.

The engine's S8 resolver is the admission authority. This module accepts
only measured run output; it does not generate calibration defaults or
activate production serving.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SCOPE_DIGESTS = ("model_digest", "engine_build_digest", "helper_build_digest",
                 "layout_digest", "inventory_digest", "owner_map_digest")
SCOPE_FIELDS = frozenset((*SCOPE_DIGESTS, "context_target_tokens",
                          "retention_target_tokens", "kv_bytes_per_token"))
RECORD_FIELDS = frozenset(("key", "value", "evidence_id", "raw_run", "scope",
                          "measured_at_unix_s", "valid_until_unix_s",
                          "measured_candidates", "method", "source", "recorded_date"))
KEYS = frozenset(("host_ledger_margin.mib", "handoff.max_rows", "handoff.small_rows",
                  "handoff.gemv_large_rows", "handoff.moe_down_tile",
                  "handoff.histogram_period_s", "handoff.live_cadence_ns"))
MEASUREMENT_FIELDS = frozenset(("schema_version", "kind", "key", "scope", "raw_run",
                                "evidence_id", "method", "measured_at_unix_s",
                                "valid_until_unix_s", "source", "selected_value", "samples"))
MAX_INT = 9007199254740991


class CalibrationInvalid(ValueError):
    """Calibration evidence is incomplete, expired, duplicated or out of scope."""


def _object(value, fields, title):
    if not isinstance(value, dict) or set(value) != fields:
        raise CalibrationInvalid(f"{title}: missing or unexpected fields")
    return value


def _int(value, positive=False):
    return type(value) is int and (value > 0 if positive else value >= 0) and value <= MAX_INT


def _text(value, title):
    if not isinstance(value, str) or not value.strip():
        raise CalibrationInvalid(f"{title}: nonempty text required")
    return value


def validate_scope(value):
    scope = _object(value, SCOPE_FIELDS, "scope")
    if any(not isinstance(scope[k], str) or not re.fullmatch("[0-9a-f]{64}", scope[k])
           for k in SCOPE_DIGESTS):
        raise CalibrationInvalid("scope: invalid model/build/layout/inventory/owner digest")
    if any(not _int(scope[k], True) for k in
           ("context_target_tokens", "retention_target_tokens", "kv_bytes_per_token")):
        raise CalibrationInvalid("scope: positive context, retention and measured KV bytes/token required")
    return scope


def validate_record(value, *, scope=None, now_unix_s=None):
    row = _object(value, RECORD_FIELDS, "S8 record")
    if row["key"] not in KEYS:
        raise CalibrationInvalid("unknown S8 calibration key")
    if not _int(row["value"]):
        raise CalibrationInvalid("invalid calibration value")
    for field in ("evidence_id", "raw_run", "method"):
        _text(row[field], field)
    validate_scope(row["scope"])
    if scope is not None and row["scope"] != validate_scope(scope):
        raise CalibrationInvalid("calibration scope mismatch")
    if row["source"] not in ("cache", "seed"):
        raise CalibrationInvalid("S8 source must be cache or seed")
    candidates = row["measured_candidates"]
    if (not isinstance(candidates, list) or not candidates or
            any(not _int(c) for c in candidates) or
            len(set(candidates)) != len(candidates) or row["value"] not in candidates):
        raise CalibrationInvalid("selected value must occur in distinct measured candidates")
    start, end = row["measured_at_unix_s"], row["valid_until_unix_s"]
    if not _int(start, True) or not _int(end, True) or end <= start:
        raise CalibrationInvalid("invalid measurement validity interval")
    if now_unix_s is not None:
        if not _int(now_unix_s, True) or start > now_unix_s or end <= now_unix_s:
            raise CalibrationInvalid("stale or future-dated calibration")
    try:
        date = datetime.strptime(row["recorded_date"], "%Y-%m-%d").date()
        if date.isoformat() != row["recorded_date"]:
            raise ValueError("noncanonical")
    except (TypeError, ValueError) as exc:
        raise CalibrationInvalid("invalid recorded ISO date") from exc
    return row


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _read(path):
    def unique(pairs):
        output = {}
        for key, val in pairs:
            if key in output:
                raise CalibrationInvalid(f"duplicate JSON field {key}")
            output[key] = val
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(
                          CalibrationInvalid(f"invalid JSON constant {x}")))


def emit_from_run(run_file: Path, store: Path) -> Path:
    """Derive S8Calibration solely from run output with per-candidate evidence refs."""
    run = _object(_read(run_file), MEASUREMENT_FIELDS, "measurement run")
    if type(run["schema_version"]) is not int or run["schema_version"] != 1 or (
            run["kind"] != "s8_calibration_measurement"):
        raise CalibrationInvalid("not a measured S8 calibration run")
    samples = run["samples"]
    if not isinstance(samples, list) or not samples:
        raise CalibrationInvalid("measurement requires recorded samples")
    candidates = []
    for sample in samples:
        _object(sample, frozenset(("candidate", "evidence_ref")), "sample")
        if not _int(sample["candidate"]):
            raise CalibrationInvalid("sample requires measured integer candidate")
        _text(sample["evidence_ref"], "sample evidence_ref")
        candidates.append(sample["candidate"])
    if len(set(candidates)) != len(candidates):
        raise CalibrationInvalid("duplicate candidate")
    if not _int(run["measured_at_unix_s"], True):
        raise CalibrationInvalid("measurement timestamp missing")
    record = {
        "key": run["key"], "value": run["selected_value"],
        "evidence_id": run["evidence_id"], "raw_run": run["raw_run"],
        "scope": run["scope"], "measured_at_unix_s": run["measured_at_unix_s"],
        "valid_until_unix_s": run["valid_until_unix_s"],
        "measured_candidates": candidates, "method": run["method"],
        "source": run["source"], "recorded_date": datetime.fromtimestamp(
            run["measured_at_unix_s"], timezone.utc).date().isoformat(),
    }
    validate_record(record)
    # The filename binds evidence to the unaltered input run bytes. Records
    # are written once: a later run cannot silently rewrite a decision.
    digest = hashlib.sha256(canonical(run)).hexdigest()
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    target = store / f"{digest}.json"
    if target.exists():
        if _read(target) != record:
            raise CalibrationInvalid("same evidence hash has conflicting record")
        return target
    with target.open("x", encoding="utf-8") as out:
        out.write(canonical(record).decode("utf-8") + "\n")
    return target


def select_records(store: Path, *, scope: dict, required_keys, now_unix_s: int):
    """Return one fresh exact-scope S8 record per mandatory key; no fallback."""
    validate_scope(scope)
    keys = tuple(required_keys)
    if len(set(keys)) != len(keys) or any(key not in KEYS for key in keys):
        raise CalibrationInvalid("invalid required calibration keys")
    found = {}
    for path in sorted(Path(store).glob("*.json")):
        record = _read(path)
        if not isinstance(record, dict) or record.get("key") not in keys:
            continue
        if record.get("scope") != scope:
            continue
        validate_record(record, scope=scope, now_unix_s=now_unix_s)
        key = record["key"]
        if key in found:
            raise CalibrationInvalid(f"ambiguous calibration {key}")
        found[key] = record
    missing = set(keys) - found.keys()
    if missing:
        raise CalibrationInvalid("missing S8 calibration: " + ", ".join(sorted(missing)))
    return tuple(found[key] for key in keys)


def main(argv=None):
    parser = argparse.ArgumentParser(description="S8 measured calibration record producer (inactive)")
    parser.add_argument("action", choices=("emit",))
    parser.add_argument("--run", type=Path, required=True, help="measured run JSON")
    parser.add_argument("--store", type=Path, required=True, help="immutable record directory")
    args = parser.parse_args(argv)
    try:
        print(emit_from_run(args.run, args.store))
        return 0
    except (CalibrationInvalid, OSError, ValueError, OverflowError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
