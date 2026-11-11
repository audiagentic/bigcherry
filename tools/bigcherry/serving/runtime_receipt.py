"""Inactive S13 runtime receipt consumer for S13 display.

This parses the typed durable receipt that separates the immutable S10 v2
manifest ``spec.handoff.placement.live_policy_state`` startup decision from
actual runtime policy degradation / placement-stopped evidence. Missing
runtime telemetry is never reported as successful runtime policy.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_MAX_WIRE = (1 << 53) - 1
_DIGEST = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")
_MAX_DIGEST = (1 << 64) - 1
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")

_LIVE_POLICY_FIELDS = frozenset((
    "requested", "configured", "reason_code", "reason_detail",
    "calibration_key", "record_identity", "effect_gate",
    "histogram_period_s", "cadence_ns", "max_heat_age_ns",
))
_SNAPSHOT_FIELDS = frozenset((
    "manifest_digest", "epoch", "owner_generation_digest",
    "owner_generation_digest_hex", "state", "fault_code", "fault_actor",
    "transaction_id", "cleanup_fault", "cleanup_actor",
    "placement_stopped", "degraded",
))
_SNAPSHOT_REQUIRED = _SNAPSHOT_FIELDS - {"owner_generation_digest_hex"}
_RECEIPT_FIELDS = frozenset((
    "schema_version", "kind", "run_id", "manifest", "startup_decision",
    "bounding", "runtime_status", "placement_stopped", "degraded",
    "bounded", "max_snapshots", "snapshot_count", "snapshots",
    "required_producer", "provenance",
))
_RECEIPT_REQUIRED = _RECEIPT_FIELDS - {"provenance"}


class RuntimeReceiptRefused(Exception):
    """Raised when a runtime receipt is malformed or overclaims success."""


@dataclass(frozen=True)
class RuntimeSnapshot:
    manifest_digest: str
    epoch: int
    owner_generation_digest: int
    state: int = 0
    fault_code: int = 0
    fault_actor: int = 0
    transaction_id: int = 0
    cleanup_fault: int = 0
    cleanup_actor: int = 0
    placement_stopped: bool = False
    degraded: bool = False
    owner_generation_digest_hex: str = ""


@dataclass(frozen=True)
class RuntimeReceipt:
    schema_version: int
    run_id: str = ""
    manifest_digest: str = ""
    startup_decision: dict = field(default_factory=dict)
    bounding_manifest_digest: str = ""
    epoch: int = 0
    owner_generation_digest: int = 0
    runtime_status: str = "missing_evidence"
    placement_stopped: bool = False
    degraded: bool = False
    bounded: bool = True
    max_snapshots: int = 2
    snapshots: tuple[RuntimeSnapshot, ...] = ()
    required_producer: str = ""
    provenance: str = ""


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeReceiptRefused(f"runtime receipt duplicate JSON member {key}")
        value[key] = item
    return value


def _no_fraction(_):
    raise RuntimeReceiptRefused("runtime receipt only I-JSON-safe integer numbers are valid")


def _digest(value, name):
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise RuntimeReceiptRefused(f"runtime receipt {name}: manifest/owner digest required")
    return value


def _int(value, name, *, positive=False, allow_zero=True):
    if type(value) is not int or not (0 <= value <= _MAX_WIRE):
        raise RuntimeReceiptRefused(f"runtime receipt {name}: wire integer required")
    if positive and not value:
        raise RuntimeReceiptRefused(f"runtime receipt {name}: positive integer required")
    if not allow_zero and value == 0:
        raise RuntimeReceiptRefused(f"runtime receipt {name}: nonzero integer required")
    return value


def _owner_digest(value, name):
    if type(value) is not int or not (1 <= value <= _MAX_DIGEST):
        raise RuntimeReceiptRefused(f"runtime receipt {name}: uint64 digest required")
    return value


def _bool(value, name):
    if type(value) is not bool:
        raise RuntimeReceiptRefused(f"runtime receipt {name}: boolean required")
    return value


def _object(raw, fields, where):
    if not isinstance(raw, dict) or set(raw) != fields:
        raise RuntimeReceiptRefused(f"runtime receipt {where}: missing/unknown fields")
    return raw


def decode_runtime_receipt(raw: Any, *, require_telemetry: bool = False,
                           expected_run_id: str | None = None,
                           expected_manifest_digest: str | None = None) -> RuntimeReceipt:
    if not isinstance(raw, dict) or not _RECEIPT_REQUIRED.issubset(raw) or not set(raw).issubset(_RECEIPT_FIELDS):
        raise RuntimeReceiptRefused("runtime receipt record: missing/unknown fields")
    record = raw
    if (record["schema_version"] != 1 or type(record["schema_version"]) is not int
            or record["kind"] != "bc_serve_v2_runtime_receipt"):
        raise RuntimeReceiptRefused("runtime receipt has wrong identity")
    run_id = record["run_id"]
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise RuntimeReceiptRefused("runtime receipt run_id is invalid")
    if expected_run_id is not None and run_id != expected_run_id:
        raise RuntimeReceiptRefused("runtime receipt belongs to another run")
    manifest = _object(record["manifest"], frozenset(("content_digest",)), "manifest")
    manifest_digest = _digest(manifest["content_digest"], "manifest.content_digest")

    if expected_manifest_digest is not None and manifest_digest != _digest(
            expected_manifest_digest, "expected_manifest_digest"):
        raise RuntimeReceiptRefused("runtime receipt belongs to another manifest")
    startup = record["startup_decision"]
    if not isinstance(startup, dict) or not set(startup).issubset(_LIVE_POLICY_FIELDS):
        raise RuntimeReceiptRefused("runtime receipt startup_decision is invalid")
    # Required startup fields must all be present.
    if set(startup) != _LIVE_POLICY_FIELDS:
        raise RuntimeReceiptRefused("runtime receipt startup_decision missing/unknown fields")
    if startup["requested"] not in ("on", "off", "not_applicable"):
        raise RuntimeReceiptRefused("runtime receipt requested must be on/off/not_applicable")
    _bool(startup["configured"], "startup_decision.configured")
    if startup["configured"] and startup["requested"] != "on":
        raise RuntimeReceiptRefused("runtime receipt configured startup requires requested=on")
    if startup["configured"] and startup["reason_code"] != "configured":
        raise RuntimeReceiptRefused("runtime receipt configured startup requires reason_code=configured")
    if (not startup["configured"] and startup["requested"] == "on"
            and startup["reason_code"] == "configured"):
        raise RuntimeReceiptRefused("runtime receipt configured=true required when configured")
    if startup["requested"] in ("off", "not_applicable") and startup["configured"]:
        raise RuntimeReceiptRefused("runtime receipt configured must be false when policy is off")
    if not isinstance(startup["reason_code"], str) or not startup["reason_code"]:
        raise RuntimeReceiptRefused("runtime receipt reason_code required")
    for key in ("histogram_period_s", "cadence_ns", "max_heat_age_ns"):
        _int(startup[key], f"startup_decision.{key}")

    bounding = _object(record["bounding"], frozenset(("manifest_digest", "epoch",
                                                     "owner_generation_digest")),
                       "bounding")
    bounding_manifest_digest = _digest(bounding["manifest_digest"], "bounding.manifest_digest")
    if bounding_manifest_digest != manifest_digest:
        raise RuntimeReceiptRefused("runtime receipt manifest digest disagrees between blocks")
    epoch = _int(bounding["epoch"], "bounding.epoch", positive=True)
    owner_generation_digest = _owner_digest(bounding["owner_generation_digest"],
                                           "bounding.owner_generation_digest")

    runtime_status = record["runtime_status"]
    if runtime_status not in ("active", "transitioning", "degraded", "placement_stopped", "missing_evidence"):
        raise RuntimeReceiptRefused("runtime receipt invalid runtime_status")
    placement_stopped = _bool(record["placement_stopped"], "placement_stopped")
    degraded = _bool(record["degraded"], "degraded")
    bounded = _bool(record["bounded"], "bounded")
    if not bounded:
        raise RuntimeReceiptRefused("runtime receipt must be bounded")
    max_snapshots = _int(record["max_snapshots"], "max_snapshots", positive=True)
    if max_snapshots != 2:
        raise RuntimeReceiptRefused("runtime receipt snapshot bound must be 2")
    snapshot_count = _int(record["snapshot_count"], "snapshot_count")
    if snapshot_count > max_snapshots:
        raise RuntimeReceiptRefused("runtime receipt exceeds bounded snapshot count")

    snapshots_raw = record["snapshots"]
    if not isinstance(snapshots_raw, list) or len(snapshots_raw) != snapshot_count:
        raise RuntimeReceiptRefused("runtime receipt snapshot_count disagrees")
    snapshots = []
    for item in snapshots_raw:
        if not isinstance(item, dict) or not _SNAPSHOT_REQUIRED.issubset(item) or not set(item).issubset(_SNAPSHOT_FIELDS):
            raise RuntimeReceiptRefused("runtime receipt snapshot: missing/unknown fields")
        snap = item
        snap_manifest = _digest(snap["manifest_digest"], "snapshot.manifest_digest")
        if snap_manifest != manifest_digest:
            raise RuntimeReceiptRefused("runtime receipt snapshot manifest digest differs")
        snap_epoch = _int(snap["epoch"], "snapshot.epoch", positive=True)
        if snap_epoch != epoch and runtime_status != "missing_evidence":
            # Snapshots are bounded to the same epoch; a differing epoch is a
            # different runtime generation and cannot be mixed into this receipt.
            raise RuntimeReceiptRefused("runtime receipt snapshot epoch differs from bounding epoch")
        snap_owner = _owner_digest(snap["owner_generation_digest"],
                                   "snapshot.owner_generation_digest")
        if snap_owner != owner_generation_digest:
            raise RuntimeReceiptRefused("runtime receipt snapshot owner-generation digest differs")
        snap_placement = _bool(snap["placement_stopped"], "snapshot.placement_stopped")
        snap_degraded = _bool(snap["degraded"], "snapshot.degraded")
        snapshots.append(RuntimeSnapshot(
            manifest_digest=snap_manifest,
            epoch=snap_epoch,
            owner_generation_digest=snap_owner,
            state=_int(snap["state"], "snapshot.state"),
            fault_code=_int(snap["fault_code"], "snapshot.fault_code"),
            fault_actor=_int(snap["fault_actor"], "snapshot.fault_actor"),
            transaction_id=_int(snap["transaction_id"], "snapshot.transaction_id"),
            cleanup_fault=_int(snap["cleanup_fault"], "snapshot.cleanup_fault"),
            cleanup_actor=_int(snap["cleanup_actor"], "snapshot.cleanup_actor"),
            placement_stopped=snap_placement,
            degraded=snap_degraded,
            owner_generation_digest_hex=str(snap.get("owner_generation_digest_hex", "")),
        ))

    # Fail closed: missing telemetry may not be reported as successful.
    if not snapshots and runtime_status == "active":
        raise RuntimeReceiptRefused("runtime receipt claims active runtime without telemetry")
    if require_telemetry and not snapshots:
        raise RuntimeReceiptRefused("runtime receipt requires authoritative telemetry")

    # Consistency: top-level status/flags must agree with bounded snapshots.
    snap_placement = any(s.placement_stopped for s in snapshots)
    snap_degraded = any(s.degraded for s in snapshots)
    if placement_stopped != snap_placement and snapshots:
        raise RuntimeReceiptRefused("runtime receipt placement_stopped disagrees with snapshots")
    if degraded != snap_degraded and snapshots:
        raise RuntimeReceiptRefused("runtime receipt degraded disagrees with snapshots")
    for s in snapshots:
        if s.placement_stopped != (s.cleanup_fault != 0):
            raise RuntimeReceiptRefused("runtime receipt placement_stopped must follow cleanup_fault")
    if snapshots:
        if placement_stopped:
            if runtime_status != "placement_stopped":
                raise RuntimeReceiptRefused("runtime receipt runtime_status disagrees with placement_stopped")
        elif degraded:
            if runtime_status != "degraded":
                raise RuntimeReceiptRefused("runtime receipt runtime_status disagrees with degraded")
        else:
            expected = "transitioning" if any(s.state != 0 for s in snapshots) else "active"
            if runtime_status != expected:
                raise RuntimeReceiptRefused("runtime receipt runtime_status disagrees with telemetry")
    else:
        if runtime_status not in ("missing_evidence",):
            raise RuntimeReceiptRefused("runtime receipt without telemetry cannot claim runtime status")

    required_producer = record["required_producer"]
    if not isinstance(required_producer, str) or not required_producer.strip():
        raise RuntimeReceiptRefused("runtime receipt required_producer is missing")
    provenance = record.get("provenance", "")
    if provenance and not isinstance(provenance, str):
        raise RuntimeReceiptRefused("runtime receipt provenance must be a string")

    return RuntimeReceipt(
        schema_version=1,
        run_id=run_id,
        manifest_digest=manifest_digest,
        startup_decision=dict(startup),
        bounding_manifest_digest=bounding_manifest_digest,
        epoch=epoch,
        owner_generation_digest=owner_generation_digest,
        runtime_status=runtime_status,
        placement_stopped=placement_stopped,
        degraded=degraded,
        bounded=bounded,
        max_snapshots=max_snapshots,
        snapshots=tuple(snapshots),
        required_producer=required_producer,
        provenance=provenance,
    )


def parse_runtime_receipt(text: str, *, require_telemetry: bool = False,
                          expected_run_id: str | None = None,
                          expected_manifest_digest: str | None = None) -> RuntimeReceipt:
    try:
        raw = json.loads(text, object_pairs_hook=_pairs, parse_float=_no_fraction,
                         parse_constant=_no_fraction)
    except (ValueError, TypeError) as exc:
        raise RuntimeReceiptRefused("runtime receipt invalid JSON") from exc
    return decode_runtime_receipt(
        raw, require_telemetry=require_telemetry, expected_run_id=expected_run_id,
        expected_manifest_digest=expected_manifest_digest)


def validate_runtime_receipt(raw: Any, *, require_telemetry: bool = False) -> None:
    decode_runtime_receipt(raw, require_telemetry=require_telemetry)


