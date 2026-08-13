"""Fail-closed validation for SMI/device-state tune evidence (HI60).

The HIP runtime deliberately keeps this evidence collection lightweight and
backwards-compatible.  This module validates the JSON boundary instead: an
empty object or an unannotated zero is never promoted to hardware evidence.
"""

from __future__ import annotations

import math
import re
from typing import Any


class DeviceStateEvidenceError(ValueError):
    """Device-state evidence is missing, ambiguous, or contradictory."""


_METRICS = ("sclk_mhz", "mclk_mhz", "edge_temp_mc", "junction_temp_mc", "socket_power_uw")
_UNAVAILABLE = {"unavailable", "not_supported", "read_error"}
_BDF = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}$")


def _number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeviceStateEvidenceError(f"{name} must be numeric")
    if isinstance(value, float) and not math.isfinite(value):
        raise DeviceStateEvidenceError(f"{name} must be finite")
    if value < 0:
        raise DeviceStateEvidenceError(f"{name} must be non-negative")


def _finite_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeviceStateEvidenceError(f"{name} must be numeric")
    if isinstance(value, float) and not math.isfinite(value):
        raise DeviceStateEvidenceError(f"{name} must be finite")


def _unavailable_reason(snapshot: dict[str, Any], metric: str) -> bool:
    states = snapshot.get("metric_status")
    if not isinstance(states, dict):
        return False
    state = states.get(metric)
    if state not in _UNAVAILABLE:
        return False
    reasons = snapshot.get("metric_reasons")
    return isinstance(reasons, dict) and isinstance(reasons.get(metric), str) and bool(reasons[metric].strip())


def validate_device_state_snapshot(snapshot: Any, *, name: str = "device_state") -> None:
    """Validate one serialized SMI snapshot.

    ``status=unavailable`` is the only valid unresolved representation.  The
    captured shape matches the existing flat runtime JSON and optionally
    accepts per-metric unavailable annotations for ASICs lacking a sensor.
    """
    if not isinstance(snapshot, dict):
        raise DeviceStateEvidenceError(f"{name} must be an object")
    status = snapshot.get("status")
    if status == "unavailable":
        if not isinstance(snapshot.get("reason"), str) or not snapshot["reason"].strip():
            raise DeviceStateEvidenceError(f"{name} unavailable state needs a reason")
        return
    if status not in (None, "captured"):
        raise DeviceStateEvidenceError(f"{name} has an unknown status")
    required = ("identity_valid", "hip_device", "pci_bdf", *_METRICS, "busy_percent")
    missing = [key for key in required if key not in snapshot]
    if missing:
        raise DeviceStateEvidenceError(f"{name} missing fields: {', '.join(missing)}")
    if snapshot["identity_valid"] is not True:
        raise DeviceStateEvidenceError(f"{name} identity is not valid")
    if isinstance(snapshot["hip_device"], bool) or not isinstance(snapshot["hip_device"], int) or snapshot["hip_device"] < 0:
        raise DeviceStateEvidenceError(f"{name}.hip_device is invalid")
    if not isinstance(snapshot["pci_bdf"], str) or not _BDF.fullmatch(snapshot["pci_bdf"]):
        raise DeviceStateEvidenceError(f"{name}.pci_bdf is invalid")
    for metric in _METRICS:
        _number(snapshot[metric], f"{name}.{metric}")
        if snapshot[metric] == 0 and not _unavailable_reason(snapshot, metric):
            raise DeviceStateEvidenceError(
                f"{name}.{metric}=0 lacks explicit unavailable state")
    _number(snapshot["busy_percent"], f"{name}.busy_percent")
    if snapshot["busy_percent"] > 100:
        raise DeviceStateEvidenceError(f"{name}.busy_percent is above 100")


def validate_device_state_report(report: Any) -> None:
    """Validate optional device snapshots and their explicit drift outcome."""
    if not isinstance(report, dict):
        raise DeviceStateEvidenceError("device-state report must be an object")
    present = [key for key in ("device_state_pre", "device_state_post") if key in report]
    if not present:
        raise DeviceStateEvidenceError("device-state report has no snapshots")
    if len(present) != 2:
        raise DeviceStateEvidenceError("device-state report must include pre and post snapshots")
    for key in present:
        validate_device_state_snapshot(report[key], name=key)
    drift = report.get("device_clock_drift")
    if not isinstance(drift, dict):
        raise DeviceStateEvidenceError("device_clock_drift must be an object")
    drift_status = drift.get("status")
    if drift_status not in {"captured", "unavailable", "identity_mismatch", "clock_unavailable"}:
        raise DeviceStateEvidenceError("device_clock_drift has an invalid status")
    if drift_status == "captured":
        for key in ("sclk_delta_mhz", "mclk_delta_mhz"):
            if key not in drift:
                raise DeviceStateEvidenceError(f"captured drift missing {key}")
            _finite_number(drift[key], f"device_clock_drift.{key}")
        for key in ("max_abs_pct", "threshold_pct"):
            if key not in drift:
                raise DeviceStateEvidenceError(f"captured drift missing {key}")
            _number(drift[key], f"device_clock_drift.{key}")
        if not isinstance(drift.get("drift"), bool):
            raise DeviceStateEvidenceError("device_clock_drift.drift must be boolean")
        if drift["threshold_pct"] <= 0:
            raise DeviceStateEvidenceError("device_clock_drift.threshold_pct must be positive")
        if drift["drift"] != (drift["max_abs_pct"] > drift["threshold_pct"]):
            raise DeviceStateEvidenceError("device_clock_drift.drift contradicts max_abs_pct")

    snapshots_unavailable = any(
        isinstance(report[key], dict) and report[key].get("status") == "unavailable"
        for key in ("device_state_pre", "device_state_post")
    )
    if drift_status == "captured" and snapshots_unavailable:
        raise DeviceStateEvidenceError("captured drift requires captured snapshots")

    status = report.get("retime_status")
    counters = ("clock_drift_rounds", "reverse_retime_attempts", "reverse_retime_passed")
    has_counters = any(key in report for key in counters)
    if has_counters and status is None:
        raise DeviceStateEvidenceError("retime counters require retime_status")
    if status is not None:
        if status not in {"not_needed", "corrected", "unresolved", "unavailable"}:
            raise DeviceStateEvidenceError("retime_status has an invalid value")
        values = {}
        for key in counters:
            value = report.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DeviceStateEvidenceError(f"{key} must be a non-negative integer")
            values[key] = value
        if values["reverse_retime_passed"] > values["reverse_retime_attempts"]:
            raise DeviceStateEvidenceError("reverse_retime_passed exceeds attempts")
        if status == "corrected" and not (
                values["clock_drift_rounds"] >= 1 and
                values["reverse_retime_attempts"] >= 1 and
                values["reverse_retime_passed"] >= 1):
            raise DeviceStateEvidenceError("corrected retime lacks complete reverse evidence")
