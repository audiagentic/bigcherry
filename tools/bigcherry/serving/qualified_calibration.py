"""RSC11 S13 pre-cutover: bind each native P1 measurement to P2 evidence.

Production still runs v1 until the complete atomic v2 cutover. No synthetic
default, prior manifest or unqualified cache entry may authorize S8 values.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .calibration_rebind import select_with_rebind
from .calibration_store import CalibrationInvalid, validate_scope
from .measured_plan import PlanBounds, PlanMeasurement

_HANDOFF = (
    "handoff.max_rows", "handoff.small_rows",
    "handoff.gemv_large_rows", "handoff.moe_down_tile",
)
_MARGIN = "host_ledger_margin.mib"
_HISTOGRAM = "handoff.histogram_period_s"
_LIVE = "handoff.live_cadence_ns"


def required_s8_records(*, helper_count: int, histogram_enabled: bool,
                        live_policy_requested: bool) -> tuple[str, ...]:
    """Compute requirements from selected roles and explicit policy."""
    if type(helper_count) is not int or helper_count < 0:
        raise CalibrationInvalid("S13 invalid selected helper count")
    if type(histogram_enabled) is not bool or type(live_policy_requested) is not bool:
        raise CalibrationInvalid("S13 histogram/live policy decisions must be explicit")
    if histogram_enabled and not helper_count:
        raise CalibrationInvalid("S13 histogram requires a selected helper")
    if live_policy_requested and (not helper_count or not histogram_enabled):
        raise CalibrationInvalid(
            "S13 live_policy_refused: live placement requires helpers and histogram")
    keys = [_MARGIN]
    if helper_count:
        keys.extend(_HANDOFF)
    if histogram_enabled:
        keys.append(_HISTOGRAM)
    if live_policy_requested:
        keys.append(_LIVE)
    return tuple(keys)


def select_for_measurement(
    measurement: PlanMeasurement,
    store: Path,
    *,
    helper_count: int,
    histogram_enabled: bool,
    live_policy_requested: bool,
    now_unix_s: int,
) -> tuple[dict, ...]:
    """Select fresh exact-scope records and cross-check native margin provenance."""
    if not isinstance(measurement.scope, dict) or not isinstance(measurement.resource_inputs, dict):
        raise CalibrationInvalid("S13 native P1 scope and resource inputs required")
    scope = validate_scope(measurement.scope)
    keys = required_s8_records(
        helper_count=helper_count,
        histogram_enabled=histogram_enabled,
        live_policy_requested=live_policy_requested,
    )
    try:
        records = select_with_rebind(
            Path(store), scope=scope, required_keys=keys, now_unix_s=now_unix_s,
        )
    except CalibrationInvalid as exc:
        if _MARGIN in str(exc):
            category = "host_margin_refused"
        elif _LIVE in str(exc) or _HISTOGRAM in str(exc):
            category = "live_policy_refused" if live_policy_requested else "histogram_refused"
        else:
            category = "handoff_refused"
        raise CalibrationInvalid(f"S13 {category}: {exc}") from exc

    selected = records[0]
    expected = {
        "key": _MARGIN,
        "source": selected["source"],
        "evidence": selected["evidence_id"],
        "mib": selected["value"],
    }
    if measurement.resource_inputs.get("host_pool_margin_calibration") != expected:
        raise CalibrationInvalid(
            "S13 host_margin_refused: native P1 margin differs from "
            "exact-scope P2 record (key/source/evidence/value)")
    return records


class QualifiedP1MeasurementSource:
    """S11 source that cannot return an unqualified S8 measurement.

    S13 must inject the real native command and this gate. A bounded retry
    remeasures and selects calibration anew, never reuses an old scope.
    """

    def __init__(
        self,
        source,
        *,
        store: Path,
        helper_count: int,
        histogram_enabled: bool,
        live_policy_requested: bool,
        clock: Callable[[], int] = lambda: int(time.time()),
    ):
        self.source = source
        self.store = Path(store)
        self.helper_count = helper_count
        self.histogram_enabled = histogram_enabled
        self.live_policy_requested = live_policy_requested
        self.clock = clock

    def measure(self, layout, bounds: PlanBounds) -> PlanMeasurement:
        if self.helper_count != len(layout.helper_bdfs):
            raise CalibrationInvalid("S13 helper count differs from selected layout")
        measured = self.source.measure(layout, bounds)
        records = select_for_measurement(
            measured, self.store,
            helper_count=self.helper_count,
            histogram_enabled=self.histogram_enabled,
            live_policy_requested=self.live_policy_requested,
            now_unix_s=self.clock(),
        )
        return replace(measured, calibration_records=records)
