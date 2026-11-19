"""MRC03 platform registry, coverage gate, and engine-neutral production warning.

Native engines own their actual start observations. This registry declares the
feature IDs and expected production capabilities, not their runtime states.
Older MRC01 records remain readable; new producers must provide a full snapshot.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from . import engines, feature_states

# A production expectation applies only when the engine reports that the
# feature is applicable. Engine-only Radiance sessions have no live placement.
FEATURES: dict[str, dict[str, dict[str, Any]]] = {
    engines.RADIANCE: {
        "live_policy": {"production": True, "not_applicable": ("no_helpers",)},
        "mtp": {"production": False},
        "prefix_cache": {"production": False},
        "kv_host_tier": {"production": False},
        "helper_combine": {"production": False},
        "vision_tower_placement": {"production": False},
        "mm_max_patches": {"production": False},
    },
    engines.LLAMACPP: {
        "dispatch_cache": {"production": False},
        "tuning": {"production": False},
    },
}


class ProductionFeatureError(feature_states.FeatureStateError):
    """One or more required production features are inactive or degraded."""


def feature_ids(engine: str) -> tuple[str, ...]:
    if engine == engines.LLAMACPP_RECORD:
        engine = engines.LLAMACPP
    try:
        return tuple(FEATURES[engine])
    except KeyError as exc:
        raise feature_states.FeatureStateError(f"unknown engine feature registry: {engine}") from exc


def is_production(engine: str, feature_id: str) -> bool:
    if feature_id not in feature_ids(engine):
        raise feature_states.FeatureStateError(f"undeclared feature: {engine}/{feature_id}")
    canonical = engines.LLAMACPP if engine == engines.LLAMACPP_RECORD else engine
    return FEATURES[canonical][feature_id]["production"]


def require_coverage(engine: str, states: list[dict]) -> list[dict]:
    """Refuse missing, extra, duplicate, or wrong-engine measured snapshots."""
    feature_states.validate_feature_states(states)
    expected = set(feature_ids(engine))
    actual = {entry["feature_id"] for entry in states}
    if any(entry["engine"] != engine for entry in states):
        raise feature_states.FeatureStateError(f"feature-state engine mismatch for {engine}")
    missing, extra = expected - actual, actual - expected
    if missing:
        raise feature_states.FeatureStateError(f"missing declared features for {engine}: {sorted(missing)}")
    if extra:
        raise feature_states.FeatureStateError(f"undeclared features for {engine}: {sorted(extra)}")
    return states


def production_warnings(engine: str, states: list[dict], *, strict: bool = False) -> list[str]:
    require_coverage(engine, states)
    warnings: list[str] = []
    for entry in states:
        canonical = engines.LLAMACPP if engine == engines.LLAMACPP_RECORD else engine
        declaration = FEATURES[canonical][entry["feature_id"]]
        if not declaration["production"] or entry["state"] == "active":
            continue
        if entry["reason_code"] in declaration.get("not_applicable", ()):
            continue
        reason = entry["reason_detail"].replace("\n", " ").replace("\r", " ")
        warnings.append(
            f"WARNING production feature {entry['feature_id']} {entry['state']}: "
            f"{entry['reason_code']}: {reason}"
        )
    if strict and warnings:
        raise ProductionFeatureError("; ".join(warnings))
    return warnings


def llamacpp_states(environment: dict[str, str], *, at: str | None = None) -> list[dict]:
    """Read dispatch/tuning decisions from the actual measured run environment.

    A configured path or mode is not execution proof. The status here describes
    configured enablement; VA25 validity independently tracks device evidence.
    """
    if not isinstance(environment, dict) or any(
        not isinstance(k, str) or not isinstance(v, str)
        for k, v in environment.items()
    ):
        raise feature_states.FeatureStateError("llamacpp feature environment must be strings")
    stamp = at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    mode = environment.get("GGML_HIP_DISPATCH_MODE", "")
    cache = environment.get("GGML_HIP_DISPATCH_CACHE", "")
    tuning = environment.get("GGML_HIP_TUNE_WORKLOAD", "")
    configured = mode in ("replay", "record") and bool(cache)
    values = (
        ("dispatch_cache", configured, f"mode={mode or 'unset'}; cache={'configured' if cache else 'absent'}"),
        ("tuning", bool(tuning), f"workload={'configured' if tuning else 'absent'}"),
    )
    states = [
        {
            "feature_id": name,
            "state": "active" if enabled else "inactive",
            "reason_code": "configured" if enabled else "not_configured",
            "reason_detail": detail,
            "engine": engines.LLAMACPP_RECORD,
            "observed_at": stamp,
        }
        for name, enabled, detail in values
    ]
    return require_coverage(engines.LLAMACPP_RECORD, states)
