"""MRC03 platform feature registry, snapshot coverage and production warnings.

Radiance owns the measured runtime states. The platform declares which
features are expected in production without inferring runtime enablement.
"""
from __future__ import annotations

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
}


class ProductionFeatureError(feature_states.FeatureStateError):
    """One or more required production features are inactive or degraded."""


def feature_ids(engine: str) -> tuple[str, ...]:
    try:
        return tuple(FEATURES[engine])
    except KeyError as exc:
        raise feature_states.FeatureStateError(f"unknown engine feature registry: {engine}") from exc


def is_production(engine: str, feature_id: str) -> bool:
    if feature_id not in feature_ids(engine):
        raise feature_states.FeatureStateError(f"undeclared feature: {engine}/{feature_id}")
    return FEATURES[engine][feature_id]["production"]


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
        declaration = FEATURES[engine][entry["feature_id"]]
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
