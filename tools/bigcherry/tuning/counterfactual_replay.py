"""Fail-closed construction of winner/runner-up replay pairs (HI171).

This module only constructs and validates immutable, analysis-only manifests.
It deliberately does not rank candidates or mutate tune/promotion/cache data.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


SCHEMA_VERSION = 1
UNREPLAYABLE_PROVENANCE = "UNREPLAYABLE_PROVENANCE"


class CounterfactualProvenanceError(ValueError):
    """Raised when a source decision cannot support causal replay."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(value: Any) -> str:
    """Return the content hash used by identity-bearing manifests."""
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CounterfactualProvenanceError(f"{UNREPLAYABLE_PROVENANCE}: invalid {field}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CounterfactualProvenanceError(
            f"{UNREPLAYABLE_PROVENANCE}: invalid {field}"
        ) from exc
    return value.lower()


def validate_selection(selection: dict[str, Any]) -> dict[str, Any]:
    """Validate one immutable source top-2 decision without reranking it."""
    if not isinstance(selection, dict):
        raise CounterfactualProvenanceError(f"{UNREPLAYABLE_PROVENANCE}: selection is not an object")
    required = ("schema_version", "selection_id", "dispatch", "winner", "runner_up", "source_decision_sha256")
    missing = [name for name in required if name not in selection]
    if missing:
        raise CounterfactualProvenanceError(
            f"{UNREPLAYABLE_PROVENANCE}: missing fields {missing}"
        )
    if selection["schema_version"] != SCHEMA_VERSION:
        raise CounterfactualProvenanceError(f"{UNREPLAYABLE_PROVENANCE}: unsupported schema")
    if not isinstance(selection["selection_id"], str) or not selection["selection_id"]:
        raise CounterfactualProvenanceError(f"{UNREPLAYABLE_PROVENANCE}: invalid selection_id")
    if not isinstance(selection["dispatch"], str) or not selection["dispatch"]:
        raise CounterfactualProvenanceError(f"{UNREPLAYABLE_PROVENANCE}: invalid dispatch")
    for role in ("winner", "runner_up"):
        candidate = selection[role]
        if not isinstance(candidate, dict) or not isinstance(candidate.get("name"), str) or not candidate["name"]:
            raise CounterfactualProvenanceError(f"{UNREPLAYABLE_PROVENANCE}: invalid {role}")
        _require_digest(candidate.get("config_sha256"), f"{role}.config_sha256")
    if selection["winner"]["name"] == selection["runner_up"]["name"]:
        raise CounterfactualProvenanceError(f"{UNREPLAYABLE_PROVENANCE}: winner equals runner_up")
    _require_digest(selection["source_decision_sha256"], "source_decision_sha256")
    return deepcopy(selection)


def build_pair(selection: dict[str, Any], base_manifest: dict[str, Any]) -> dict[str, Any]:
    """Build control/counterfactual manifests with exactly one changed dispatch."""
    source = validate_selection(selection)
    if not isinstance(base_manifest, dict):
        raise CounterfactualProvenanceError(f"{UNREPLAYABLE_PROVENANCE}: invalid base manifest")
    control = deepcopy(base_manifest)
    counterfactual = deepcopy(base_manifest)
    dispatches = counterfactual.get("dispatches")
    if not isinstance(dispatches, dict) or source["dispatch"] not in dispatches:
        raise CounterfactualProvenanceError(f"{UNREPLAYABLE_PROVENANCE}: target dispatch missing")
    control["dispatches"][source["dispatch"]] = deepcopy(source["winner"])
    counterfactual["dispatches"][source["dispatch"]] = deepcopy(source["runner_up"])
    if control == counterfactual:
        raise CounterfactualProvenanceError(f"{UNREPLAYABLE_PROVENANCE}: pair has no controlled difference")
    changed = [key for key in control["dispatches"] if control["dispatches"].get(key) != counterfactual["dispatches"].get(key)]
    if changed != [source["dispatch"]]:
        raise CounterfactualProvenanceError(f"{UNREPLAYABLE_PROVENANCE}: invariant diff is not singular")
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_only": True,
        "source_selection_id": source["selection_id"],
        "source_decision_sha256": source["source_decision_sha256"].lower(),
        "winner": deepcopy(source["winner"]),
        "runner_up": deepcopy(source["runner_up"]),
        "control_manifest": control,
        "counterfactual_manifest": counterfactual,
        "control_manifest_sha256": sha256(control),
        "counterfactual_manifest_sha256": sha256(counterfactual),
        "invariant_diff": {"dispatch": source["dispatch"]},
    }
