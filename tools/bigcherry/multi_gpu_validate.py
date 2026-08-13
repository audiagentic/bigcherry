"""Offline validation for multi-GPU tuning evidence (HI14).

This module validates evidence produced by a hardware run; it never probes
HIP and never infers a claim from partial observations.  The intentionally
small schema keeps topology, graph mode, and per-device observations joined at
the claim boundary without changing tuning or replay formats.
"""

from __future__ import annotations

from typing import Any


class MultiGPUEvidenceError(ValueError):
    """The evidence cannot support a multi-GPU claim."""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MultiGPUEvidenceError(f"{field} must be a positive integer")
    return value


def _ordinal(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MultiGPUEvidenceError(f"{field} must be a non-negative integer")
    return value


def validate_multi_gpu_evidence(evidence: dict[str, Any]) -> None:
    """Fail closed unless topology, graph, and device evidence agree.

    Expected shape::

        {"topology": {"device_count": 2, "ordinals": [0, 1],
                       "devices": [{"ordinal": 0, ...}, ...]},
         "graph": {"mode": "enabled", "capture_observed": True},
         "per_device": [{"ordinal": 0, "dispatches": 10,
                         "signatures": 3, "winners": [...]}, ...]}

    ``graph.mode`` may be ``enabled`` or ``disabled``.  Enabled graph claims
    require observed capture; disabled mode is still explicit evidence and is
    not silently treated as enabled.
    """
    if not isinstance(evidence, dict):
        raise MultiGPUEvidenceError("multi_gpu evidence must be an object")

    topology = evidence.get("topology")
    if not isinstance(topology, dict):
        raise MultiGPUEvidenceError("multi_gpu evidence lacks topology")
    count = topology.get("device_count")
    if (isinstance(count, bool) or not isinstance(count, int) or count < 2):
        raise MultiGPUEvidenceError("multi-GPU topology needs at least two devices")

    devices = topology.get("devices")
    ordinals = topology.get("ordinals")
    if not isinstance(devices, list) or len(devices) != count:
        raise MultiGPUEvidenceError("topology devices do not match device_count")
    if not isinstance(ordinals, list) or len(ordinals) != count:
        raise MultiGPUEvidenceError("topology ordinals do not match device_count")

    topology_ordinals = [_ordinal(value, "topology ordinal") for value in ordinals]
    if len(set(topology_ordinals)) != count:
        raise MultiGPUEvidenceError("topology ordinals must be unique")
    if sorted(topology_ordinals) != list(range(count)):
        raise MultiGPUEvidenceError("topology ordinals must cover 0..device_count-1")

    device_ordinals: list[int] = []
    for index, device in enumerate(devices):
        if not isinstance(device, dict):
            raise MultiGPUEvidenceError(f"topology device {index} must be an object")
        ordinal = _ordinal(device.get("ordinal"), f"topology device {index} ordinal")
        if ordinal in device_ordinals:
            raise MultiGPUEvidenceError("topology device ordinals must be unique")
        device_ordinals.append(ordinal)
        if not isinstance(device.get("identity"), str) or not device["identity"].strip():
            raise MultiGPUEvidenceError(f"topology device {ordinal} lacks identity")
    if set(device_ordinals) != set(topology_ordinals):
        raise MultiGPUEvidenceError("topology device and ordinal lists disagree")

    graph = evidence.get("graph")
    if not isinstance(graph, dict):
        raise MultiGPUEvidenceError("multi_gpu evidence lacks graph mode")
    mode = graph.get("mode")
    if mode not in {"enabled", "disabled"}:
        raise MultiGPUEvidenceError("graph mode must be enabled or disabled")
    captured = graph.get("capture_observed")
    if not isinstance(captured, bool):
        raise MultiGPUEvidenceError("graph capture_observed must be boolean")
    if mode == "enabled" and not captured:
        raise MultiGPUEvidenceError("enabled graph mode lacks observed capture")
    if captured and mode != "enabled":
        raise MultiGPUEvidenceError("observed graph capture contradicts disabled mode")

    per_device = evidence.get("per_device")
    if not isinstance(per_device, list) or len(per_device) != count:
        raise MultiGPUEvidenceError("per_device evidence does not match device_count")
    seen: set[int] = set()
    for entry in per_device:
        if not isinstance(entry, dict):
            raise MultiGPUEvidenceError("per_device entries must be objects")
        ordinal = _ordinal(entry.get("ordinal"), "per_device ordinal")
        if ordinal in seen:
            raise MultiGPUEvidenceError("per_device ordinals must be unique")
        seen.add(ordinal)
        _positive_int(entry.get("dispatches"), f"per_device[{ordinal}].dispatches")
        _positive_int(entry.get("signatures"), f"per_device[{ordinal}].signatures")
        winners = entry.get("winners")
        if not isinstance(winners, list) or not winners or any(
            not isinstance(winner, str) or not winner.strip() for winner in winners
        ):
            raise MultiGPUEvidenceError(f"per_device[{ordinal}] lacks winners")
    if seen != set(topology_ordinals):
        raise MultiGPUEvidenceError("per_device evidence does not cover topology")


def validate_multi_gpu_claim(record: dict[str, Any]) -> None:
    """Validate the optional multi-GPU claim attached to a release record."""
    if not isinstance(record, dict):
        raise MultiGPUEvidenceError("release record must be an object")
    claim = record.get("multi_gpu")
    if claim is None:
        return
    validate_multi_gpu_evidence(claim)
