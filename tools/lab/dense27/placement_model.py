"""Card-free, evidence-gated contiguous-stage placement estimator for dense 27B.

No BigCherry serving configuration is changed by this tool. A 'hybrid' group
is a hypothetical execution stage, not a feature claimed to exist in Radiance.
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

MEMORY_CLASSES = ("weights_mib", "kv_mib", "state_mib")
OBJECTIVES = ("decode_latency", "decode_pipeline_interval",
              "prefill_latency", "prefill_pipeline_interval")


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")  # noqa: TRY004
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0 or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def _expand_layers(data: dict) -> list[dict]:
    pattern = data.get("layer_pattern")
    if not isinstance(pattern, list) or not pattern:
        raise ValueError("layer_pattern must be a nonempty list")
    repeats = data.get("repeat_pattern", 1)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeat_pattern must be a positive integer")
    out = []
    for _ in range(repeats):
        for entry in pattern:
            count = entry.get("repeat", 1)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("layer repeat must be positive")
            for _ in range(count):
                out.append(entry)
    if len(out) > 4096:
        raise ValueError("too many layers")
    for layer in out:
        for field in MEMORY_CLASSES:
            _number(layer.get(field, 0), field)
        for metric in ("decode_us", "prefill_us"):
            if not isinstance(layer.get(metric), dict):
                raise ValueError(f"each layer needs measured {metric} by group")  # noqa: TRY004
    return out


def _validate(data: dict) -> tuple[list[dict], dict, dict, dict]:
    if data.get("data_kind") not in ("measured", "synthetic"):
        raise ValueError("data_kind must be measured or synthetic")
    if data["data_kind"] == "measured" and not data.get("evidence_id"):
        raise ValueError("measured data requires evidence_id")
    devices = data.get("devices")
    groups = data.get("groups")
    topologies = data.get("topologies")
    links = data.get("transfer_us")
    if not all(isinstance(item, dict) and item for item in (devices, groups, topologies, links)):
        raise ValueError("devices, groups, topologies, and transfer_us must be nonempty maps")
    for name, dev in devices.items():
        _number(dev["budget_mib"], f"{name} budget", positive=True)
    for name, group in groups.items():
        fractions = group.get("fractions")
        if not isinstance(fractions, dict) or not fractions:
            raise ValueError(f"{name} must provide explicit per-device fractions")
        for device, share in fractions.items():
            if device not in devices:
                raise ValueError(f"{name} references unknown device {device}")
            for field in MEMORY_CLASSES:
                fraction = _number(share.get(field, 0), f"{name}/{device}/{field}")
                if fraction > 1:
                    raise ValueError(f"{name}/{device}/{field} fraction exceeds 1")
        for field in MEMORY_CLASSES:
            if sum(float(share.get(field, 0)) for share in fractions.values()) < 0.999999:
                raise ValueError(f"{name} fails to account for all {field}")
    for name, topology in topologies.items():
        if not isinstance(topology, list) or not topology:
            raise ValueError(f"{name} must be an ordered group list")
        used_devices = set()
        for group in topology:
            if group not in groups:
                raise ValueError(f"{name} references unknown group {group}")
            overlapping = used_devices.intersection(groups[group]["fractions"])
            if overlapping:
                raise ValueError(f"{name} has overlapping stage devices: {sorted(overlapping)}")
            used_devices.update(groups[group]["fractions"])
        for a, b in pairwise(topology):
            link = links.get(f"{a}>{b}")
            if not isinstance(link, dict):
                raise ValueError(f"missing measured transfer {a}>{b}")  # noqa: TRY004
            for field in ("decode", "prefill"):
                _number(link.get(field), f"transfer {a}>{b} {field}")
    layers = _expand_layers(data)
    for layer in layers:
        for group in groups:
            for field in ("decode_us", "prefill_us"):
                _number(layer[field].get(group), f"{group}/{field}", positive=True)
    return layers, devices, groups, topologies


def estimate(data: dict, topology_name: str, objective: str) -> dict:
    """Enumerate contiguous cut points exactly; figures are a simple stage model."""
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective}")
    layers, devices, groups, topologies = _validate(data)
    if topology_name not in topologies:
        raise ValueError(f"unknown topology {topology_name}")
    topology = topologies[topology_name]
    if len(topology) > len(layers):
        raise ValueError("fewer layers than nonempty stages")
    best = None
    n = len(layers)
    for cuts in combinations(range(1, n), len(topology) - 1):
        boundaries = (0, *cuts, n)
        used = {name: 0.0 for name in devices}
        stages = []
        feasible = True
        for index, group_name in enumerate(topology):
            lo, hi = boundaries[index:index + 2]
            portion = layers[lo:hi]
            fraction_by_device = groups[group_name]["fractions"]
            for device, fraction in fraction_by_device.items():
                used[device] += sum(
                    float(layer.get(field, 0)) * float(fraction.get(field, 0))
                    for layer in portion for field in MEMORY_CLASSES
                )
                if used[device] > float(devices[device]["budget_mib"]) + 1e-8:
                    feasible = False
                    break
            if not feasible:
                break
            stages.append({
                "group": group_name, "start_layer": lo, "end_layer_exclusive": hi,
                "decode_us": sum(float(layer["decode_us"][group_name]) for layer in portion),
                "prefill_us": sum(float(layer["prefill_us"][group_name]) for layer in portion),
            })
        if not feasible:
            continue
        decode_transfer = [
            float(data["transfer_us"][f"{a}>{b}"]["decode"])
            for a, b in pairwise(topology)
        ]
        prefill_transfer = [
            float(data["transfer_us"][f"{a}>{b}"]["prefill"])
            for a, b in pairwise(topology)
        ]
        metrics = {
            "decode_latency": sum(stage["decode_us"] for stage in stages) + sum(decode_transfer),
            "prefill_latency": sum(stage["prefill_us"] for stage in stages) + sum(prefill_transfer),
            # Optimistic pipeline lower bound; excludes host scheduling and link/compute contention.
            "decode_pipeline_interval": max(
                [stage["decode_us"] for stage in stages] + decode_transfer),
            "prefill_pipeline_interval": max(
                [stage["prefill_us"] for stage in stages] + prefill_transfer),
        }
        key = (metrics[objective], metrics["decode_latency"], cuts)
        if best is None or key < best[0]:
            best = (key, {
                "data_kind": data["data_kind"], "evidence_id": data.get("evidence_id"),
                "topology": topology_name, "objective": objective,
                "cuts": list(cuts), "stages": stages,
                "memory_mib": {
                    name: {"used": round(used[name], 3),
                           "spare": round(float(device["budget_mib"]) - used[name], 3)}
                    for name, device in devices.items()
                },
                "estimates_us": {k: round(v, 3) for k, v in metrics.items()},
                "limitations": [
                    "The pipeline interval is an optimistic lower bound, not a measured rate.",
                    "Transfer_us must include real GPU/host hand-off and coordination costs.",
                    "No GPU allocations, kernel calls, model loading, quality, or serving integration are tested.",
                ],
            })
    if best is None:
        raise ValueError("no VRAM-feasible contiguous partition for that topology")
    return best[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--objective", choices=OBJECTIVES, default="decode_latency")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--output", type=Path, help="write report JSON (not a qualification)")
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
        if data.get("data_kind") == "synthetic" and not args.allow_synthetic:
            raise ValueError("synthetic input needs explicit --allow-synthetic")
        result = estimate(data, args.topology, args.objective)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
