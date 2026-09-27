"""Accepted-inventory persistence and deterministic GPU capability binding."""
from __future__ import annotations

import json
from pathlib import Path

from bigcherry.jobs.model import GpuRequirement, digest
from .model import HardwareInventory, SeriesGpuBinding, inventory_from_mapping


class HardwareBindingError(RuntimeError):
    pass


class InventoryCatalog:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def accepted_path(self, executor_id: str) -> Path:
        return self.root / executor_id / "accepted.json"

    def load(self, executor_id: str) -> HardwareInventory:
        path = self.accepted_path(executor_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HardwareBindingError(f"accepted hardware inventory unavailable for {executor_id}: {exc}") from exc
        if not isinstance(value, dict):
            raise HardwareBindingError("accepted hardware inventory must be an object")
        return inventory_from_mapping(value)


def _peer_ok(inventory: HardwareInventory, ids: tuple[str, ...]) -> bool:
    if len(ids) <= 1:
        return True
    allowed = {tuple(sorted(edge)) for edge in inventory.peer_access}
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            if tuple(sorted((left, right))) not in allowed:
                return False
    return True


def bind_gpu_requirement(req: GpuRequirement, inventory: HardwareInventory) -> SeriesGpuBinding:
    candidates = tuple(sorted(
        (device for device in inventory.devices
         if device.architecture == req.architecture and device.vram_bytes >= req.min_vram_bytes),
        key=lambda device: device.device_id,
    ))
    if req.model is not None:
        candidates = tuple(device for device in candidates if device.model == req.model)
    if req.exact_device_ids:
        by_id = {device.device_id: device for device in candidates}
        missing = tuple(device_id for device_id in req.exact_device_ids if device_id not in by_id)
        if missing:
            raise HardwareBindingError(f"exact GPU IDs unavailable/ineligible: {missing}")
        selected = tuple(sorted(req.exact_device_ids))
    else:
        if req.homogeneous_model:
            groups: dict[str, list[str]] = {}
            for device in candidates:
                groups.setdefault(device.model, []).append(device.device_id)
            viable = [(model, tuple(sorted(ids))) for model, ids in groups.items() if len(ids) >= req.count]
            if req.model is None and len(viable) > 1:
                raise HardwareBindingError("GPU request is ambiguous across multiple same-architecture models; specify model or exact IDs")
            if not viable:
                raise HardwareBindingError("no homogeneous GPU cohort satisfies request")
            candidate_ids = viable[0][1]
        else:
            candidate_ids = tuple(device.device_id for device in candidates)
        if req.require_peer_access:
            import itertools
            valid = [tuple(combo) for combo in itertools.combinations(candidate_ids, req.count) if _peer_ok(inventory, tuple(combo))]
            if not valid:
                raise HardwareBindingError("no peer-capable GPU cohort satisfies request")
            selected = min(valid)
        else:
            if len(candidate_ids) < req.count:
                raise HardwareBindingError("not enough GPUs satisfy request")
            selected = tuple(candidate_ids[:req.count])
    if req.require_peer_access and not _peer_ok(inventory, selected):
        raise HardwareBindingError("selected exact GPU cohort is not peer-capable")
    selected_records = tuple(sorted((device for device in inventory.devices if device.device_id in selected), key=lambda d: d.device_id))
    cohort_material = {
        "host_id": inventory.host_id,
        "platform_family": inventory.platform_family,
        "platform_environment_hash": inventory.platform_environment_hash,
        "devices": [
            {
                "device_id": device.device_id,
                "architecture": device.architecture,
                "model": device.model,
                "vram_bytes": device.vram_bytes,
            }
            for device in selected_records
        ],
        "peer_edges": sorted(edge for edge in inventory.peer_access if edge[0] in selected and edge[1] in selected),
    }
    all_arch = tuple(device.device_id for device in inventory.devices if device.architecture == req.architecture)
    reserve_all = set(selected) != set(all_arch)
    scheduler_count = len(all_arch) if reserve_all else req.count
    return SeriesGpuBinding(
        requirement=req,
        selected_device_ids=selected,
        hardware_cohort_hash=digest(cohort_material, person=b"bc-hw-cohort"),
        accepted_inventory_hash=inventory.material_hash,
        reserve_all_of_arch=reserve_all,
        scheduler_architecture=req.architecture,
        scheduler_count=scheduler_count,
    )
