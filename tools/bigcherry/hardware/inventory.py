"""Accepted/observed inventory persistence and deterministic GPU binding."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from bigcherry.jobs.executor import Allocation
from bigcherry.jobs.model import GpuRequirement, digest
from bigcherry.tuning.journal import atomic_write
from .drift import DriftReport, classify_drift
from .model import HardwareInventory, SeriesGpuBinding, inventory_from_mapping


class HardwareBindingError(RuntimeError):
    pass


def _inventory_bytes(inventory: HardwareInventory) -> bytes:
    value = dataclasses.asdict(inventory)
    value["material_hash"] = inventory.material_hash
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True).encode("ascii") + b"\n"


def _load_inventory(path: Path, *, label: str) -> HardwareInventory:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HardwareBindingError(f"{label} hardware inventory unavailable: {exc}") from exc
    if not isinstance(value, dict):
        raise HardwareBindingError(f"{label} hardware inventory must be an object")
    stored_hash = value.pop("material_hash", None)
    inventory = inventory_from_mapping(value)
    if stored_hash is not None and stored_hash != inventory.material_hash:
        raise HardwareBindingError(f"{label} hardware inventory material_hash mismatch")
    return inventory


class InventoryCatalog:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def executor_root(self, executor_id: str) -> Path:
        if not executor_id.strip() or any(part in {".", ".."} for part in Path(executor_id).parts):
            raise ValueError("invalid executor_id")
        return self.root / executor_id

    def accepted_path(self, executor_id: str) -> Path:
        return self.executor_root(executor_id) / "accepted.json"

    def observed_path(self, executor_id: str) -> Path:
        return self.executor_root(executor_id) / "observed.json"

    def load(self, executor_id: str) -> HardwareInventory:
        return _load_inventory(self.accepted_path(executor_id), label="accepted")

    def load_observed(self, executor_id: str) -> HardwareInventory:
        return _load_inventory(self.observed_path(executor_id), label="observed")

    def record_observed(self, executor_id: str, inventory: HardwareInventory) -> Path:
        root = self.executor_root(executor_id)
        root.mkdir(parents=True, exist_ok=True)
        if inventory.host_id.strip() == "":
            raise HardwareBindingError("observed inventory host_id is empty")
        path = self.observed_path(executor_id)
        atomic_write(path, _inventory_bytes(inventory))
        return path

    def drift(self, executor_id: str) -> DriftReport:
        return classify_drift(self.load(executor_id), self.load_observed(executor_id))

    def accept_observed(
        self,
        executor_id: str,
        *,
        expected_material_hash: str,
        allow_material_change: bool = False,
    ) -> HardwareInventory:
        observed = self.load_observed(executor_id)
        if observed.material_hash != expected_material_hash:
            raise HardwareBindingError(
                "observed inventory changed since review; expected hash does not match"
            )
        accepted_path = self.accepted_path(executor_id)
        if accepted_path.exists() and not allow_material_change:
            accepted = self.load(executor_id)
            report = classify_drift(accepted, observed)
            if report.material:
                raise HardwareBindingError(
                    f"material hardware drift {report.kind!r} requires explicit allow_material_change"
                )
        accepted_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(accepted_path, _inventory_bytes(observed))
        return observed


def _peer_ok(inventory: HardwareInventory, ids: tuple[str, ...]) -> bool:
    if len(ids) <= 1:
        return True
    allowed = {tuple(sorted(edge)) for edge in inventory.peer_access}
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            if tuple(sorted((left, right))) not in allowed:
                return False
    return True


def bind_gpu_requirement(
    req: GpuRequirement, inventory: HardwareInventory
) -> SeriesGpuBinding:
    candidates = tuple(
        sorted(
            (
                device
                for device in inventory.devices
                if device.architecture == req.architecture
                and device.vram_bytes >= req.min_vram_bytes
            ),
            key=lambda device: device.device_id,
        )
    )
    if req.model is not None:
        candidates = tuple(device for device in candidates if device.model == req.model)
    if req.exact_device_ids:
        by_id = {device.device_id: device for device in candidates}
        missing = tuple(
            device_id for device_id in req.exact_device_ids if device_id not in by_id
        )
        if missing:
            raise HardwareBindingError(f"exact GPU IDs unavailable/ineligible: {missing}")
        selected = tuple(sorted(req.exact_device_ids))
    else:
        if req.homogeneous_model:
            groups: dict[str, list[str]] = {}
            for device in candidates:
                groups.setdefault(device.model, []).append(device.device_id)
            viable = [
                (model, tuple(sorted(ids)))
                for model, ids in groups.items()
                if len(ids) >= req.count
            ]
            if req.model is None and len(viable) > 1:
                raise HardwareBindingError(
                    "GPU request is ambiguous across multiple same-architecture models; "
                    "specify model or exact IDs"
                )
            if not viable:
                raise HardwareBindingError("no homogeneous GPU cohort satisfies request")
            candidate_ids = viable[0][1]
        else:
            candidate_ids = tuple(device.device_id for device in candidates)
        if req.require_peer_access:
            import itertools

            valid = [
                tuple(combo)
                for combo in itertools.combinations(candidate_ids, req.count)
                if _peer_ok(inventory, tuple(combo))
            ]
            if not valid:
                raise HardwareBindingError("no peer-capable GPU cohort satisfies request")
            selected = min(valid)
        else:
            if len(candidate_ids) < req.count:
                raise HardwareBindingError("not enough GPUs satisfy request")
            selected = tuple(candidate_ids[: req.count])
    if req.require_peer_access and not _peer_ok(inventory, selected):
        raise HardwareBindingError("selected exact GPU cohort is not peer-capable")
    selected_records = tuple(
        sorted(
            (device for device in inventory.devices if device.device_id in selected),
            key=lambda d: d.device_id,
        )
    )
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
        "peer_edges": sorted(
            edge
            for edge in inventory.peer_access
            if edge[0] in selected and edge[1] in selected
        ),
    }
    all_arch = tuple(
        device.device_id
        for device in inventory.devices
        if device.architecture == req.architecture
    )
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


def verify_series_allocation(
    binding: SeriesGpuBinding,
    allocation: Allocation,
    inventory: HardwareInventory,
) -> tuple[str, ...]:
    """Verify executor allocation evidence against an immutable series binding."""
    if inventory.material_hash != binding.accepted_inventory_hash:
        raise HardwareBindingError(
            "accepted hardware inventory changed after series binding"
        )
    stable = tuple(sorted(allocation.stable_gpu_ids))
    if not stable:
        raise HardwareBindingError(
            "executor allocation has no attested stable GPU identities"
        )
    selected = tuple(sorted(binding.selected_device_ids))
    missing = tuple(device_id for device_id in selected if device_id not in stable)
    if missing:
        raise HardwareBindingError(
            f"allocation does not contain frozen series GPU IDs: {missing}"
        )
    current = bind_gpu_requirement(binding.requirement, inventory)
    if current.selected_device_ids != binding.selected_device_ids:
        raise HardwareBindingError(
            "current capability resolution no longer matches frozen series cohort"
        )
    if current.hardware_cohort_hash != binding.hardware_cohort_hash:
        raise HardwareBindingError("hardware cohort/topology drifted")
    extras = tuple(device_id for device_id in stable if device_id not in selected)
    if extras and not binding.reserve_all_of_arch:
        raise HardwareBindingError(
            f"unexpected extra stable GPUs in allocation: {extras}"
        )
    if binding.reserve_all_of_arch:
        expected_pool = tuple(
            sorted(
                device.device_id
                for device in inventory.devices
                if device.architecture == binding.scheduler_architecture
            )
        )
        if stable != expected_pool:
            raise HardwareBindingError(
                "safe over-allocation must contain the complete accepted architecture pool"
            )
    return selected
