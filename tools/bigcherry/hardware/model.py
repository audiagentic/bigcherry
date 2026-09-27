"""Stable hardware identity records used by job placement and series binding."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from bigcherry.jobs.model import GpuRequirement, digest


@dataclass(frozen=True)
class DeviceRecord:
    device_id: str
    identity_source: str
    architecture: str
    model: str
    vram_bytes: int
    pci_bdf: str | None = None
    render_node: str | None = None
    numa_node: int | None = None
    driver_version: str | None = None
    launch_ordinal: int | None = None

    def __post_init__(self) -> None:
        if not self.device_id or not self.architecture or not self.model:
            raise ValueError("device_id, architecture and model are required")
        if self.vram_bytes < 1:
            raise ValueError("vram_bytes must be positive")


@dataclass(frozen=True)
class HardwareInventory:
    host_id: str
    platform_family: str
    devices: tuple[DeviceRecord, ...]
    peer_access: tuple[tuple[str, str], ...] = ()
    platform_environment_hash: str = ""
    hardware_epoch: str | None = None

    def __post_init__(self) -> None:
        ids = [device.device_id for device in self.devices]
        if len(ids) != len(set(ids)):
            raise ValueError("device IDs must be unique")

    @property
    def material_hash(self) -> str:
        return digest(dataclasses.asdict(self), person=b"bc-hardware-inv")


@dataclass(frozen=True)
class SeriesGpuBinding:
    requirement: GpuRequirement
    selected_device_ids: tuple[str, ...]
    hardware_cohort_hash: str
    accepted_inventory_hash: str
    reserve_all_of_arch: bool
    scheduler_architecture: str
    scheduler_count: int


def device_from_mapping(value: dict[str, Any]) -> DeviceRecord:
    return DeviceRecord(
        device_id=str(value["device_id"]),
        identity_source=str(value.get("identity_source", "fixture")),
        architecture=str(value["architecture"]),
        model=str(value["model"]),
        vram_bytes=int(value["vram_bytes"]),
        pci_bdf=None if value.get("pci_bdf") is None else str(value["pci_bdf"]),
        render_node=None if value.get("render_node") is None else str(value["render_node"]),
        numa_node=None if value.get("numa_node") is None else int(value["numa_node"]),
        driver_version=None if value.get("driver_version") is None else str(value["driver_version"]),
        launch_ordinal=None if value.get("launch_ordinal") is None else int(value["launch_ordinal"]),
    )


def inventory_from_mapping(value: dict[str, Any]) -> HardwareInventory:
    return HardwareInventory(
        host_id=str(value["host_id"]),
        platform_family=str(value["platform_family"]),
        devices=tuple(device_from_mapping(item) for item in value.get("devices", [])),
        peer_access=tuple(tuple(str(part) for part in edge) for edge in value.get("peer_access", [])),
        platform_environment_hash=str(value.get("platform_environment_hash", "")),
        hardware_epoch=None if value.get("hardware_epoch") is None else str(value["hardware_epoch"]),
    )


def binding_from_mapping(value: dict[str, Any]) -> SeriesGpuBinding:
    req = value.get("requirement")
    if not isinstance(req, dict):
        raise ValueError("series GPU binding is missing requirement")
    requirement = GpuRequirement(
        architecture=str(req["architecture"]),
        count=int(req.get("count", 1)),
        min_vram_bytes=int(req.get("min_vram_bytes", 0)),
        homogeneous_model=bool(req.get("homogeneous_model", True)),
        model=None if req.get("model") is None else str(req["model"]),
        require_peer_access=bool(req.get("require_peer_access", False)),
        exact_device_ids=tuple(str(item) for item in req.get("exact_device_ids", [])),
    )
    return SeriesGpuBinding(
        requirement=requirement,
        selected_device_ids=tuple(str(item) for item in value["selected_device_ids"]),
        hardware_cohort_hash=str(value["hardware_cohort_hash"]),
        accepted_inventory_hash=str(value["accepted_inventory_hash"]),
        reserve_all_of_arch=bool(value["reserve_all_of_arch"]),
        scheduler_architecture=str(value["scheduler_architecture"]),
        scheduler_count=int(value["scheduler_count"]),
    )
