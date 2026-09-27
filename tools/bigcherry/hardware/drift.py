"""Fail-closed hardware inventory drift classification."""
from __future__ import annotations

from dataclasses import dataclass

from .model import HardwareInventory


@dataclass(frozen=True)
class DriftReport:
    kind: str
    material: bool
    requires_new_cohort: bool
    details: tuple[str, ...] = ()


def classify_drift(
    accepted: HardwareInventory, observed: HardwareInventory
) -> DriftReport:
    if accepted.host_id != observed.host_id:
        return DriftReport("host-change", True, True, (accepted.host_id, observed.host_id))
    if accepted.platform_environment_hash != observed.platform_environment_hash:
        return DriftReport("environment-change", True, True)

    old = {device.device_id: device for device in accepted.devices}
    new = {device.device_id: device for device in observed.devices}
    removed = tuple(sorted(set(old) - set(new)))
    added = tuple(sorted(set(new) - set(old)))
    if removed and added:
        return DriftReport(
            "device-replacement",
            True,
            True,
            tuple(f"removed:{x}" for x in removed) + tuple(f"added:{x}" for x in added),
        )
    if removed:
        return DriftReport(
            "device-remove", True, True, tuple(f"removed:{x}" for x in removed)
        )
    if added:
        return DriftReport(
            "device-add", True, True, tuple(f"added:{x}" for x in added)
        )

    topology_change = False
    locator_change = False
    details: list[str] = []
    for device_id in sorted(old):
        before = old[device_id]
        after = new[device_id]
        if (
            before.architecture,
            before.model,
            before.vram_bytes,
            before.identity_source,
        ) != (
            after.architecture,
            after.model,
            after.vram_bytes,
            after.identity_source,
        ):
            return DriftReport("device-identity-change", True, True, (device_id,))
        if before.numa_node != after.numa_node:
            topology_change = True
            details.append(f"numa:{device_id}:{before.numa_node}->{after.numa_node}")
        if (
            before.pci_bdf != after.pci_bdf
            or before.render_node != after.render_node
            or before.launch_ordinal != after.launch_ordinal
        ):
            locator_change = True
            details.append(f"locator:{device_id}")

    if {
        tuple(sorted(edge)) for edge in accepted.peer_access
    } != {tuple(sorted(edge)) for edge in observed.peer_access}:
        topology_change = True
        details.append("peer-access")

    if topology_change:
        return DriftReport("topology-change", True, True, tuple(details))
    if locator_change or accepted.material_hash != observed.material_hash:
        return DriftReport("locator-only", True, False, tuple(details))
    return DriftReport("same", False, False)
