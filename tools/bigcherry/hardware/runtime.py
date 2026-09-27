"""Runtime GPU visibility -> accepted stable-ID allocation attestation."""
from __future__ import annotations

from bigcherry.jobs.executor import Allocation
from .inventory import HardwareBindingError
from .model import HardwareInventory, SeriesGpuBinding


def allocation_from_visible_tokens(
    raw: str,
    inventory: HardwareInventory,
    *,
    environment_key: str = "ROCR_VISIBLE_DEVICES",
) -> Allocation:
    """Map ordered runtime-visible tokens to stable accepted device IDs.

    A token may already be a stable device ID/UUID, or it may be a decimal
    launch ordinal from the accepted inventory. Other values fail closed.
    Order is preserved because it defines allocation-local device positions.
    """
    tokens = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not tokens:
        raise HardwareBindingError(f"{environment_key} is empty")
    by_id = {device.device_id: device for device in inventory.devices}
    by_ordinal = {
        str(device.launch_ordinal): device
        for device in inventory.devices
        if device.launch_ordinal is not None
    }
    stable: list[str] = []
    for token in tokens:
        if token in by_id:
            stable.append(token)
            continue
        device = by_ordinal.get(token)
        if device is None:
            raise HardwareBindingError(
                f"cannot map {environment_key} token {token!r} to accepted stable identity"
            )
        stable.append(device.device_id)
    if len(set(stable)) != len(stable):
        raise HardwareBindingError("runtime visibility maps multiple tokens to one stable GPU")
    return Allocation(
        native_gpu_ids=tokens,
        environment=((environment_key, raw),),
        stable_gpu_ids=tuple(stable),
    )


def selected_local_positions(
    binding: SeriesGpuBinding, allocation: Allocation
) -> tuple[int, ...]:
    """Return allocation-local positions for the exact frozen series cohort."""
    if not allocation.stable_gpu_ids:
        raise HardwareBindingError("stable allocation identities are required")
    positions: list[int] = []
    for device_id in binding.selected_device_ids:
        try:
            positions.append(allocation.stable_gpu_ids.index(device_id))
        except ValueError as exc:
            raise HardwareBindingError(
                f"frozen GPU {device_id!r} is not present in runtime allocation"
            ) from exc
    if len(set(positions)) != len(positions):
        raise HardwareBindingError("duplicate allocation-local GPU position")
    return tuple(positions)


def selected_launch_ordinals(
    binding: SeriesGpuBinding, inventory: HardwareInventory
) -> tuple[int, ...]:
    """Resolve a local/direct frozen cohort to current accepted launch ordinals."""
    by_id = {device.device_id: device for device in inventory.devices}
    ordinals: list[int] = []
    for device_id in binding.selected_device_ids:
        device = by_id.get(device_id)
        if device is None:
            raise HardwareBindingError(f"frozen GPU {device_id!r} is no longer accepted")
        if device.launch_ordinal is None:
            raise HardwareBindingError(
                f"accepted GPU {device_id!r} has no launch_ordinal for local execution"
            )
        ordinals.append(device.launch_ordinal)
    if len(set(ordinals)) != len(ordinals):
        raise HardwareBindingError("accepted launch ordinals are not unique")
    return tuple(ordinals)
