"""PWR04: the sysfs writes that apply, and restore, one card's configured power settings.

The writes depend on the card's kernel driver, so they come from a per-driver backend chosen by the driver the card is
bound to (``/sys/bus/pci/devices/<locator>/driver``); a card whose driver has no backend is refused, never written to.
Backends: ``amdgpu`` (OverDrive table ``pp_od_clk_voltage``, staged and committed with "c", reset with "r"; board
power cap ``power1_cap`` in microwatts under the card's hwmon directory, which the caller resolves).

Pure: these functions only build (path, value) writes; the caller performs them and reads them back.
"""

from __future__ import annotations

import os

from bigcherry.core.environment import Device, DevicePower


def driver_of(locator: str) -> str | None:
    """The kernel driver bound to the PCI device, or None."""
    link = f"/sys/bus/pci/devices/{locator}/driver"
    return os.path.basename(os.path.realpath(link)) if os.path.islink(link) else None


class _Amdgpu:
    @staticmethod
    def apply(device: Device, power: DevicePower, hwmon_dir: str | None) -> list[tuple[str, str]]:
        od = f"/sys/bus/pci/devices/{device.locator}/pp_od_clk_voltage"
        writes: list[tuple[str, str]] = []
        if power.sclk_max_mhz is not None:
            writes.append((od, f"s 1 {power.sclk_max_mhz}"))
        if power.vddgfx_offset_mv is not None:
            writes.append((od, f"vo {power.vddgfx_offset_mv}"))
        if writes:
            writes.append((od, "c"))
        if power.power_cap_w is not None:
            if not hwmon_dir:
                raise ValueError(f"device {device.index}: a power cap needs the card's hwmon directory")
            writes.append((f"{hwmon_dir}/power1_cap", str(power.power_cap_w * 1_000_000)))
        return writes

    @staticmethod
    def restore(device: Device, hwmon_dir: str | None, default_cap_uw: int | None) -> list[tuple[str, str]]:
        od = f"/sys/bus/pci/devices/{device.locator}/pp_od_clk_voltage"
        writes = [(od, "r"), (od, "c")]
        if hwmon_dir and default_cap_uw is not None:
            writes.append((f"{hwmon_dir}/power1_cap", str(default_cap_uw)))
        return writes


BACKENDS = {"amdgpu": _Amdgpu}


def _backend(device: Device, driver: str | None):
    if not device.locator:
        raise ValueError(f"device {device.index}: no locator (PCI address); power settings need one")
    if driver not in BACKENDS:
        raise ValueError(f"device {device.index} ({device.locator}): no power backend for driver {driver!r} "
                         f"(supported: {', '.join(sorted(BACKENDS))})")
    return BACKENDS[driver]


def apply_writes(device: Device, power: DevicePower, driver: str | None,
                 hwmon_dir: str | None) -> list[tuple[str, str]]:
    """The writes that put ``power`` on ``device`` (bound to ``driver``), in order."""
    return _backend(device, driver).apply(device, power, hwmon_dir)


def restore_writes(device: Device, driver: str | None, hwmon_dir: str | None,
                   default_cap_uw: int | None) -> list[tuple[str, str]]:
    """The writes that return ``device`` to its driver defaults (and the default power cap when one was changed)."""
    return _backend(device, driver).restore(device, hwmon_dir, default_cap_uw)
