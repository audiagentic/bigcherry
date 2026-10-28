"""PWR04: the sysfs writes that apply, and restore, one card's configured power settings.

The writes depend on the card's kernel driver, so they come from a per-driver backend chosen by the driver the card is
bound to (``/sys/bus/pci/devices/<locator>/driver``); a card whose driver has no backend is refused, never written to.
Backends: ``amdgpu`` (OverDrive table ``pp_od_clk_voltage``, staged and committed with "c", reset with "r"; board
power cap ``power1_cap`` in microwatts under the card's hwmon directory, which the caller resolves).

Pure: these functions only build (path, value) writes, the files they touch, and the udev rules that let a group write
those files (so an unprivileged launch can apply them); the caller performs the writes and reads them back.
"""

from __future__ import annotations

import os
import re

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

    @staticmethod
    def udev(device: Device, group: str) -> list[str]:
        return _amdgpu_udev(device, group)


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


def targets(device: Device, power: DevicePower, driver: str | None, hwmon_dir: str | None) -> list[str]:
    """The files that applying (and restoring) ``power`` on ``device`` writes, once each, in write order."""
    return list(dict.fromkeys(path for path, _ in apply_writes(device, power, driver, hwmon_dir)))


def udev_rules(cards: list[tuple[Device, str | None]], group: str) -> str:
    """A udev rules file that lets ``group`` write the power files of exactly these (device, driver) cards.

    The rules match each card by its PCI address, so other cards on the host are untouched; they run when the driver
    binds, so the permission survives reboots and driver reloads."""
    if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", group):
        raise ValueError(f"not a group name: {group!r}")
    lines = ["# bigcherry gpu-power: group write on the power settings of the listed cards only (bigcherry gpu-power grant)"]
    for device, driver in cards:
        lines += _backend(device, driver).udev(device, group)
    return "\n".join(lines) + "\n"


def _udev_chgrp(match: str, path: str, group: str) -> str:
    return (f'{match}, RUN+="/bin/sh -c \'chgrp {group} /sys%p/{path} && chmod g+w /sys%p/{path}\'"')


def _amdgpu_udev(device: Device, group: str) -> list[str]:
    return [
        _udev_chgrp(f'ACTION=="add|bind|change", SUBSYSTEM=="pci", KERNEL=="{device.locator}", DRIVER=="amdgpu"',
                    "pp_od_clk_voltage", group),
        _udev_chgrp(f'ACTION=="add|change", SUBSYSTEM=="hwmon", KERNELS=="{device.locator}"', "power1_cap", group),
    ]

