"""PWR04: the sysfs writes that apply, and restore, one card's configured power settings (amdgpu OverDrive and hwmon).

Pure: these functions only build the (path, value) writes from a device's locator and its ``DevicePower``; a privileged
helper performs them and reads them back. OverDrive changes are staged in ``pp_od_clk_voltage`` and take effect on the
commit write ("c"); "r" resets the staged table to the card's defaults. The power cap is ``power1_cap`` in microwatts
under the card's hwmon directory, which the caller resolves (its index is not stable across boots).
"""

from __future__ import annotations

from bigcherry.core.environment import Device, DevicePower


def _od_path(device: Device) -> str:
    if not device.locator:
        raise ValueError(f"device {device.index}: no locator (PCI address); power settings need one")
    return f"/sys/bus/pci/devices/{device.locator}/pp_od_clk_voltage"


def apply_writes(device: Device, power: DevicePower, hwmon_dir: str | None) -> list[tuple[str, str]]:
    """The writes that put ``power`` on ``device``, in order. ``hwmon_dir`` is needed only for a power cap."""
    od = _od_path(device)
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


def restore_writes(device: Device, hwmon_dir: str | None, default_cap_uw: int | None) -> list[tuple[str, str]]:
    """The writes that return ``device`` to its driver defaults: reset and commit the OverDrive table, and put back the
    default power cap (``power1_cap_default``, read by the caller) when one was changed."""
    od = _od_path(device)
    writes = [(od, "r"), (od, "c")]
    if hwmon_dir and default_cap_uw is not None:
        writes.append((f"{hwmon_dir}/power1_cap", str(default_cap_uw)))
    return writes
