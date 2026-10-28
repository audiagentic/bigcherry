"""PWR04: GPU power settings on Linux sysfs -- reading what a card runs, applying a setting, and putting back exactly what
ran before. Self-contained: the standard library and sysfs only; callers pass plain values (a PCI locator and a
``Setting``), wherever those come from.

The writes depend on the card's kernel driver, so they come from a per-driver backend chosen by the driver the card is
bound to (``/sys/bus/pci/devices/<locator>/driver``); a card whose driver has no backend is refused, never written to.
Backends: ``amdgpu`` (OverDrive table ``pp_od_clk_voltage``, staged and committed with "c"; board power cap
``power1_cap`` in microwatts under the card's hwmon directory).

``apply`` records each card's running values before changing anything (a ``State``, kept by the caller) and ``restore``
writes those values back, so a run leaves a card as it found it -- including values someone set before the run -- not
merely at the driver defaults.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import asdict, dataclass

SYSFS_PCI = "/sys/bus/pci/devices"


@dataclass(frozen=True)
class Setting:
    """What to put on one card; None leaves that value as it is."""

    vddgfx_offset_mv: int | None = None   #: OverDrive graphics voltage offset; negative undervolts
    sclk_max_mhz: int | None = None       #: OverDrive maximum shader clock
    power_cap_w: int | None = None        #: board power cap


@dataclass(frozen=True)
class State:
    """What one card runs now (None: unreadable). ``power_cap_uw`` is exact, so a restore is exact."""

    vddgfx_offset_mv: int | None = None
    sclk_max_mhz: int | None = None
    power_cap_uw: int | None = None
    power_cap_default_uw: int | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "State":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


def driver_of(locator: str) -> str | None:
    """The kernel driver bound to the PCI device, or None."""
    link = f"{SYSFS_PCI}/{locator}/driver"
    return os.path.basename(os.path.realpath(link)) if os.path.islink(link) else None


def hwmon_of(locator: str) -> str | None:
    found = sorted(glob.glob(f"{SYSFS_PCI}/{locator}/hwmon/hwmon*"))
    return found[0] if found else None


def display_cards(root: str = SYSFS_PCI) -> list[tuple[str, str | None]]:
    """Every display-class PCI device on the host (class 0x03xxxx): (locator, bound driver)."""
    out = []
    for path in sorted(glob.glob(f"{root}/*")):
        try:
            with open(f"{path}/class", encoding="ascii") as f:
                if not f.read().strip().startswith("0x03"):
                    continue
        except OSError:
            continue
        locator = os.path.basename(path)
        out.append((locator, driver_of(locator)))
    return out


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _int(text: str | None) -> int | None:
    try:
        return int(text.strip()) if text else None
    except ValueError:
        return None


class _Amdgpu:
    #: the kernel command-line option that turns on OverDrive (the voltage/clock table), off by default
    kernel_option = "amdgpu.ppfeaturemask=0xfff7ffff"

    @staticmethod
    def od(locator: str) -> str:
        return f"{SYSFS_PCI}/{locator}/pp_od_clk_voltage"

    @classmethod
    def overdrive_on(cls, locator: str) -> bool:
        return os.path.exists(cls.od(locator))

    @classmethod
    def files(cls, locator: str, hwmon: str | None) -> list[str]:
        return [cls.od(locator)] + ([f"{hwmon}/power1_cap"] if hwmon else [])

    @classmethod
    def read(cls, locator: str, hwmon: str | None) -> State:
        od = _read(cls.od(locator)) or ""
        sclk = re.search(r"OD_SCLK:\s*\n\s*0:\s*\d+Mhz\s*\n\s*1:\s*(\d+)Mhz", od)
        off = re.search(r"OD_VDDGFX_OFFSET:\s*\n\s*(-?\d+)mV", od)
        return State(
            vddgfx_offset_mv=int(off.group(1)) if off else None,
            sclk_max_mhz=int(sclk.group(1)) if sclk else None,
            power_cap_uw=_int(_read(f"{hwmon}/power1_cap")) if hwmon else None,
            power_cap_default_uw=_int(_read(f"{hwmon}/power1_cap_default")) if hwmon else None,
        )

    @classmethod
    def writes(cls, locator: str, hwmon: str | None, sclk: int | None, offset: int | None,
               cap_uw: int | None) -> list[tuple[str, str]]:
        od = cls.od(locator)
        out: list[tuple[str, str]] = []
        # A clock write and commit resets the voltage offset on these cards, so the offset is staged after the clock
        # and both are committed together.
        if sclk is not None:
            out.append((od, f"s 1 {sclk}"))
        if offset is not None:
            out.append((od, f"vo {offset}"))
        if out:
            out.append((od, "c"))
        if cap_uw is not None:
            if not hwmon:
                raise ValueError(f"{locator}: a power cap needs the card's hwmon directory")
            out.append((f"{hwmon}/power1_cap", str(cap_uw)))
        return out

    @staticmethod
    def udev(locator: str, group: str) -> list[str]:
        return [
            _udev_chgrp(f'ACTION=="add|bind|change", SUBSYSTEM=="pci", KERNEL=="{locator}", DRIVER=="amdgpu"',
                        "pp_od_clk_voltage", group),
            _udev_chgrp(f'ACTION=="add|change", SUBSYSTEM=="hwmon", KERNELS=="{locator}"', "power1_cap", group),
        ]


BACKENDS = {"amdgpu": _Amdgpu}


def backend(locator: str, driver: str | None):
    if not locator:
        raise ValueError("a card needs its locator (PCI address) for power settings")
    if driver not in BACKENDS:
        raise ValueError(f"{locator}: no power backend for driver {driver!r} (supported: {', '.join(sorted(BACKENDS))})")
    return BACKENDS[driver]


def read_state(locator: str, driver: str | None, hwmon: str | None) -> State:
    """What the card runs now."""
    return backend(locator, driver).read(locator, hwmon)


def apply_writes(locator: str, driver: str | None, hwmon: str | None, setting: Setting) -> list[tuple[str, str]]:
    """The writes that put ``setting`` on the card, in order."""
    return backend(locator, driver).writes(
        locator, hwmon, setting.sclk_max_mhz, setting.vddgfx_offset_mv,
        setting.power_cap_w * 1_000_000 if setting.power_cap_w is not None else None)


def restore_writes(locator: str, driver: str | None, hwmon: str | None, before: State,
                   setting: Setting) -> list[tuple[str, str]]:
    """The writes that put back what ``before`` recorded, for the values ``setting`` changed."""
    return backend(locator, driver).writes(
        locator, hwmon,
        before.sclk_max_mhz if setting.sclk_max_mhz is not None or setting.vddgfx_offset_mv is not None else None,
        before.vddgfx_offset_mv if setting.sclk_max_mhz is not None or setting.vddgfx_offset_mv is not None else None,
        before.power_cap_uw if setting.power_cap_w is not None else None)


def targets(locator: str, driver: str | None, hwmon: str | None, setting: Setting) -> list[str]:
    """The files that applying (and restoring) ``setting`` writes, once each, in write order."""
    return list(dict.fromkeys(path for path, _ in apply_writes(locator, driver, hwmon, setting)))


def power_files(locator: str, driver: str | None, hwmon: str | None) -> list[str]:
    """Every power file the backend can write on the card (what a grant covers)."""
    return backend(locator, driver).files(locator, hwmon)


def udev_rules(cards: list[tuple[str, str | None]], group: str) -> str:
    """A udev rules file that lets ``group`` write the power files of exactly these (locator, driver) cards.

    The rules match each card by its PCI address, so other devices are untouched; they run when the driver binds, so
    the permission survives reboots and driver reloads."""
    if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", group):
        raise ValueError(f"not a group name: {group!r}")
    lines = ["# bigcherry gpu-power: group write on the power settings of the listed cards only (bigcherry gpu-power grant)"]
    for locator, driver in cards:
        lines += backend(locator, driver).udev(locator, group)
    return "\n".join(lines) + "\n"


def _udev_chgrp(match: str, path: str, group: str) -> str:
    return f'{match}, RUN+="/bin/sh -c \'chgrp {group} /sys%p/{path} && chmod g+w /sys%p/{path}\'"'
