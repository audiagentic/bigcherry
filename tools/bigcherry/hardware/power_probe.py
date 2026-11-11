"""PWR07: portable, read-only discovery of CPU power interfaces (cpufreq policies, pstate drivers, boost, cpuidle,
energy counters). Observation only: nothing here writes, and nothing raises to the caller -- every field is an
``Observation`` that says whether it was read and, if not, why.

Capabilities are discovered, never inferred from the CPU vendor: an interface that is absent is reported ``unsupported``
with a reason, not emulated. Every path resolves under an injectable ``sysfs_root`` (tests build a fake tree) and the
platform is an argument so non-Linux behaviour is testable anywhere. Only the standard library is imported, and no
POSIX-only module at import time.
"""

from __future__ import annotations

import errno
import os
import re as _re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Status = Literal["supported", "unsupported", "unreadable", "not_permitted"]


@dataclass(frozen=True)
class Observation:
    """One field: its value when read, otherwise why not. ``value`` is None unless ``status`` is "supported"."""

    status: Status
    value: object | None = None
    unit: str | None = None
    source: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _unavailable(status: Status, reason: str, source: str | None, unit: str | None = None) -> Observation:
    return Observation(status=status, value=None, unit=unit, source=source, reason=reason)


def read_field(path: Path, *, unit: str | None = None) -> Observation:
    """Read one sysfs text file into an Observation; an error becomes a status and a stable reason, never an exception."""
    source = str(path)
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return _unavailable("unsupported", "interface_absent", source, unit)
    except NotADirectoryError:
        return _unavailable("unsupported", "interface_absent", source, unit)
    except PermissionError:
        return _unavailable("not_permitted", "access_denied", source, unit)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            return _unavailable("not_permitted", "access_denied", source, unit)
        if exc.errno == errno.EROFS:
            return _unavailable("not_permitted", "read_only_filesystem", source, unit)
        if exc.errno in (errno.ENOENT, errno.ENODEV, errno.ENOTDIR):
            return _unavailable("unsupported", "interface_absent", source, unit)
        return _unavailable("unreadable", f"io_error:{errno.errorcode.get(exc.errno or 0, 'unknown')}", source, unit)
    except UnicodeDecodeError:
        return _unavailable("unreadable", "invalid_contents", source, unit)
    return Observation(status="supported", value=text, unit=unit, source=source)


def read_int(path: Path, *, unit: str | None = None) -> Observation:
    obs = read_field(path, unit=unit)
    if obs.status != "supported":
        return obs
    try:
        return Observation(status="supported", value=int(str(obs.value), 0), unit=unit, source=obs.source)
    except ValueError:
        return _unavailable("unreadable", "invalid_contents", obs.source, unit)


def read_words(path: Path) -> Observation:
    """A whitespace-separated list (available governors, EPP choices) as a list of strings."""
    obs = read_field(path)
    if obs.status != "supported":
        return obs
    return Observation(status="supported", value=str(obs.value).split(), source=obs.source)


def _guard(fn, source: str) -> Observation:
    """Last-resort boundary: a probe bug or odd filesystem becomes an unreadable field, never a raised exception."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - the contract is that discovery never raises
        return _unavailable("unreadable", f"probe_error:{type(exc).__name__}", source)


@dataclass(frozen=True)
class CpuPolicy:
    policy: str
    driver: Observation
    governor: Observation
    available_governors: Observation
    epp: Observation
    available_epp: Observation

    def to_dict(self) -> dict:
        return {k: (v.to_dict() if isinstance(v, Observation) else v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class EnergyDomain:
    """One powercap or hwmon energy counter; ``max_range_uj`` is what a later delta needs to handle wrap-around."""

    provider: str
    name: Observation
    energy_uj: Observation
    max_range_uj: Observation

    def to_dict(self) -> dict:
        return {k: (v.to_dict() if isinstance(v, Observation) else v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class CpuProbe:
    platform: str
    vendor: Observation
    model: Observation
    policies: tuple[CpuPolicy, ...]
    boost: Observation
    intel_no_turbo: Observation
    amd_pstate_status: Observation
    cpuidle: Observation
    energy_domains: tuple[EnergyDomain, ...]
    energy_status: Observation

    def to_dict(self) -> dict:
        out: dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Observation):
                out[key] = value.to_dict()
            elif isinstance(value, tuple):
                out[key] = [item.to_dict() for item in value]
            else:
                out[key] = value
        return out


def _platform_is_linux(platform_name: str | None) -> tuple[str, bool]:
    name = platform_name if platform_name is not None else sys.platform
    return name, name.startswith("linux")


def _cpuinfo(root: Path) -> tuple[Observation, Observation]:
    path = root / "proc" / "cpuinfo"
    obs = read_field(path)
    if obs.status != "supported":
        return obs, obs
    vendor = model = None
    for line in str(obs.value).splitlines():
        key, _, val = line.partition(":")
        key = key.strip()
        if vendor is None and key == "vendor_id":
            vendor = val.strip()
        elif model is None and key == "model name":
            model = val.strip()
        if vendor and model:
            break

    def make(value: str | None, what: str) -> Observation:
        if value:
            return Observation(status="supported", value=value, source=str(path))
        return _unavailable("unsupported", f"{what}_not_reported", str(path))

    return make(vendor, "vendor"), make(model, "model")


def _policies(root: Path) -> tuple[CpuPolicy, ...]:
    base = root / "sys" / "devices" / "system" / "cpu" / "cpufreq"
    try:
        names = sorted(p.name for p in base.iterdir() if p.name.startswith("policy") and p.name[6:].isdigit())
    except OSError:
        return ()
    names.sort(key=lambda n: int(n[6:]))
    policies = []
    for name in names:
        d = base / name
        policies.append(
            CpuPolicy(
                policy=name,
                driver=read_field(d / "scaling_driver"),
                governor=read_field(d / "scaling_governor"),
                available_governors=read_words(d / "scaling_available_governors"),
                epp=read_field(d / "energy_performance_preference"),
                available_epp=read_words(d / "energy_performance_available_preferences"),
            )
        )
    return tuple(policies)


def _cpuidle(root: Path) -> Observation:
    """Which idle-state counters exist on cpu0 (names and whether usage/time are readable); not per-cpu residency."""
    base = root / "sys" / "devices" / "system" / "cpu" / "cpu0" / "cpuidle"
    source = str(base)
    try:
        states = sorted(p for p in base.iterdir() if p.name.startswith("state") and p.name[5:].isdigit())
    except FileNotFoundError:
        return _unavailable("unsupported", "interface_absent", source)
    except PermissionError:
        return _unavailable("not_permitted", "access_denied", source)
    except OSError:
        return _unavailable("unreadable", "io_error", source)
    states.sort(key=lambda p: int(p.name[5:]))
    out = []
    for st in states:
        name = read_field(st / "name")
        out.append(
            {
                "state": st.name,
                "name": name.value if name.status == "supported" else None,
                "usage": read_int(st / "usage").status,
                "time": read_int(st / "time", unit="microseconds").status,
            }
        )
    return Observation(status="supported", value=out, source=source)


def _energy(root: Path) -> tuple[tuple[EnergyDomain, ...], Observation]:
    """Every readable-or-not energy counter, regardless of CPU vendor: powercap zones and ``amd_energy`` hwmon."""
    domains: list[EnergyDomain] = []
    pc = root / "sys" / "class" / "powercap"
    try:
        zones = sorted(p for p in pc.iterdir() if (p / "energy_uj").exists() or (p / "name").exists())
    except OSError:
        zones = []
    for zone in zones:
        domains.append(
            EnergyDomain(
                provider="powercap",
                name=read_field(zone / "name"),
                energy_uj=read_int(zone / "energy_uj", unit="microjoules"),
                max_range_uj=read_int(zone / "max_energy_range_uj", unit="microjoules"),
            )
        )
    hw = root / "sys" / "class" / "hwmon"
    try:
        hwmons = sorted(p for p in hw.iterdir() if p.name.startswith("hwmon"))
    except OSError:
        hwmons = []
    for mon in hwmons:
        name = read_field(mon / "name")
        if name.status != "supported" or name.value != "amd_energy":
            continue
        try:
            inputs = sorted(p for p in mon.iterdir() if p.name.startswith("energy") and p.name.endswith("_input"))
        except OSError:
            inputs = []
        for inp in inputs:
            domains.append(
                EnergyDomain(
                    provider="hwmon:amd_energy",
                    name=Observation(status="supported", value=f"{mon.name}/{inp.name}", source=str(inp)),
                    energy_uj=read_int(inp, unit="microjoules"),
                    max_range_uj=_unavailable("unsupported", "range_not_exposed", str(inp)),
                )
            )
    if not domains:
        return (), _unavailable("unsupported", "no_energy_counters", str(pc))
    readable = [d for d in domains if d.energy_uj.status == "supported"]
    if not readable:
        reason = domains[0].energy_uj.reason or "unreadable"
        status: Status = domains[0].energy_uj.status
        return tuple(domains), _unavailable(status, reason, str(pc))
    return tuple(domains), Observation(status="supported", value=len(readable), source=str(pc))


def probe_cpu(*, sysfs_root: Path = Path("/"), platform_name: str | None = None) -> CpuProbe:
    """Discover the CPU's power interfaces. Never raises; never writes."""
    plat, is_linux = _platform_is_linux(platform_name)
    if not is_linux:
        gone = _unavailable("unsupported", "platform_not_linux", None)
        return CpuProbe(plat, gone, gone, (), gone, gone, gone, gone, (), gone)
    root = Path(sysfs_root)
    if not (root / "sys").is_dir():
        gone = _unavailable("unsupported", "sysfs_unavailable", str(root / "sys"))
        return CpuProbe(plat, gone, gone, (), gone, gone, gone, gone, (), gone)
    cpu = root / "sys" / "devices" / "system" / "cpu"
    vendor, model = _guard_pair(lambda: _cpuinfo(root), str(root / "proc" / "cpuinfo"))
    try:
        policies = _policies(root)
    except Exception:  # noqa: BLE001
        policies = ()
    try:
        energy, energy_status = _energy(root)
    except Exception as exc:  # noqa: BLE001
        energy = ()
        energy_status = _unavailable("unreadable", f"probe_error:{type(exc).__name__}", None)
    # Boost and turbo are independent interfaces with their own semantics: report each as the kernel states it.
    return CpuProbe(
        platform=plat,
        vendor=vendor,
        model=model,
        policies=policies,
        boost=_guard(lambda: read_field(cpu / "cpufreq" / "boost"), str(cpu / "cpufreq" / "boost")),
        intel_no_turbo=_guard(lambda: read_field(cpu / "intel_pstate" / "no_turbo"), str(cpu / "intel_pstate")),
        amd_pstate_status=_guard(lambda: read_field(cpu / "amd_pstate" / "status"), str(cpu / "amd_pstate")),
        cpuidle=_guard(lambda: _cpuidle(root), str(cpu / "cpu0" / "cpuidle")),
        energy_domains=energy,
        energy_status=energy_status,
    )


def _guard_pair(fn, source: str) -> tuple[Observation, Observation]:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        gone = _unavailable("unreadable", f"probe_error:{type(exc).__name__}", source)
        return gone, gone


# --------------------------------------------------------------------------------------------------------------------
# GPU capability probe (PWR07 slice 2). Per physical PCI device; amdgpu OverDrive is recognised from its table, with the
# device-reported OD_RANGE as the only source of limits. A control the parser cannot positively identify, with its range
# and current value, is "unsupported" for application; the rest of the card's observations continue.


@dataclass(frozen=True)
class ControlCapability:
    status: Status
    current: Observation
    minimum: int | None
    maximum: int | None
    unit: str
    reason: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass(frozen=True)
class GpuCapability:
    bdf: str
    vendor: Observation
    device: Observation
    driver: Observation
    controls: dict
    sensors: dict
    hwmon: Observation

    def to_dict(self) -> dict:
        return {
            "bdf": self.bdf,
            "vendor": self.vendor.to_dict(),
            "device": self.device.to_dict(),
            "driver": self.driver.to_dict(),
            "controls": {k: v.to_dict() for k, v in self.controls.items()},
            "sensors": {k: v.to_dict() for k, v in self.sensors.items()},
            "hwmon": self.hwmon.to_dict(),
        }


@dataclass(frozen=True)
class PowerProbe:
    cpu: CpuProbe
    gpus: tuple[GpuCapability, ...]

    def to_dict(self) -> dict:
        return {"cpu": self.cpu.to_dict(), "gpus": [g.to_dict() for g in self.gpus]}


#: the controls an amdgpu card can carry; every one is optional per card
CONTROLS = ("vddgfx_offset_mv", "sclk_max_mhz", "mclk_max_mhz", "sclk_offset_mhz", "power_cap_w")


def _no_control(reason: str, status: Status = "unsupported", unit: str = "") -> ControlCapability:
    return ControlCapability(status, _unavailable(status, reason, None, unit or None), None, None, unit, reason)


def _od_section(text: str, header: str) -> list[str] | None:
    """Lines of the OverDrive table section that starts at ``header:`` and ends at the next header."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == f"{header}:":
            out = []
            for nxt in lines[i + 1:]:
                if _re.match(r"^[A-Z_]+:\s*$", nxt.strip()) or not nxt.strip():
                    break
                out.append(nxt.strip())
            return out
    return None


def _od_range(text: str, key: str, unit: str) -> tuple[int, int] | None:
    """The device-reported ``key:   LOmhz  HImhz`` pair of the OD_RANGE section, or None."""
    sec = _od_section(text, "OD_RANGE")
    if not sec:
        return None
    for line in sec:
        m = _re.match(rf"^{key}:\s*(-?\d+)\s*{unit}\s+(-?\d+)\s*{unit}\s*$", line, _re.IGNORECASE)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def _writable(path: Path) -> bool:
    """Preliminary only: the kernel can still refuse a write that permission bits allow."""
    try:
        return os.access(path, os.W_OK)
    except OSError:
        return False


def _overdrive_controls(od_path: Path) -> dict:
    obs = read_field(od_path)
    if obs.status != "supported":
        reason = "overdrive_absent" if obs.reason == "interface_absent" else (obs.reason or "unreadable")
        return {k: _no_control(reason, obs.status, u) for k, u in
                (("vddgfx_offset_mv", "mV"), ("sclk_max_mhz", "MHz"), ("mclk_max_mhz", "MHz"), ("sclk_offset_mhz", "MHz"))}
    text = str(obs.value)
    src = str(od_path)
    out: dict = {}

    # voltage offset: needs the OD_VDDGFX_OFFSET current value AND a VDDGFX_OFFSET range from OD_RANGE
    sec = _od_section(text, "OD_VDDGFX_OFFSET")
    rng = _od_range(text, "VDDGFX_OFFSET", "mv")
    cur = _re.match(r"^(-?\d+)\s*mV$", sec[0], _re.IGNORECASE) if sec else None
    if cur and rng:
        out["vddgfx_offset_mv"] = ControlCapability(
            "supported", Observation("supported", int(cur.group(1)), "mV", src), rng[0], rng[1], "mV")
    else:
        out["vddgfx_offset_mv"] = _no_control("range_not_reported" if cur else "control_not_recognised", unit="mV")

    # maximum clock, absolute form (OD_SCLK rows 0/1) with an SCLK range
    sec = _od_section(text, "OD_SCLK")
    rng = _od_range(text, "SCLK", "mhz")
    rows = {}
    for line in sec or []:
        m = _re.match(r"^(\d+):\s*(\d+)\s*Mhz$", line, _re.IGNORECASE)
        if m:
            rows[int(m.group(1))] = int(m.group(2))
    if 1 in rows and 0 in rows and rng:
        out["sclk_max_mhz"] = ControlCapability(
            "supported", Observation("supported", rows[1], "MHz", src), rng[0], rng[1], "MHz")
    else:
        out["sclk_max_mhz"] = _no_control("control_not_recognised", unit="MHz")

    # Absolute memory clock, OD_MCLK rows 0/1 and the card's OD_RANGE MCLK bounds.
    sec = _od_section(text, "OD_MCLK")
    rng = _od_range(text, "MCLK", "mhz")
    rows = {}
    for line in sec or []:
        m = _re.match(r"^(\d+):\s*(\d+)\s*MHz$", line, _re.IGNORECASE)
        if m:
            rows[int(m.group(1))] = int(m.group(2))
    if 0 in rows and 1 in rows and rng and rng[0] <= rng[1]:
        out["mclk_max_mhz"] = ControlCapability(
            "supported", Observation("supported", rows[1], "MHz", src), rng[0], rng[1], "MHz")
    else:
        out["mclk_max_mhz"] = _no_control("control_not_recognised", unit="MHz")

    # newer cards: a signed offset instead of a table
    sec = _od_section(text, "OD_SCLK_OFFSET")
    rng = _od_range(text, "SCLK_OFFSET", "mhz")
    cur = _re.match(r"^(-?\d+)\s*Mhz$", sec[0], _re.IGNORECASE) if sec else None
    if cur and rng:
        out["sclk_offset_mhz"] = ControlCapability(
            "supported", Observation("supported", int(cur.group(1)), "MHz", src), rng[0], rng[1], "MHz")
    else:
        out["sclk_offset_mhz"] = _no_control("control_not_recognised", unit="MHz")
    for k, c in out.items():
        if c.status == "supported" and not _writable(od_path):
            out[k] = ControlCapability("not_permitted", c.current, c.minimum, c.maximum, c.unit, "not_writable")
    return out


def _pick_hwmon(dev: Path) -> tuple[Path | None, Observation]:
    """The one amdgpu hwmon directory under the device; zero or several is not a controller we may write through."""
    base = dev / "hwmon"
    src = str(base)
    try:
        mons = sorted(p for p in base.iterdir() if p.name.startswith("hwmon"))
    except FileNotFoundError:
        return None, _unavailable("unsupported", "interface_absent", src)
    except PermissionError:
        return None, _unavailable("not_permitted", "access_denied", src)
    except OSError:
        return None, _unavailable("unreadable", "io_error", src)
    named = [m for m in mons if read_field(m / "name").value == "amdgpu"]
    if len(named) == 1:
        return named[0], Observation("supported", named[0].name, source=str(named[0]))
    if not named:
        return None, _unavailable("unsupported", "no_matching_hwmon", src)
    return None, _unavailable("unreadable", "ambiguous_hwmon", src)


def _cap_control(hw: Path | None) -> ControlCapability:
    if hw is None:
        return _no_control("hwmon_unavailable", unit="microwatts")
    cur = read_int(hw / "power1_cap", unit="microwatts")
    lo = read_int(hw / "power1_cap_min", unit="microwatts")
    hi = read_int(hw / "power1_cap_max", unit="microwatts")
    for o in (cur, lo, hi):
        if o.status != "supported":
            return ControlCapability(o.status, cur, None, None, "microwatts", o.reason)
    if lo.value > hi.value:
        return ControlCapability("unreadable", cur, None, None, "microwatts", "range_inverted")
    status: Status = "supported" if _writable(hw / "power1_cap") else "not_permitted"
    return ControlCapability(status, cur, int(lo.value), int(hi.value), "microwatts",
                             None if status == "supported" else "not_writable")


def _gpu(dev: Path, bdf: str) -> GpuCapability:
    drv_link = dev / "driver"
    try:
        driver = Observation("supported", os.path.basename(os.path.realpath(drv_link)), source=str(drv_link)) \
            if drv_link.is_symlink() else _unavailable("unsupported", "no_driver_bound", str(drv_link))
    except OSError:
        driver = _unavailable("unreadable", "io_error", str(drv_link))
    hw, hw_obs = _pick_hwmon(dev)
    if driver.value == "amdgpu":
        controls = _overdrive_controls(dev / "pp_od_clk_voltage")
        controls["power_cap_w"] = _cap_control(hw)
    else:
        controls = {k: _no_control("no_backend_for_driver") for k in CONTROLS}
    sensors = {}
    if hw is not None:
        for key, fname, unit in (("power_w_average", "power1_average", "microwatts"),
                                 ("temp_c", "temp1_input", "millidegrees_c"),
                                 ("energy_uj", "energy1_input", "microjoules")):
            sensors[key] = read_int(hw / fname, unit=unit)
    return GpuCapability(
        bdf=bdf,
        vendor=read_field(dev / "vendor"),
        device=read_field(dev / "device"),
        driver=driver,
        controls=controls,
        sensors=sensors,
        hwmon=hw_obs,
    )


def probe_gpus(bdfs, *, sysfs_root: Path = Path("/"), platform_name: str | None = None) -> tuple[GpuCapability, ...]:
    """Capabilities of exactly the named PCI devices; never raises, never writes."""
    _, is_linux = _platform_is_linux(platform_name)
    out = []
    for bdf in bdfs:
        dev = Path(sysfs_root) / "sys" / "bus" / "pci" / "devices" / bdf
        if not is_linux:
            gone = _unavailable("unsupported", "platform_not_linux", None)
            out.append(GpuCapability(bdf, gone, gone, gone, {k: _no_control("platform_not_linux") for k in CONTROLS}, {}, gone))
            continue
        try:
            if not dev.is_dir():
                gone = _unavailable("unsupported", "device_absent", str(dev))
                out.append(GpuCapability(bdf, gone, gone, gone, {k: _no_control("device_absent") for k in CONTROLS}, {}, gone))
                continue
            out.append(_gpu(dev, bdf))
        except Exception as exc:  # noqa: BLE001
            gone = _unavailable("unreadable", f"probe_error:{type(exc).__name__}", str(dev))
            out.append(GpuCapability(bdf, gone, gone, gone, {k: _no_control("probe_error") for k in CONTROLS}, {}, gone))
    return tuple(out)


def probe_host(bdfs, *, sysfs_root: Path = Path("/"), platform_name: str | None = None) -> PowerProbe:
    return PowerProbe(cpu=probe_cpu(sysfs_root=sysfs_root, platform_name=platform_name),
                      gpus=probe_gpus(bdfs, sysfs_root=sysfs_root, platform_name=platform_name))
