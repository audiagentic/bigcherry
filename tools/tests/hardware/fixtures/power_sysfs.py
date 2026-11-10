"""Fake sysfs trees for the power probe tests (PWR07). Each builder writes a small tree under a temp root."""

from __future__ import annotations

from pathlib import Path


def _w(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text + "\n", encoding="utf-8")


def _cpuinfo(root: Path, vendor: str, model: str) -> None:
    _w(root, "proc/cpuinfo", f"processor\t: 0\nvendor_id\t: {vendor}\nmodel name\t: {model}\n")


def _policy(root: Path, n: int, driver: str, gov: str, govs: str, epp: str | None = None, epps: str | None = None) -> None:
    base = f"sys/devices/system/cpu/cpufreq/policy{n}"
    _w(root, f"{base}/scaling_driver", driver)
    _w(root, f"{base}/scaling_governor", gov)
    _w(root, f"{base}/scaling_available_governors", govs)
    if epp is not None:
        _w(root, f"{base}/energy_performance_preference", epp)
        _w(root, f"{base}/energy_performance_available_preferences", epps or epp)


def intel_pstate_active_hwp(root: Path) -> Path:
    _cpuinfo(root, "GenuineIntel", "Intel(R) Core(TM) i9")
    for n in range(2):
        _policy(root, n, "intel_pstate", "powersave", "performance powersave", "balance_performance",
                "default performance balance_performance balance_power power")
    _w(root, "sys/devices/system/cpu/intel_pstate/no_turbo", "0")
    _w(root, "sys/class/powercap/intel-rapl:0/name", "package-0")
    _w(root, "sys/class/powercap/intel-rapl:0/energy_uj", "123456789")
    _w(root, "sys/class/powercap/intel-rapl:0/max_energy_range_uj", "262143328850")
    _w(root, "sys/devices/system/cpu/cpu0/cpuidle/state0/name", "POLL")
    _w(root, "sys/devices/system/cpu/cpu0/cpuidle/state0/usage", "10")
    _w(root, "sys/devices/system/cpu/cpu0/cpuidle/state0/time", "99")
    return root


def intel_pstate_passive(root: Path) -> Path:
    _cpuinfo(root, "GenuineIntel", "Intel(R) Xeon(R)")
    _policy(root, 0, "intel_cpufreq", "schedutil", "performance schedutil")
    _w(root, "sys/devices/system/cpu/intel_pstate/no_turbo", "1")
    return root


def intel_acpi_cpufreq(root: Path) -> Path:
    _cpuinfo(root, "GenuineIntel", "Intel(R) Xeon(R)")
    _policy(root, 0, "acpi-cpufreq", "ondemand", "ondemand performance")
    _w(root, "sys/devices/system/cpu/cpufreq/boost", "1")
    return root


def amd_pstate_active(root: Path) -> Path:
    _cpuinfo(root, "AuthenticAMD", "AMD EPYC 9654")
    for n in range(4):
        _policy(root, n, "amd-pstate-epp", "powersave", "performance powersave", "performance",
                "default performance balance_performance balance_power power")
    _w(root, "sys/devices/system/cpu/amd_pstate/status", "active")
    _w(root, "sys/devices/system/cpu/cpufreq/boost", "1")
    _w(root, "sys/class/hwmon/hwmon3/name", "amd_energy")
    _w(root, "sys/class/hwmon/hwmon3/energy1_input", "5000000")
    _w(root, "sys/class/hwmon/hwmon3/energy2_input", "6000000")
    return root


def amd_pstate_guided(root: Path) -> Path:
    _cpuinfo(root, "AuthenticAMD", "AMD Ryzen 9")
    _policy(root, 0, "amd-pstate", "schedutil", "performance schedutil")
    _w(root, "sys/devices/system/cpu/amd_pstate/status", "guided")
    return root


def amd_acpi_cpufreq(root: Path) -> Path:
    _cpuinfo(root, "AuthenticAMD", "AMD EPYC 7002")
    _policy(root, 0, "acpi-cpufreq", "ondemand", "ondemand performance")
    return root


def vm_without_cpufreq(root: Path) -> Path:
    _cpuinfo(root, "GenuineIntel", "Intel(R) Xeon(R) virtual")
    (root / "sys/devices/system/cpu").mkdir(parents=True, exist_ok=True)
    return root


# --- GPUs: the OverDrive tables below are copied from real cards on the dev host (gfx1100, gfx1201, gfx1030) -------

OD_RDNA3 = """OD_SCLK:
0: 500Mhz
1: 3010Mhz
OD_MCLK:
0: 97Mhz
1: 1250MHz
OD_RANGE:
SCLK:     500Mhz       5000Mhz
MCLK:      97Mhz       1500Mhz
VDDGFX_OFFSET:    -450mv          0mv
OD_VDDGFX_OFFSET:
0mV
"""
OD_RDNA4 = """OD_SCLK_OFFSET:
0Mhz
OD_MCLK:
0: 97Mhz
1: 1259MHz
OD_RANGE:
SCLK_OFFSET:    -500Mhz       1000Mhz
MCLK:      97Mhz       1500Mhz
VDDGFX_OFFSET:    -200mv          0mv
OD_VDDGFX_OFFSET:
0mV
"""
OD_RDNA2 = """OD_SCLK:
0: 500Mhz
1: 2574Mhz
OD_MCLK:
0: 97Mhz
1: 1000MHz
OD_RANGE:
SCLK:     500Mhz       4000Mhz
MCLK:     674Mhz       1312Mhz
OD_VDDGFX_OFFSET:
0mV
"""


def gpu(root: Path, bdf: str, *, driver: str = "amdgpu", od: str | None = OD_RDNA3, cap: tuple[int, int, int] | None = (303, 272, 348),
        hwmons: tuple[str, ...] = ("hwmon8",), device: str = "0x744c") -> Path:
    dev = root / "sys/bus/pci/devices" / bdf
    dev.mkdir(parents=True, exist_ok=True)
    _w(root, f"sys/bus/pci/devices/{bdf}/class", "0x030000")
    _w(root, f"sys/bus/pci/devices/{bdf}/vendor", "0x1002")
    _w(root, f"sys/bus/pci/devices/{bdf}/device", device)
    drv = root / "sys/bus/pci/drivers" / driver
    drv.mkdir(parents=True, exist_ok=True)
    link = dev / "driver"
    if not link.exists():
        link.symlink_to(drv)
    if od is not None:
        _w(root, f"sys/bus/pci/devices/{bdf}/pp_od_clk_voltage", od.rstrip("\n"))
    for i, mon in enumerate(hwmons):
        base = f"sys/bus/pci/devices/{bdf}/hwmon/{mon}"
        _w(root, f"{base}/name", "amdgpu")
        _w(root, f"{base}/power1_average", "120000000")
        _w(root, f"{base}/temp1_input", "45000")
        _w(root, f"{base}/energy1_input", "777")
        if cap is not None and i == 0:
            c, lo, hi = cap
            _w(root, f"{base}/power1_cap", str(c * 1_000_000))
            _w(root, f"{base}/power1_cap_min", str(lo * 1_000_000))
            _w(root, f"{base}/power1_cap_max", str(hi * 1_000_000))
    return root


def four_amdgpu_mixed(root: Path) -> Path:
    gpu(root, "0000:03:00.0", od=OD_RDNA3, cap=(303, 272, 348))
    gpu(root, "0000:06:00.0", od=OD_RDNA3, cap=(303, 272, 348), hwmons=("hwmon9",))
    gpu(root, "0000:09:00.0", od=OD_RDNA4, cap=(300, 210, 330), device="0x7551", hwmons=("hwmon10",))
    gpu(root, "0000:17:00.0", od=OD_RDNA2, cap=(303, 272, 339), device="0x73af", hwmons=("hwmon11",))
    return root
