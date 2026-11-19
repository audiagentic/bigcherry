"""MRC02 platform host-facts: passive, engine-neutral sysfs/procfs snapshots.

File readers are pure over injectable roots. Optional external commands are
read-only and injected in tests; unavailable/zero/invalid values carry explicit
readability rather than a fabricated zero. This module never changes clocks,
power, PCIe or GPU leases.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

_SYS = Path("/sys")
_PROC = Path("/proc")
_BDF = re.compile(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\Z")
_FREQ = re.compile(r"(?im)^\s*\d+:\s*(\d+(?:\.\d+)?)\s*(M|G)Hz\b")
_PCI_FIELDS = {
    "current_speed": "current_link_speed",
    "current_width": "current_link_width",
    "max_speed": "max_link_speed",
    "max_width": "max_link_width",
}


def fact(value: Any, reason: str = "unreadable") -> dict[str, Any]:
    """A typed value with an independent readability verdict."""
    if value is None:
        return {"value": None, "readable": False, "reason": reason}
    return {"value": value, "readable": True}


def _text(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return value or None


def _number(
    path: Path | None, *, positive: bool = False, zero_reason: str = "invalid_zero"
) -> dict:
    raw = _text(path)
    if raw is None:
        return fact(None)
    if not re.fullmatch(r"[0-9]+", raw):
        return fact(None, "malformed")
    value = int(raw)
    if positive and value == 0:
        return fact(None, zero_reason)
    return fact(value)


def _hwmon(dev: Path, filename: str) -> Path | None:
    # Stable selection if more than one hwmon directory exists.
    return next(iter(sorted((dev / "hwmon").glob(f"hwmon*/{filename}"))), None)


def card_paths(sys_root: Path = _SYS) -> list[tuple[str, Path]]:
    """All DRM PCI cards by BDF; used by both workloads and the host collector."""
    found = {}
    for card in (Path(sys_root) / "class/drm").glob("card[0-9]*"):
        if re.fullmatch(r"card\d+", card.name) is None:
            continue
        dev = card / "device"
        if not dev.is_dir():
            continue
        bdf = dev.resolve().name
        if _BDF.fullmatch(bdf):
            found[bdf.lower()] = dev
    return sorted(found.items())


def _od_max(text: str | None, block: str) -> int | None:
    if text is None:
        return None
    part = re.search(rf"{block}:([\s\S]*?)(?=OD_[A-Z_]+:|\Z)", text)
    if part is None:
        # Newer amdgpu firmware can expose only OD_RANGE's named bounds.
        ranges = re.search(r"OD_RANGE:([\s\S]*?)(?=OD_[A-Z_]+:|\Z)", text)
        key = {"OD_SCLK": "SCLK", "OD_MCLK": "MCLK"}.get(block)
        if not ranges or not key:
            return None
        line = re.search(rf"(?im)^\s*{key}:\s*(.+)$", ranges.group(1))
        if line is None:
            return None
        values = [
            float(freq) * (1000 if unit.upper() == "G" else 1)
            for freq, unit in re.findall(
                r"(\d+(?:\.\d+)?)\s*(M|G)Hz\b", line.group(1), re.IGNORECASE
            )
        ]
        return int(max(values)) if values else None
    mhz = [
        float(value) * (1000 if unit.upper() == "G" else 1)
        for value, unit in _FREQ.findall(part.group(1))
    ]
    return int(max(mhz)) if mhz else None


def _dpm_current(path: Path) -> int | None:
    raw = _text(path)
    if raw is None:
        return None
    marked = [line for line in raw.splitlines() if "*" in line]
    if len(marked) != 1:
        return None
    hit = _FREQ.search(marked[0])
    if hit is None:
        return None
    value = float(hit.group(1)) * (1000 if hit.group(2).upper() == "G" else 1)
    return int(value)


def power_state(sys_root: Path = _SYS) -> dict[str, dict]:
    """Existing workload power/OD contract; cap 0 is unknown, not zero watts."""
    result = {}
    for bdf, dev in card_paths(sys_root):
        od = _text(dev / "pp_od_clk_voltage")
        vo = re.search(r"OD_VDDGFX_OFFSET:\s*\n?\s*(-?\d+)\s*m?V", od or "")
        result[bdf] = {
            "power1_cap_uw": _number(
                _hwmon(dev, "power1_cap"), positive=True, zero_reason="zero_cap_suspect"
            )["value"],
            "vddgfx_offset_mv": int(vo.group(1)) if vo else None,
            "od_sclk_max_mhz": _od_max(od, "OD_SCLK"),
            "od_mclk_max_mhz": _od_max(od, "OD_MCLK"),
            "perf_level": _text(dev / "power_dpm_force_performance_level"),
        }
    return result


def board_watts(sys_root: Path = _SYS) -> dict[str, float]:
    """Passive average board power, in watts, for workload phase sampling."""
    readings = {}
    for bdf, dev in card_paths(sys_root):
        val = _number(_hwmon(dev, "power1_average"))["value"]
        if val is not None:
            readings[bdf] = val / 1e6
    return readings


def _architecture(dev: Path) -> dict:
    version = _text(dev / "gfx_target_version")
    if version is None:
        return fact(None)
    if re.fullmatch(r"gfx[0-9a-z]+", version):
        return fact(version)
    match = re.fullmatch(r"(\d+)\.(\d)\.(\d)", version)
    if match:
        return fact("gfx" + "".join(match.groups()))
    return fact(None, "malformed")


def _link_speed(path: Path) -> dict:
    raw = _text(path)
    if raw is None:
        return fact(None)
    if not re.fullmatch(r"\d+(?:\.\d+)?\s+GT/s(?:\s+PCIe)?", raw, re.IGNORECASE):
        return fact(None, "malformed")
    return fact(raw)


def _pcie_port(dev: Path) -> dict:
    return {
        key: (
            _number(dev / filename, positive=True)
            if "width" in key
            else _link_speed(dev / filename)
        )
        for key, filename in _PCI_FIELDS.items()
    }


def _pcie_tree(dev: Path, sys_root: Path) -> dict:
    card = _pcie_port(dev)
    upstream = []
    resolved = dev.resolve()
    root = Path(sys_root).resolve()
    if not resolved.is_relative_to(root):
        return {"card": card, "upstream": upstream}
    for parent in resolved.parents:
        if parent == root:
            break
        if _BDF.fullmatch(parent.name):
            upstream.append({"bdf": fact(parent.name.lower()), **_pcie_port(parent)})
    return {"card": card, "upstream": upstream}


def _firmware_version(dev: Path, sys_root: Path) -> dict:
    raw = _text(dev / "firmware_version")
    if raw is not None:
        return fact(raw)
    # The amdgpu firmware inventory is on debugfs on several kernels, not
    # in the per-card device directory. It may require privileges: never
    # fail the run or mislabel a different card's firmware.
    for card in (Path(sys_root) / "class/drm").glob("card[0-9]*"):
        if not re.fullmatch(r"card\d+", card.name):
            continue
        if (card / "device").resolve() != dev.resolve():
            continue
        index = card.name.removeprefix("card")
        raw = _text(
            Path(sys_root) / "kernel/debug/dri" / index / "amdgpu_firmware_info"
        )
        if raw is not None:
            return fact(raw)
    return fact(None)


def _card_facts(dev: Path, sys_root: Path, bdf: str) -> dict:
    od = _text(dev / "pp_od_clk_voltage")
    return {
        "bdf": fact(bdf),
        "arch": _architecture(dev),
        "vram_total_bytes": _number(dev / "mem_info_vram_total", positive=True),
        "firmware_version": _firmware_version(dev, sys_root),
        "sclk_current_mhz": fact(_dpm_current(dev / "pp_dpm_sclk")),
        "mclk_current_mhz": fact(_dpm_current(dev / "pp_dpm_mclk")),
        "sclk_od_max_mhz": fact(_od_max(od, "OD_SCLK")),
        "mclk_od_max_mhz": fact(_od_max(od, "OD_MCLK")),
        "power": {
            "cap_uw": _number(
                _hwmon(dev, "power1_cap"), positive=True, zero_reason="zero_cap_suspect"
            ),
            "cap_default_uw": _number(_hwmon(dev, "power1_cap_default"), positive=True),
            "cap_min_uw": _number(_hwmon(dev, "power1_cap_min"), positive=True),
            "cap_max_uw": _number(_hwmon(dev, "power1_cap_max"), positive=True),
            "average_uw": _number(_hwmon(dev, "power1_average"), positive=True),
            "perf_level": fact(_text(dev / "power_dpm_force_performance_level")),
        },
        "pcie": _pcie_tree(dev, sys_root),
    }


def _command(argv: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            argv, text=True, capture_output=True, timeout=3, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _lease_holder(output: str | None) -> dict:
    if output is None:
        return fact(None, "lease_status_unavailable")
    match = re.search(r"(?m)^gpu:\s+held by\s+(.+?)\s+for\s+", output)
    if match:
        return fact(match.group(1).strip())
    if re.search(r"(?m)^gpu:\s+(?:free|unheld|not held)\b", output):
        return {"value": None, "readable": True, "reason": "unheld"}
    return fact(None, "lease_status_unrecognized")


def _memory_facts(output: str | None) -> dict:
    if output is None:
        return {
            "populated_dimms": fact(None),
            "configured_speed_mt_s": fact(None),
            "profile": fact(None, "not_exposed_by_dmidecode"),
        }
    # Each dmidecode "Memory Device" section describes one physical slot.
    sections = re.split(r"(?m)^\s*Memory Device\s*$", output)[1:]
    populated = []
    for section in sections:
        size = re.search(r"(?m)^\s*Size:\s*(.+)$", section)
        if size and re.match(r"\d+\s+(?:GB|MB|TB)", size.group(1)):
            populated.append(section)
    speeds = set()
    for block in populated:
        speed = re.search(r"(?m)^\s*Configured Memory Speed:\s*(\d+)\s*MT/s", block)
        if speed and int(speed.group(1)) > 0:
            speeds.add(int(speed.group(1)))
    return {
        "populated_dimms": fact(len(populated)) if sections else fact(None),
        "configured_speed_mt_s": fact(sorted(speeds)) if speeds else fact(None),
        "profile": fact(None, "not_exposed_by_dmidecode"),
    }


def _available_memory(proc_root: Path) -> dict:
    raw = _text(proc_root / "meminfo") or ""
    match = re.search(r"(?m)^MemAvailable:\s*(\d+)\s+kB$", raw)
    return fact(int(match.group(1))) if match else fact(None)


def _environment_hash(environment_file: Path) -> dict:
    try:
        return fact(hashlib.sha256(environment_file.read_bytes()).hexdigest())
    except OSError:
        return fact(None, "environment_unreadable")


def collect_host_facts(
    *,
    sys_root: Path = _SYS,
    proc_root: Path = _PROC,
    rocm_root: Path | None = None,
    environment_file: Path | None = None,
    command: Callable[[list[str]], str | None] = _command,
) -> dict[str, Any]:
    """Collect independent facts; missing files/tools never abort a measurement.

    The procfs/sysfs roots, ROCm install and external read-only command runner
    are injectable for offline fixtures. Never substitutes a current clock for
    an OD max or a zero power cap for a valid wattage.
    """
    sys_root, proc_root = map(Path, (sys_root, proc_root))
    # The SDK the environment selected (a versioned stack, never the unversioned SDK link).
    if rocm_root is None:
        selected = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH")
        rocm_root = Path(selected) if selected else None
    else:
        rocm_root = Path(rocm_root)
    environment_file = Path(
        environment_file
        if environment_file is not None
        else os.environ.get("BIGCHERRY_ENVIRONMENT", "/etc/bigcherry/environment.toml")
    )
    try:
        lease = command(["agent-lease", "status"])
    except (OSError, subprocess.SubprocessError):
        lease = None
    try:
        dimms = command(["dmidecode", "-t", "memory"])
    except (OSError, subprocess.SubprocessError):
        dimms = None
    return {
        "kernel_release": fact(_text(proc_root / "sys/kernel/osrelease")),
        "kernel_build": fact(_text(proc_root / "version")),
        "rocm_version": fact(_text(rocm_root / ".info/version") if rocm_root else None,
                             "unreadable" if rocm_root else "sdk_not_selected"),
        "amdgpu_version": fact(_text(sys_root / "module/amdgpu/version")),
        "amdgpu_srcversion": fact(_text(sys_root / "module/amdgpu/srcversion")),
        "cards": {
            bdf: _card_facts(dev, sys_root, bdf) for bdf, dev in card_paths(sys_root)
        },
        "mem_available_kib": _available_memory(proc_root),
        "memory": _memory_facts(dimms),
        "environment_sha256": _environment_hash(environment_file),
        "gpu_lease_holder": _lease_holder(lease),
    }
