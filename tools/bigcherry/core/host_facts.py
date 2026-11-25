"""MRC02 platform host-facts: passive, engine-neutral sysfs/procfs snapshots.

File readers are pure over injectable roots. Optional external commands are
read-only and injected in tests; unavailable/zero/invalid values carry explicit
readability rather than a fabricated zero. This module never changes clocks,
power, PCIe or GPU leases.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
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


_BUILDER = re.compile(r"\s*\([^()\s@]+@[^()\s]+\)")


def _kernel_build(text: str | None) -> str | None:
    """The kernel's version line without the account and machine that built it: the record keeps the release,
    compiler and build date, not who compiled it."""
    return None if text is None else _BUILDER.sub("", text, count=1)


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
        "kernel_build": fact(_kernel_build(_text(proc_root / "version"))),
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

# MRC02: the timestamp is captured before the measured child/request begins,
# NOT when the immutable run record is eventually published.
START_RECEIPT_FILE = "host-facts.start.v1.json"
START_FIELDS = (
    "kernel_release", "kernel_build", "rocm_version", "amdgpu_version",
    "amdgpu_srcversion", "mem_available_kib", "memory",
    "environment_sha256", "gpu_lease_holder",
)
START_PHASES = frozenset({"measurement_start", "owner_start"})


_CARD_FIELDS = (
    "bdf", "arch", "vram_total_bytes", "firmware_version",
    "sclk_current_mhz", "mclk_current_mhz",
    "sclk_od_max_mhz", "mclk_od_max_mhz",
)
_POWER_FIELDS = (
    "cap_uw", "cap_default_uw", "cap_min_uw", "cap_max_uw",
    "average_uw", "perf_level",
)
_CARD_PCIE_FIELDS = ("current_speed", "current_width", "max_speed", "max_width")


def _complete_fields(value: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    return {
        **{name: value.get(name, fact(None, "not_captured_at_start")) for name in required},
        **value,
    }


def _complete_card(bdf: str, raw: Any) -> dict[str, Any]:
    if not isinstance(bdf, str) or _BDF.fullmatch(bdf) is None or not isinstance(raw, dict):
        raise ValueError("start host card requires a PCI BDF and fact object")
    card = _complete_fields(raw, _CARD_FIELDS)
    if "bdf" not in raw:
        card["bdf"] = fact(bdf)
    power = raw.get("power", {})
    pcie = raw.get("pcie", {})
    if not isinstance(power, dict) or not isinstance(pcie, dict):
        raise ValueError("start host card power/PCIe must be objects")  # noqa: TRY004 - schema violations use ValueError
    port = pcie.get("card", {})
    upstream = pcie.get("upstream", [])
    if not isinstance(port, dict) or not isinstance(upstream, list):
        raise ValueError("start host PCIe card/upstream must be an object/list")  # noqa: TRY004 - schema violations use ValueError
    card["power"] = _complete_fields(power, _POWER_FIELDS)
    card["pcie"] = {
        **pcie,
        "card": _complete_fields(port, _CARD_PCIE_FIELDS),
        "upstream": upstream,
        "upstream_readability": pcie.get(
            "upstream_readability", fact(len(upstream)) if upstream
            else fact(None, "upstream_unobserved"),
        ),
    }
    return card


def _validate_cards(cards: dict[str, Any]) -> None:
    for bdf, card in cards.items():
        if not isinstance(bdf, str) or _BDF.fullmatch(bdf) is None or not isinstance(card, dict):
            raise ValueError("start host card requires a PCI BDF and fact object")
        identity = card.get("bdf")
        if not set(_CARD_FIELDS).issubset(card) or (
            not isinstance(identity, dict)
            or identity.get("readable") is not True
            or identity.get("value") != bdf
        ):
            raise ValueError("start host card identity or field readability is incomplete")
        if not isinstance(card.get("power"), dict) or not set(_POWER_FIELDS).issubset(card["power"]):
            raise ValueError("start host card power field readability is incomplete")
        pcie = card.get("pcie")
        if not isinstance(pcie, dict) or not isinstance(pcie.get("card"), dict) or (
            not set(_CARD_PCIE_FIELDS).issubset(pcie["card"])
            or not isinstance(pcie.get("upstream"), list)
            or "upstream_readability" not in pcie
        ):
            raise ValueError("start host card PCIe field readability is incomplete")


def _typed(value: Any, where: str) -> None:
    if isinstance(value, dict):
        if "value" in value or "readable" in value:
            if not isinstance(value.get("readable"), bool) or "value" not in value:
                raise ValueError(f"{where}: missing per-field readability")
            if value["readable"] is False and (
                value["value"] is not None
                or not isinstance(value.get("reason"), str)
                or not value["reason"]
            ):
                raise ValueError(f"{where}: unreadable requires null value and explicit reason")
            if value["readable"] and value["value"] is None and value.get("reason") != "unheld":
                # A legitimate idle GPU lease has no holder, but is readable.
                raise ValueError(f"{where}: readable null must identify an observed absence")
            return
        if not value and where != "host_facts.cards":
            raise ValueError(f"{where}: missing per-field readability")
        for key, item in value.items():
            _typed(item, f"{where}.{key}")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _typed(item, f"{where}[{i}]")
        return
    raise ValueError(f"{where}: host-facts leaf is missing per-field readability")


def snapshot_at_start(
    facts: dict[str, Any] | None = None, *,
    captured_at: str | None = None,
    phase: str = "measurement_start",
) -> dict[str, Any]:
    """Capture passive host readings only when the caller starts measuring.

    Missing fields are explicit unreadable markers, never end-of-run values.
    For native owners, the C++ start receipt supplies the source fields.
    """
    measured = collect_host_facts() if facts is None else facts
    if not isinstance(measured, dict):
        raise ValueError("start host facts must be an object")  # noqa: TRY004 - schema violations use ValueError
    normalised = {
        key: measured.get(key, fact(None, "not_captured_at_start"))
        for key in START_FIELDS
    }
    cards = measured.get("cards")
    if cards is None:
        normalised["cards"] = {}
        normalised["cards_readability"] = fact(None, "not_captured_at_start")
    elif not isinstance(cards, dict):
        raise ValueError("start host cards must be an object keyed by BDF")
    else:
        normalised["cards"] = {bdf: _complete_card(bdf, card) for bdf, card in cards.items()}
        normalised["cards_readability"] = (
            fact(len(cards)) if cards else fact(None, "no_cards_detected")
        )
    for key, value in measured.items():
        if key not in normalised:
            normalised[key] = value
    capture = {
        "schema_version": 1,
        "capture_phase": phase,
        "captured_at": captured_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "facts": normalised,
    }
    return validate_start_snapshot(capture)


def validate_start_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Validate the sole measurement-start receipt shape and field readability."""
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "schema_version", "capture_phase", "captured_at", "facts"
    } or snapshot["schema_version"] != 1 or snapshot["capture_phase"] not in START_PHASES:
        raise ValueError("start host-facts receipt requires a schema and admitted capture phase")
    try:
        value = snapshot["captured_at"]
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("not UTC")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("start host-facts receipt requires a UTC timestamp") from exc
    if not isinstance(snapshot["facts"], dict) or not all(
        key in snapshot["facts"] for key in (*START_FIELDS, "cards", "cards_readability")
    ):
        raise ValueError("start host-facts receipt is missing field readability")
    if not isinstance(snapshot["facts"]["cards"], dict):
        raise ValueError("start host-facts cards must be an object")  # noqa: TRY004 - schema violations use ValueError
    _validate_cards(snapshot["facts"]["cards"])
    _typed(snapshot["facts"], "host_facts")
    return snapshot


def read_start_receipt(path: Path) -> dict[str, Any]:
    """Read a sealed start receipt, never recomputing current/end host values."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or unsafe native host-facts start receipt: {path}")
    return validate_start_snapshot(json.loads(path.read_text(encoding="utf-8")))

def read_owner_start(run_dir: Path, *, session_name: str = "session.json") -> dict[str, Any]:
    """Read and verify the native owner start receipt and its immutable seal.

    A post-run sampler cannot create this proof. The owner session must name the
    digest of the exact start bytes; the caller must keep both files together.
    """
    folder = Path(run_dir)
    session_path = folder / session_name
    receipt_path = folder / START_RECEIPT_FILE
    if session_path.is_symlink() or not session_path.is_file():
        raise ValueError("missing or unsafe native owner session receipt")
    session = json.loads(session_path.read_text(encoding="utf-8"))
    snapshot = read_start_receipt(receipt_path)
    if snapshot["capture_phase"] != "owner_start":
        raise ValueError("native owner requires owner_start host facts")
    if not isinstance(session, dict) or session.get("host_facts_start_sha256") != hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest():
        raise ValueError("native owner start receipt digest differs from session")
    if not isinstance(session.get("started_at"), str):
        raise ValueError("native owner start time is missing")
    started = datetime.fromisoformat(session["started_at"])
    captured = datetime.fromisoformat(snapshot["captured_at"])
    if started.tzinfo is None or started.utcoffset().total_seconds() != 0 or captured < started:
        raise ValueError("native owner receipt precedes its start or timestamp is not UTC")
    return snapshot
