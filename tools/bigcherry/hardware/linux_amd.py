"""Linux AMD GPU discovery from AMD SMI JSON plus sysfs locators.

AMD SMI is asked for machine-readable output. Parsing is container-tolerant
but field-strict: a GPU is emitted only when architecture, model, VRAM and a
stable identity can be established. PCI BDF/render/HIP ordinal are locators,
never scientific identity.
"""
from __future__ import annotations

import json
import platform
import re
import socket
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from bigcherry.jobs.model import digest

from .inventory import HardwareBindingError
from .model import DeviceRecord, HardwareInventory


class AmdDiscoveryError(HardwareBindingError):
    pass


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _direct(mapping: Mapping[str, Any], names: Iterable[str]) -> Any | None:
    wanted = {_key(name) for name in names}
    for key, value in mapping.items():
        if _key(key) in wanted and value not in (None, "", "N/A", "n/a"):
            return value
    return None


def _find(mapping: Mapping[str, Any], names: Iterable[str]) -> Any | None:
    value = _direct(mapping, names)
    if value is not None:
        return value
    for child in mapping.values():
        if isinstance(child, Mapping):
            value = _find(child, names)
            if value is not None:
                return value
    return None


def _container_rows(value: object) -> tuple[dict[str, Any], ...]:
    if isinstance(value, list):
        return tuple(dict(row) for row in value if isinstance(row, Mapping))
    if not isinstance(value, Mapping):
        return ()

    # Numeric/name keyed AMD-SMI containers also contain BDF recursively.
    # Inspect child mappings first; only a DIRECT BDF/UUID makes this mapping
    # itself one GPU row. Recursive detection here collapses all GPUs into the
    # first row and loses VRAM/model fields for the rest.
    child_rows = tuple(
        dict(row) for row in value.values() if isinstance(row, Mapping)
    )
    if child_rows and any(
        _find(row, ("bdf", "uuid")) is not None for row in child_rows
    ):
        return child_rows
    if _direct(value, ("bdf", "uuid")) is not None:
        return (dict(value),)
    return ()


def _gpu_rows(payload: object) -> tuple[dict[str, Any], ...]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if _key(key) in {"gpu", "gpus", "devices"}:
                rows = _container_rows(value)
                if rows:
                    return rows
        rows = _container_rows(payload)
        if rows:
            return rows
    if isinstance(payload, list):
        return tuple(dict(row) for row in payload if isinstance(row, Mapping))
    return ()


def _normalize_bdf(raw: object | None) -> str | None:
    if raw is None:
        return None
    match = re.search(
        r"(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",
        str(raw).strip().lower(),
    )
    if not match:
        return None
    value = match.group(0)
    return value if value.count(":") == 2 else "0000:" + value


def _render_node(raw: object | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    match = re.search(r"renderD(\d+)", text, re.IGNORECASE)
    if match:
        return f"/dev/dri/renderD{int(match.group(1))}"
    if text.isdigit():
        return f"/dev/dri/renderD{int(text)}"
    return None


def _ordinal(raw: object | None) -> int | None:
    if raw is None or isinstance(raw, bool):
        return None
    match = re.search(r"-?\d+", str(raw))
    if not match:
        return None
    value = int(match.group(0))
    return value if value >= 0 else None


def _bytes(raw: object | None) -> int | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = int(raw)
        return value if value > 0 else None
    text = str(raw).strip().replace(",", "")
    match = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b|bytes?)?", text, re.I
    )
    if not match:
        return None
    number = float(match.group(1))
    unit = (match.group(2) or "bytes").lower()
    factor = {
        "bytes": 1,
        "byte": 1,
        "b": 1,
        "kb": 1000,
        "kib": 1024,
        "mb": 1000**2,
        "mib": 1024**2,
        "gb": 1000**3,
        "gib": 1024**3,
        "tb": 1000**4,
        "tib": 1024**4,
    }.get(unit)
    if factor is None:
        return None
    value = int(number * factor)
    return value if value > 0 else None


def _vram(row: Mapping[str, Any]) -> int | None:
    for key, child in row.items():
        if _key(key) in {"vram", "memory", "vram_info"} and isinstance(
            child, Mapping
        ):
            parsed = _bytes(
                _find(child, ("size", "total", "total_memory", "vram_size"))
            )
            if parsed is not None:
                return parsed
    return _bytes(_find(row, ("vram_size", "vram_total", "total_vram")))


def _numa_node(sysfs_root: Path, bdf: str | None) -> int | None:
    if bdf is None:
        return None
    path = sysfs_root / "bus" / "pci" / "devices" / bdf / "numa_node"
    try:
        value = int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    return value if value >= 0 else None


def _identity(
    row: Mapping[str, Any],
    static: Mapping[str, Any],
    *,
    bdf: str | None,
    hardware_epoch: str | None,
) -> tuple[str, str]:
    uuid = _find(row, ("uuid", "hip_uuid")) or _find(
        static, ("uuid", "hip_uuid")
    )
    if uuid is not None:
        text = str(uuid).strip()
        if text and text.lower() not in {"n/a", "none"}:
            return text, "amd-smi-uuid"
    serial = _find(static, ("asic_serial", "serial_number", "serial"))
    if serial is not None:
        text = str(serial).strip()
        if text and text.lower() not in {"n/a", "none", "0x0"}:
            return text, "amd-smi-serial"
    if hardware_epoch and bdf:
        return f"weak:{hardware_epoch}:{bdf}", "weak-hardware-epoch"
    raise AmdDiscoveryError(
        f"GPU {bdf or '<unknown>'} has no stable UUID/serial; provide an explicit "
        "hardware_epoch to use weak identity"
    )


def inventory_from_amdsmi_json(
    list_payload: object,
    static_payload: object,
    *,
    host_id: str,
    platform_environment_hash: str,
    sysfs_root: Path = Path("/sys"),
    hardware_epoch: str | None = None,
) -> HardwareInventory:
    listed = _gpu_rows(list_payload)
    static_rows = _gpu_rows(static_payload)
    if not listed:
        raise AmdDiscoveryError("AMD SMI list JSON contained no GPU records")

    static_by_bdf = {
        bdf: row
        for row in static_rows
        if (bdf := _normalize_bdf(_find(row, ("bdf",)))) is not None
    }
    devices: list[DeviceRecord] = []
    for row in listed:
        bdf = _normalize_bdf(_find(row, ("bdf",)))
        static = static_by_bdf.get(bdf or "", row)
        architecture = _find(
            static,
            (
                "target_graphics_version",
                "gfx_version",
                "gcn_arch_name",
                "architecture",
            ),
        )
        model = _find(static, ("market_name", "product_name", "board_name"))
        vram = _vram(static)
        if architecture is None or not re.fullmatch(
            r"gfx[0-9a-f]+", str(architecture).strip().lower()
        ):
            raise AmdDiscoveryError(
                f"GPU {bdf or '<unknown>'} has no valid gfx architecture"
            )
        if model is None or not str(model).strip():
            raise AmdDiscoveryError(
                f"GPU {bdf or '<unknown>'} has no market/model name"
            )
        if vram is None:
            raise AmdDiscoveryError(
                f"GPU {bdf or '<unknown>'} has no parseable VRAM size"
            )
        device_id, identity_source = _identity(
            row, static, bdf=bdf, hardware_epoch=hardware_epoch
        )
        devices.append(
            DeviceRecord(
                device_id=device_id,
                identity_source=identity_source,
                architecture=str(architecture).strip().lower(),
                model=str(model).strip(),
                vram_bytes=vram,
                pci_bdf=bdf,
                render_node=_render_node(_find(row, ("render", "render_node"))),
                numa_node=_numa_node(sysfs_root, bdf),
                driver_version=(
                    None
                    if (driver := _find(static, ("driver_version", "driver")))
                    is None
                    else str(driver).strip()
                ),
                launch_ordinal=_ordinal(
                    _find(row, ("hip_id", "hip_index", "id"))
                ),
            )
        )
    if len({device.device_id for device in devices}) != len(devices):
        raise AmdDiscoveryError("AMD SMI stable GPU identities are not unique")
    if any(device.render_node is None for device in devices):
        raise AmdDiscoveryError(
            "AMD SMI enumeration did not provide a render node for every GPU"
        )
    return HardwareInventory(
        host_id=host_id,
        platform_family="linux-rocm",
        devices=tuple(sorted(devices, key=lambda device: device.device_id)),
        peer_access=(),
        platform_environment_hash=platform_environment_hash,
        hardware_epoch=hardware_epoch,
    )


def _run_json(binary: str, *args: str, timeout: float = 20.0) -> object:
    completed = subprocess.run(
        (binary, *args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise AmdDiscoveryError(
            f"{' '.join((binary, *args))} failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AmdDiscoveryError(
            f"{' '.join((binary, *args))} returned invalid JSON: {exc}"
        ) from exc


def _version_text(binary: str) -> str:
    completed = subprocess.run(
        (binary, "version"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10.0,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def discover_linux_amd(
    *,
    amd_smi: str = "amd-smi",
    host_id: str | None = None,
    sysfs_root: Path = Path("/sys"),
    hardware_epoch: str | None = None,
) -> HardwareInventory:
    if platform.system().lower() != "linux":
        raise AmdDiscoveryError("Linux AMD discovery requires Linux")
    listed = _run_json(amd_smi, "list", "-e", "--json")
    static = _run_json(amd_smi, "static", "-a", "-b", "-v", "-d", "--json")
    environment_hash = digest(
        {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "amd_smi_version": _version_text(amd_smi),
        },
        person=b"bc-platform-env",
    )
    return inventory_from_amdsmi_json(
        listed,
        static,
        host_id=host_id or socket.gethostname(),
        platform_environment_hash=environment_hash,
        sysfs_root=sysfs_root,
        hardware_epoch=hardware_epoch,
    )
