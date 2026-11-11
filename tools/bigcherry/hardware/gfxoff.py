"""Read-only GFX12 GFXOFF behavioural check (PWR06).

A kernel version string is not a fix detector. Compare the *same* PCI card's
no-process floor with an engine-context-loaded idle window. Both windows
contain 1 Hz sysfs samples; the conservative verdict uses observed ranges,
not board-power or voltage thresholds. Never write a power control.
"""
from __future__ import annotations

import platform
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .power_receipt import write_json_atomic

_BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
_PCI_NON_GFX12 = {0x744c, 0x73af}  # RX 7900 XTX, RX 6900 XT
_PCI_GFX12 = {0x7551}  # R9700; other GFX12 parts use the GC IP discovery major.


@dataclass(frozen=True)
class Sample:
    power_uw: int
    vddgfx_mv: int
    busy_percent: int
    observed_at: float | None = None

    @property
    def watts(self) -> float:
        return self.power_uw / 1_000_000


@dataclass(frozen=True)
class Window:
    bdf: str
    phase: str
    status: str
    reason: str | None
    identity: dict
    sensors: dict
    samples: tuple[Sample, ...]

    def to_dict(self) -> dict:
        return {**asdict(self), "samples": [asdict(s) for s in self.samples]}

    @classmethod
    def from_dict(cls, doc: dict) -> Window:
        if not isinstance(doc, dict) or not isinstance(doc.get("samples"), list):
            raise TypeError("invalid GFXOFF floor record")
        return cls(
            bdf=str(doc["bdf"]), phase=str(doc["phase"]), status=str(doc["status"]),
            reason=doc.get("reason"), identity=dict(doc["identity"]),
            sensors=dict(doc["sensors"]),
            samples=tuple(Sample(**row) for row in doc["samples"]),
        )


@dataclass(frozen=True)
class Verdict:
    bdf: str
    status: str  # active, absent, ambiguous, unavailable, not_applicable, needs_idle_engine_context
    reason: str
    floor: dict | None
    loaded_idle: dict | None

    def to_dict(self) -> dict:
        return asdict(self)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _number(path: Path) -> int:
    value = _read(path)
    if value is None:
        raise ValueError(f"unavailable sensor: {path}")
    return int(value, 0)


def _identity(device: Path, sys_root: Path, *, kernel: str | None = None) -> dict:
    # Firmware versions are optional in sysfs. Do not use debugfs (not mounted
    # in the container), shell tools or a fabricated firmware release.
    fw = {}
    for key in ("mes", "smc"):
        candidates = (
            device / f"{key}_fw_version",
            device / "fw_version" / key,
            device / "firmware_info" / key,
        )
        source = next((p for p in candidates if _read(p) is not None), None)
        fw[key] = {"status": "available", "value": _read(source), "source": str(source)} if source else {
            "status": "unavailable", "value": None, "reason": "not_exposed_in_sysfs"
        }
    return {
        "kernel_release": kernel if kernel is not None else platform.release(),
        "amdgpu_srcversion": _read(sys_root / "module" / "amdgpu" / "srcversion"),
        "firmware": fw,
        "pci_device": _read(device / "device"),
    }


def gfx12_card(bdf: str, *, sys_root: Path = Path("/sys")) -> bool | None:
    """True GFX12, False known other generation, None if undiscoverable."""
    if not _BDF.fullmatch(bdf):
        raise ValueError(f"invalid PCI BDF: {bdf!r}")
    d = Path(sys_root) / "bus" / "pci" / "devices" / bdf
    majors = []
    for path in sorted((d / "ip_discovery" / "die").glob("*/GC/*/major")):
        raw = _read(path)
        if raw is not None:
            try:
                majors.append(int(raw, 0))
            except ValueError:
                return None
    if majors:
        return majors[0] == 12 if len(set(majors)) == 1 else None
    pci = _read(d / "device")
    if pci is None:
        return None
    try:
        return True if int(pci, 0) in _PCI_GFX12 else (False if int(pci, 0) in _PCI_NON_GFX12 else None)
    except ValueError:
        return None


def _sensors(device: Path) -> tuple[dict, str | None]:
    hwmon = []
    for d in sorted((device / "hwmon").glob("hwmon*")):
        if _read(d / "name") == "amdgpu":
            hwmon.append(d)
    if len(hwmon) != 1:
        return {}, "amdgpu_hwmon_missing_or_ambiguous"
    paths = {
        "power_uw": hwmon[0] / "power1_average",
        "vddgfx_mv": hwmon[0] / "in0_input",
        "busy_percent": device / "gpu_busy_percent",
    }
    if any(_read(p) is None for p in paths.values()):
        return {k: str(v) for k, v in paths.items()}, "required_sensor_missing"
    return {k: str(v) for k, v in paths.items()}, None


def sample_cards(
    bdfs: Iterable[str], *, phase: str, sys_root: Path = Path("/sys"),
    count: int = 4, interval: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    kernel: str | None = None,
) -> dict[str, Window]:
    """Sample cards together once per tick, so TP needs just one short window."""
    if phase not in ("no_process", "engine_loaded_idle"):
        raise ValueError("unrecognised GFXOFF phase")
    if count < 2 or interval <= 0:
        raise ValueError("GFXOFF needs multiple samples at a positive interval")
    root = Path(sys_root)
    targets: dict[str, tuple[dict, dict, str | None]] = {}
    for bdf in dict.fromkeys(bdfs):
        arch = gfx12_card(bdf, sys_root=root)
        device = root / "bus" / "pci" / "devices" / bdf
        identity = _identity(device, root, kernel=kernel)
        if arch is False:
            targets[bdf] = (identity, {}, "not_applicable")
        elif arch is None:
            targets[bdf] = (identity, {}, "gfx_generation_unknown")
        else:
            sensors, error = _sensors(device)
            targets[bdf] = (identity, sensors, error)
    rows: dict[str, list[Sample]] = {bdf: [] for bdf in targets}
    errors: dict[str, str] = {}
    for i in range(count):
        for bdf, (_, sensors, error) in targets.items():
            if error is not None or bdf in errors:
                continue
            try:
                power = _number(Path(sensors["power_uw"]))
                voltage = _number(Path(sensors["vddgfx_mv"]))
                busy = _number(Path(sensors["busy_percent"]))
                if power <= 0 or voltage < 0 or not 0 <= busy <= 100:
                    raise ValueError("out_of_range")
                rows[bdf].append(Sample(power, voltage, busy, time.time()))
            except (OSError, ValueError) as exc:
                errors[bdf] = f"sensor_unreadable:{type(exc).__name__}"
        if i < count - 1 and any(e is None for _, _, e in targets.values()):
            sleep(interval)
    result = {}
    for bdf, (identity, sensors, error) in targets.items():
        reason = errors.get(bdf, error)
        status = "not_applicable" if reason == "not_applicable" else ("unavailable" if reason else "measured")
        result[bdf] = Window(bdf, phase, status, reason, identity, sensors, tuple(rows[bdf]))
    return result


def _range(samples: tuple[Sample, ...], key: str) -> tuple[int, int]:
    values = [getattr(s, key) for s in samples]
    return min(values), max(values)


def evaluate(floor: Window | None, loaded: Window | None) -> Verdict:
    src = loaded or floor
    if src is None:
        raise ValueError("a BDF window is required")
    if src.status == "not_applicable":
        return Verdict(src.bdf, "not_applicable", "non_gfx12_card", floor.to_dict() if floor else None, loaded.to_dict() if loaded else None)
    if floor is None or loaded is None:
        return Verdict(src.bdf, "needs_idle_engine_context", "paired_no_process_and_loaded_idle_windows_required",
                       floor.to_dict() if floor else None, loaded.to_dict() if loaded else None)
    a, b = floor, loaded
    raw = (a.to_dict(), b.to_dict())
    if a.status != "measured" or b.status != "measured":
        return Verdict(b.bdf, "unavailable", a.reason or b.reason or "measurement_unavailable", *raw)
    if (a.phase != "no_process" or b.phase != "engine_loaded_idle" or a.bdf != b.bdf
            or a.sensors != b.sensors or a.identity != b.identity):
        return Verdict(b.bdf, "ambiguous", "phase_card_sensor_or_driver_identity_mismatch", *raw)
    if len(a.samples) < 2 or len(b.samples) < 2:
        return Verdict(b.bdf, "unavailable", "insufficient_samples", *raw)
    # A no-process floor with GPU activity cannot establish an idle
    # baseline; comparing to another busy phase would give false assurance.
    if any(s.busy_percent != 0 for s in a.samples):
        return Verdict(b.bdf, "ambiguous", "no_process_floor_not_idle", *raw)
    a_times = [s.observed_at for s in a.samples]
    b_times = [s.observed_at for s in b.samples]
    if all(t is not None for t in a_times + b_times) and max(a_times) >= min(b_times):
        return Verdict(b.bdf, "ambiguous", "floor_not_before_loaded_idle", *raw)
    keys = ("power_uw", "vddgfx_mv", "busy_percent")
    spans = [(_range(a.samples, key), _range(b.samples, key)) for key in keys]
    # Definite absence requires mutually supporting rises in all three
    # independently measured signals. Ordinary sensor noise cannot trigger it.
    if all(before[1] < after[0] for before, after in spans):
        return Verdict(b.bdf, "absent", "loaded_idle_power_voltage_and_busy_above_floor", *raw)
    # Do not declare active merely because a noisy floor is very wide:
    # matching BOTH endpoints of the observed spread on each signal is
    # stronger evidence of unchanged loaded idle. A narrower loaded range
    # within a broad floor is inconclusive, not a proved GFXOFF state.
    if all(before == after for before, after in spans):
        return Verdict(b.bdf, "active", "loaded_idle_matches_measured_floor_spread", *raw)
    return Verdict(b.bdf, "ambiguous", "noise_or_mixed_signals_prevent_classification", *raw)


def format_verdict(verdict: Verdict) -> str:
    bdf = verdict.bdf
    if verdict.status == "active":
        return f"gfxoff: active on {bdf}"
    if verdict.status == "absent":
        a = verdict.floor["samples"]
        b = verdict.loaded_idle["samples"]
        fmt = lambda rows, key: sum(row[key] for row in rows) / len(rows)
        return (
            f"WARNING: gfx12 GFXOFF fix not active on {bdf} "
            f"(idle {fmt(b, 'power_uw') / 1e6:.1f} W, VDDGFX {fmt(b, 'vddgfx_mv'):.0f} mV "
            f"vs floor {fmt(a, 'power_uw') / 1e6:.1f} W, {fmt(a, 'vddgfx_mv'):.0f} mV): "
            "idle power ~5 W higher; see PWR06"
        )
    return f"gfxoff: {verdict.status} on {bdf} ({verdict.reason})"


def store_windows(path: Path, windows: dict[str, Window]) -> None:
    write_json_atomic(path, {"schema_version": 1, "windows": {b: w.to_dict() for b, w in windows.items()}})


def load_windows(path: Path) -> dict[str, Window]:
    import json
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema_version") != 1 or not isinstance(doc.get("windows"), dict):
        raise ValueError("invalid GFXOFF window record")
    return {b: Window.from_dict(row) for b, row in doc["windows"].items()}
