"""`bigcherry gpu-power status|apply|restore` (PWR04): a host's configured per-card power settings, on the host itself.

The settings come from the environment's device inventory (``[host.*.devices.power]``, environment.py). ``status`` reads
what each card is actually running (OverDrive voltage offset and maximum shader clock, board power cap) and compares it
with the configuration; it needs no privileges. ``apply`` and ``restore`` write sysfs and must run as root (a later
slice moves them behind a privileged helper); each write is read back and a mismatch is an error.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from argparse import Namespace

from ..core import environment
from ..hardware import gpu_power


def _hwmon(locator: str) -> str | None:
    found = sorted(glob.glob(f"/sys/bus/pci/devices/{locator}/hwmon/hwmon*"))
    return found[0] if found else None


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def read_state(locator: str) -> dict[str, int | None]:
    """What the card runs now: vddgfx_offset_mv, sclk_max_mhz, power_cap_w, power_cap_default_w (None if unreadable)."""
    od = _read(f"/sys/bus/pci/devices/{locator}/pp_od_clk_voltage") or ""
    sclk = re.search(r"OD_SCLK:\s*\n\s*0:\s*\d+Mhz\s*\n\s*1:\s*(\d+)Mhz", od)
    off = re.search(r"OD_VDDGFX_OFFSET:\s*\n\s*(-?\d+)mV", od)
    hw = _hwmon(locator)
    cap = _read(f"{hw}/power1_cap") if hw else None
    cap_default = _read(f"{hw}/power1_cap_default") if hw else None
    return {
        "vddgfx_offset_mv": int(off.group(1)) if off else None,
        "sclk_max_mhz": int(sclk.group(1)) if sclk else None,
        "power_cap_w": int(cap) // 1_000_000 if cap else None,
        "power_cap_default_w": int(cap_default) // 1_000_000 if cap_default else None,
    }


def _mismatches(want: environment.DevicePower, have: dict[str, int | None]) -> list[str]:
    out = []
    for key in ("vddgfx_offset_mv", "sclk_max_mhz", "power_cap_w"):
        w = getattr(want, key)
        if w is not None and have.get(key) != w:
            out.append(f"{key} configured {w}, running {have.get(key)}")
    return out


def _devices(args: Namespace) -> list[environment.Device]:
    host = environment.load_default().host(args.host)
    return [d for d in host.devices if d.locator]


def cmd_status(args: Namespace) -> int:
    differs = 0
    for d in _devices(args):
        have = read_state(d.locator)
        line = (f"{d.index} {d.arch} {d.locator}: offset {have['vddgfx_offset_mv']} mV, sclk max "
                f"{have['sclk_max_mhz']} MHz, cap {have['power_cap_w']} W (default {have['power_cap_default_w']} W)")
        if d.power:
            bad = _mismatches(d.power, have)
            line += "; configured: " + ("applied" if not bad else "; ".join(bad))
            differs += bool(bad)
        else:
            line += "; configured: driver defaults"
        print(line)
    return 1 if differs and args.strict else 0


def _write(path: str, value: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(value + "\n")


def _require_root() -> bool:
    if os.geteuid() != 0:
        print("bigcherry gpu-power: apply/restore write sysfs and must run as root", file=sys.stderr)
        return False
    return True


def cmd_apply(args: Namespace) -> int:
    if not _require_root():
        return 2
    failed = 0
    for d in _devices(args):
        if not d.power:
            continue
        for path, value in gpu_power.apply_writes(d, d.power, _hwmon(d.locator)):
            _write(path, value)
        bad = _mismatches(d.power, read_state(d.locator))
        print(f"{d.index} {d.locator}: " + ("applied" if not bad else "NOT APPLIED: " + "; ".join(bad)))
        failed += bool(bad)
    return 1 if failed else 0


def cmd_restore(args: Namespace) -> int:
    if not _require_root():
        return 2
    for d in _devices(args):
        if not d.power:
            continue
        hw = _hwmon(d.locator)
        default = _read(f"{hw}/power1_cap_default") if hw else None
        for path, value in gpu_power.restore_writes(d, hw, int(default) if default and d.power.power_cap_w else None):
            _write(path, value)
        print(f"{d.index} {d.locator}: restored to driver defaults")
    return 0


def register(sub) -> None:
    p = sub.add_parser("gpu-power", help="per-card power settings from the environment config (status | apply | restore)")
    psub = p.add_subparsers(dest="power_action", required=True)
    st = psub.add_parser("status", help="what each card runs now against its configured power settings (no privileges)")
    st.add_argument("--host", help="configured [host.*] entry (default: default-host)")
    st.add_argument("--strict", action="store_true", help="exit 1 when a configured setting is not applied")
    st.set_defaults(func=cmd_status)
    ap = psub.add_parser("apply", help="apply the configured settings and read them back (root)")
    ap.add_argument("--host")
    ap.set_defaults(func=cmd_apply)
    rs = psub.add_parser("restore", help="return configured cards to the driver defaults (root)")
    rs.add_argument("--host")
    rs.set_defaults(func=cmd_restore)
