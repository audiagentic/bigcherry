"""`bigcherry host-check`: warn when the host is not set up the way the tooling expects (PWR04).

Run on the host itself, before a campaign or a server start. Each finding is a WARN line naming the missing setting and
what it costs; the command exits 0 unless --strict. Checks: CPU package energy readable without root (the PWR01 power
measurements need it), and that this user can apply the cards' configured power settings ([host.*.devices.power]).
"""

from __future__ import annotations

import glob
import os
from argparse import Namespace

from ..core import environment
from ..hardware import gpu_power
from . import power_cmds


def findings(host: environment.Host) -> list[str]:
    out: list[str] = []
    rapl = sorted(glob.glob("/sys/class/powercap/intel-rapl*/energy_uj"))
    if rapl and not all(os.access(p, os.R_OK) for p in rapl):
        out.append("CPU package energy (RAPL) is not readable without root: power measurements (PWR01) report no CPU "
                   "watts; install the read-access udev rule for the powercap energy counters")
    # Power settings are applied per run (gpu-power apply / a launcher's --gpu-power), so a card at its own values is
    # normal; what matters is whether this user could apply them.
    for d in host.devices:
        if not d.locator or not d.power:
            continue
        driver = gpu_power.driver_of(d.locator)
        if driver not in gpu_power.BACKENDS:
            out.append(f"GPU {d.index} ({d.locator}) has power settings configured but driver {driver!r} has no power "
                       "backend: they cannot be applied")
            continue
        files = gpu_power.targets(d.locator, driver, gpu_power.hwmon_of(d.locator), power_cmds._setting(d.power))
        if not all(os.access(p, os.W_OK) for p in files):
            out.append(f"GPU {d.index} ({d.locator}) has power settings configured that this user cannot apply; "
                       "`bigcherry gpu-power apply` lists the setup steps still missing")
    return out


def cmd_host_check(args: Namespace) -> int:
    host = environment.load_default().host(args.host)
    found = findings(host)
    for f in found:
        print(f"WARN {f}")
    if not found:
        print(f"host-check: {host.name} OK")
    return 1 if found and args.strict else 0


def register(sub) -> None:
    p = sub.add_parser("host-check", help="warn when this host lacks settings the tooling expects (power, RAPL access)")
    p.add_argument("--host", help="configured [host.*] entry (default: default-host)")
    p.add_argument("--strict", action="store_true", help="exit 1 on any warning")
    p.set_defaults(func=cmd_host_check)
