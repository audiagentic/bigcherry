"""`bigcherry host-check`: warn when the host is not set up the way the tooling expects (PWR04).

Run on the host itself, before a campaign or a server start. Each finding is a WARN line naming the missing setting and
what it costs; the command exits 0 unless --strict. Checks: CPU package energy readable without root (the PWR01 power
measurements need it), and each card's configured power settings ([host.*.devices.power]) actually applied.
"""

from __future__ import annotations

import glob
import os
from argparse import Namespace

from ..core import environment
from . import power_cmds


def findings(host: environment.Host) -> list[str]:
    out: list[str] = []
    rapl = sorted(glob.glob("/sys/class/powercap/intel-rapl*/energy_uj"))
    if rapl and not all(os.access(p, os.R_OK) for p in rapl):
        out.append("CPU package energy (RAPL) is not readable without root: power measurements (PWR01) report no CPU "
                   "watts; install the read-access udev rule for the powercap energy counters")
    for d in host.devices:
        if not d.locator or not d.power:
            continue
        bad = power_cmds._mismatches(d.power, power_cmds.read_state(d.locator))
        if bad:
            out.append(f"GPU {d.index} ({d.locator}) runs without its configured power settings ({'; '.join(bad)}): "
                       "it draws more power than calibrated; run `bigcherry gpu-power apply` as root")
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
