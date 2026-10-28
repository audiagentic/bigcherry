"""`bigcherry gpu-power status|apply|restore|grant|revoke` (PWR04): a host's configured per-card power settings.

The settings come from the environment's device inventory (``[host.*.devices.power]``, environment.py). ``status`` reads
what each card is actually running (OverDrive voltage offset and maximum shader clock, board power cap) and compares it
with the configuration; it needs no privileges. ``apply`` and ``restore`` write sysfs as whoever runs them: root, or a
user whose group ``grant`` gave write access to exactly those files (a udev rule, installed once with sudo). When the
files are not writable they change nothing and print the command that allows it. Each write is read back and a mismatch
is an error.
"""

from __future__ import annotations

import getpass
import glob
import os
import re
import shlex
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

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
    """What an amdgpu card runs now (the only power backend so far, hardware/gpu_power.py): vddgfx_offset_mv, sclk_max_mhz, power_cap_w, power_cap_default_w (None if unreadable)."""
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
        driver = gpu_power.driver_of(d.locator)
        if driver not in gpu_power.BACKENDS:
            print(f"{d.index} {d.arch} {d.locator}: no power backend for driver {driver!r}"
                  + ("; configured settings cannot be applied" if d.power else ""))
            differs += bool(d.power)
            continue
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


#: where grant writes its rules (udev reads /etc/udev/rules.d)
RULES = Path("/etc/udev/rules.d/70-bigcherry-gpu-power.rules")
#: the group grant gives the write access to by default: its own, so only users added to it can change voltage and power
#: (video holds every GPU user on most hosts); grant creates it when it is missing
GROUP = "bigcherry-gpu"


def _gid(group: str) -> int | None:
    try:
        import grp  # noqa: PLC0415 -- POSIX only
        return grp.getgrnam(group).gr_gid
    except (ImportError, KeyError):
        return None


def _configured(args: Namespace) -> list[tuple[environment.Device, str | None, str | None]]:
    """The cards with power settings: (device, driver, hwmon dir)."""
    return [(d, gpu_power.driver_of(d.locator), _hwmon(d.locator)) for d in _devices(args) if d.power]


def _blocked(cards) -> list[str]:
    """The files the settings write that this process cannot write."""
    return [path for d, driver, hw in cards for path in gpu_power.targets(d, d.power, driver, hw)
            if not os.access(path, os.W_OK)]


def _read_only_mount(path: str) -> bool:
    """Whether ``path`` (or, when it is missing, its directory) lies on a read-only mount."""
    probe = path if os.path.exists(path) else os.path.dirname(path)
    try:
        return bool(os.statvfs(probe).f_flag & os.ST_RDONLY)
    except (OSError, AttributeError):   # no statvfs on Windows
        return False


def _bigcherry(args: Namespace, *words: str, here: bool) -> str:
    """A bigcherry command line: with this checkout's paths when it runs ``here``, else as run from a checkout's root."""
    host = ["--host", args.host] if getattr(args, "host", None) else []
    if not here:
        return " ".join(["PYTHONPATH=tools python3 -m bigcherry", *words, *host])
    env = os.environ.get("BIGCHERRY_ENVIRONMENT")
    cmd = [*(["BIGCHERRY_ENVIRONMENT=" + env] if env else []), f"PYTHONPATH={Path(__file__).resolve().parents[2]}",
           sys.executable or "python3", "-m", "bigcherry", *words, *host]
    return " ".join(shlex.quote(c) for c in cmd)


def grant_command(args: Namespace, user: str) -> str:
    """The exact command, run here, that grants ``user`` the write access."""
    return "sudo " + _bigcherry(args, "gpu-power", "grant", "--user", user, here=True)


def _locators(paths: list[str]) -> list[str]:
    return sorted({m.group(1) for p in paths if (m := re.search(r"/devices/([0-9a-fA-F:.]+)/", p))})


def explain_blocked(blocked: list[str], args: Namespace) -> str:
    """What to tell someone whose launch cannot write the power settings: numbered steps, exact commands."""
    user = os.environ.get("SUDO_USER") or getpass.getuser()
    cards = ", ".join(_locators(blocked))
    if any(p.endswith("pp_od_clk_voltage") and not os.path.exists(p) for p in blocked):
        return "\n".join([
            f"gpu-power: the GPU driver's OverDrive controls are off on {cards}, so these settings cannot be applied.",
            "To turn them on (once, on the host):",
            "  1. Add amdgpu.ppfeaturemask=0xfff7ffff to the kernel command line (e.g. GRUB_CMDLINE_LINUX in "
            "/etc/default/grub, then update-grub).",
            "  2. Reboot, then run the launch again.",
        ])
    if any(_read_only_mount(p) for p in blocked):
        # A container's /sys is mounted read-only (Docker/Podman default): no permission change reaches it, and mounts
        # and groups are fixed when the container is created.
        gid = _gid(GROUP)
        dirs = [os.path.realpath(f"/sys/bus/pci/devices/{loc}") for loc in _locators(blocked)]
        return "\n".join([
            f"gpu-power: cannot change the GPU power settings of {cards}: /sys is read-only in this container.",
            "Either A, apply them from the host for each run (no container change), from a bigcherry checkout there:",
            f"  {_bigcherry(args, 'gpu-power', 'apply', here=False)}      (and after the run: ... gpu-power restore)",
            "Or B, let this container apply them itself (once):",
            "  1. On the host, from a bigcherry checkout:",
            f"       sudo {_bigcherry(args, 'gpu-power', 'grant', here=False)}",
            "  2. In the container's compose service add the lines below, then recreate it "
            "(docker compose up -d <service>; a plain restart keeps the old mounts):",
            "       group_add:",
            f"         - \"{gid if gid is not None else '<gid of ' + GROUP + ' on the host: getent group ' + GROUP + '>'}\"",
            "       volumes:",
            *[f"         - {d}:{d}:rw" for d in dirs],
            "  3. Run the launch again.",
        ])
    return "\n".join([
        f"gpu-power: {user} cannot change the GPU power settings of {cards}.",
        f"To allow it (once; adds {user} to the {GROUP} group, which may write only these cards' power files):",
        f"  1. {grant_command(args, user)}",
        f"  2. Log out and back in (or run: newgrp {GROUP}).",
        "  3. Run the launch again.",
        f"To see the rule first add --print to step 1; to undo: sudo {_bigcherry(args, 'gpu-power', 'revoke', here=True)}",
    ])


def cmd_apply(args: Namespace) -> int:
    cards = _configured(args)
    blocked = _blocked(cards)
    if blocked:
        print(explain_blocked(blocked, args), file=sys.stderr)
        return 2
    failed = 0
    for d, driver, hw in cards:
        for path, value in gpu_power.apply_writes(d, d.power, driver, hw):
            _write(path, value)
        bad = _mismatches(d.power, read_state(d.locator))
        print(f"{d.index} {d.locator}: " + ("applied" if not bad else "NOT APPLIED: " + "; ".join(bad)))
        failed += bool(bad)
    return 1 if failed else 0


def cmd_restore(args: Namespace) -> int:
    cards = _configured(args)
    blocked = _blocked(cards)
    if blocked:
        print(explain_blocked(blocked, args), file=sys.stderr)
        return 2
    for d, driver, hw in cards:
        default = _read(f"{hw}/power1_cap_default") if hw else None
        for path, value in gpu_power.restore_writes(d, driver, hw,
                                                    int(default) if default and d.power.power_cap_w else None):
            _write(path, value)
        print(f"{d.index} {d.locator}: restored to driver defaults")
    return 0


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def _trigger(cards) -> None:
    _run(["udevadm", "control", "--reload"])
    for d, _, _ in cards:
        _run(["udevadm", "trigger", "--action=change", f"/sys/bus/pci/devices/{d.locator}"])
        for hw in sorted(glob.glob(f"/sys/bus/pci/devices/{d.locator}/hwmon/hwmon*")):
            _run(["udevadm", "trigger", "--action=change", hw])
    _run(["udevadm", "settle"])


def cmd_grant(args: Namespace) -> int:
    cards = _configured(args)
    if not cards:
        print("gpu-power grant: no card on this host has power settings configured; nothing to grant", file=sys.stderr)
        return 1
    rules = gpu_power.udev_rules([(d, driver) for d, driver, _ in cards], args.group)
    if args.print:
        print(rules, end="")
        return 0
    if os.geteuid() != 0:
        print("gpu-power grant: run it with sudo (it writes a udev rule" + (" and adds the user to the group" if args.user
              else "") + "); --print shows the rule without installing it", file=sys.stderr)
        return 2
    if _gid(args.group) is None:
        if args.group != GROUP:
            print(f"gpu-power grant: group {args.group!r} does not exist (only the default {GROUP} is created)",
                  file=sys.stderr)
            return 1
        _run(["groupadd", "--system", GROUP])
    RULES.write_text(rules, encoding="utf-8")
    print(f"wrote {RULES} ({len(cards)} card(s), group {args.group})")
    if args.user:
        _run(["usermod", "-aG", args.group, args.user])
    _trigger(cards)
    print(f"done; {args.user or 'members of ' + args.group} can apply the settings after logging in again "
          f"(or `newgrp {args.group}`)")
    return 0


def cmd_revoke(args: Namespace) -> int:
    if os.geteuid() != 0:
        print("gpu-power revoke: run it with sudo", file=sys.stderr)
        return 2
    if RULES.exists():
        RULES.unlink()
        print(f"removed {RULES}; the files return to root-only at the next driver bind or reboot")
    else:
        print(f"{RULES} is not installed")
    return 0


def register(sub) -> None:
    p = sub.add_parser("gpu-power", help="per-card power settings from the environment config (status | apply | restore)")
    psub = p.add_subparsers(dest="power_action", required=True)
    st = psub.add_parser("status", help="what each card runs now against its configured power settings (no privileges)")
    st.add_argument("--host", help="configured [host.*] entry (default: default-host)")
    st.add_argument("--strict", action="store_true", help="exit 1 when a configured setting is not applied")
    st.set_defaults(func=cmd_status)
    ap = psub.add_parser("apply", help="apply the configured settings and read them back (root, or after grant)")
    ap.add_argument("--host")
    ap.set_defaults(func=cmd_apply)
    rs = psub.add_parser("restore", help="return configured cards to the driver defaults (root, or after grant)")
    rs.add_argument("--host")
    rs.set_defaults(func=cmd_restore)
    gr = psub.add_parser("grant", help="let a group write the configured cards' power files, via a udev rule (sudo)")
    gr.add_argument("--host")
    gr.add_argument("--group", default=GROUP, help=f"group given write access (default {GROUP}, created when missing)")
    gr.add_argument("--user", help="also add this user to the group")
    gr.add_argument("--print", action="store_true", help="print the udev rule instead of installing it (no privileges)")
    gr.set_defaults(func=cmd_grant)
    rv = psub.add_parser("revoke", help="remove the rule grant installed (sudo)")
    rv.set_defaults(func=cmd_revoke)
