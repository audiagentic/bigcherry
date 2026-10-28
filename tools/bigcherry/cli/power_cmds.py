"""`bigcherry gpu-power status|apply|restore|grant|revoke` (PWR04): a host's configured per-card power settings.

The settings come from the environment's device inventory (``[host.*.devices.power]``, environment.py); the sysfs work
is hardware/gpu_power.py and the setup checks hardware/power_setup.py, neither of which knows about bigcherry config.

``status`` reads what each card runs and compares it with the configuration (no privileges). ``apply`` records what
each configured card runs, then applies its setting; ``restore`` puts the recorded values back and forgets them, so a
run leaves the cards as it found them. Both run as whoever launches: root, or a member of the group ``grant`` gave write
access to the cards' power files (a udev rule, installed once with sudo). When they cannot write, they change nothing
and print the setup steps still missing on this host, worked out from what is actually there.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from ..core import environment
from ..hardware import gpu_power, power_setup

#: where grant writes its rules (udev reads /etc/udev/rules.d)
RULES = Path(power_setup.RULES)
#: the group grant gives the write access to by default: its own, so only users added to it can change voltage and power
#: (video holds every GPU user on most hosts); grant creates it when it is missing
GROUP = "bigcherry-gpu"


def _state_file() -> Path:
    """Where apply records what the cards ran before it (per user; survives the launcher's process)."""
    override = os.environ.get("BIGCHERRY_GPU_POWER_STATE")
    if override:
        return Path(override)
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return Path(base) / "bigcherry" / "gpu-power.json"


def _load_saved() -> dict:
    try:
        return json.loads(_state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(saved: dict) -> None:
    path = _state_file()
    if not saved:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _setting(power: environment.DevicePower) -> gpu_power.Setting:
    return gpu_power.Setting(power.vddgfx_offset_mv, power.sclk_max_mhz, power.power_cap_w)


def _devices(args: Namespace) -> list[environment.Device]:
    host = environment.load_default().host(args.host)
    return [d for d in host.devices if d.locator]


def _configured(args: Namespace) -> list[tuple[environment.Device, str | None, str | None]]:
    """The cards with power settings: (device, driver, hwmon dir)."""
    return [(d, gpu_power.driver_of(d.locator), gpu_power.hwmon_of(d.locator)) for d in _devices(args) if d.power]


def _mismatches(want: gpu_power.Setting, have: gpu_power.State) -> list[str]:
    out = []
    if want.vddgfx_offset_mv is not None and have.vddgfx_offset_mv != want.vddgfx_offset_mv:
        out.append(f"offset configured {want.vddgfx_offset_mv} mV, running {have.vddgfx_offset_mv}")
    if want.sclk_max_mhz is not None and have.sclk_max_mhz != want.sclk_max_mhz:
        out.append(f"sclk max configured {want.sclk_max_mhz} MHz, running {have.sclk_max_mhz}")
    if want.power_cap_w is not None and have.power_cap_uw != want.power_cap_w * 1_000_000:
        out.append(f"cap configured {want.power_cap_w} W, running {_w(have.power_cap_uw)}")
    return out


def _w(uw: int | None) -> str:
    return "?" if uw is None else f"{uw // 1_000_000} W"


def cmd_status(args: Namespace) -> int:
    differs = 0
    for d in _devices(args):
        driver = gpu_power.driver_of(d.locator)
        if driver not in gpu_power.BACKENDS:
            print(f"{d.index} {d.arch} {d.locator}: no power backend for driver {driver!r}"
                  + ("; configured settings cannot be applied" if d.power else ""))
            differs += bool(d.power)
            continue
        have = gpu_power.read_state(d.locator, driver, gpu_power.hwmon_of(d.locator))
        line = (f"{d.index} {d.arch} {d.locator}: offset {have.vddgfx_offset_mv} mV, sclk max {have.sclk_max_mhz} MHz, "
                f"cap {_w(have.power_cap_uw)} (default {_w(have.power_cap_default_uw)})")
        if d.power:
            bad = _mismatches(_setting(d.power), have)
            line += "; configured: " + ("applied" if not bad else "; ".join(bad))
            differs += bool(bad)
        else:
            line += "; configured: driver defaults"
        print(line)
    return 1 if differs and args.strict else 0


def _write(path: str, value: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(value + "\n")


def _command(args: Namespace, *words: str) -> str:
    """A bigcherry command line as this session runs it: this checkout's tools and environment file."""
    host = ["--host", args.host] if getattr(args, "host", None) else []
    env = os.environ.get("BIGCHERRY_ENVIRONMENT")
    cmd = [*(["BIGCHERRY_ENVIRONMENT=" + env] if env else []), f"PYTHONPATH={Path(__file__).resolve().parents[2]}",
           sys.executable or "python3", "-m", "bigcherry", *words, *host]
    return " ".join(shlex.quote(c) for c in cmd)


def _blocked(cards) -> list[str]:
    return [path for d, driver, hw in cards for path in gpu_power.targets(d.locator, driver, hw, _setting(d.power))
            if not os.access(path, os.W_OK)]


def _explain(args: Namespace, blocked: list[str]) -> str:
    p = power_setup.probe(args_group(args), str(RULES), blocked)
    return power_setup.render(p, lambda *w: _command(args, *w))


def args_group(args: Namespace) -> str:
    return getattr(args, "group", None) or GROUP


def cmd_apply(args: Namespace) -> int:
    cards = _configured(args)
    blocked = _blocked(cards)
    if blocked:
        print(_explain(args, blocked), file=sys.stderr)
        return 2
    saved = _load_saved()
    failed = 0
    for d, driver, hw in cards:
        setting = _setting(d.power)
        # The first apply records what ran before; a repeat (no restore in between) keeps that original record.
        saved.setdefault(d.locator, {"before": gpu_power.read_state(d.locator, driver, hw).to_json(),
                                     "setting": setting.__dict__})
        _save(saved)
        for path, value in gpu_power.apply_writes(d.locator, driver, hw, setting):
            _write(path, value)
        bad = _mismatches(setting, gpu_power.read_state(d.locator, driver, hw))
        print(f"{d.index} {d.locator}: " + ("applied" if not bad else "NOT APPLIED: " + "; ".join(bad)))
        failed += bool(bad)
    return 1 if failed else 0


def cmd_restore(args: Namespace) -> int:
    saved = _load_saved()
    if not saved:
        print("gpu-power restore: nothing applied by gpu-power to put back")
        return 0
    cards = [(loc, gpu_power.driver_of(loc), gpu_power.hwmon_of(loc)) for loc in saved]
    blocked = [p for loc, driver, hw in cards
               for p in gpu_power.targets(loc, driver, hw, gpu_power.Setting(**saved[loc]["setting"]))
               if not os.access(p, os.W_OK)]
    if blocked:
        print(_explain(args, blocked), file=sys.stderr)
        return 2
    failed = 0
    for loc, driver, hw in cards:
        before = gpu_power.State.from_json(saved[loc]["before"])
        for path, value in gpu_power.restore_writes(loc, driver, hw, before, gpu_power.Setting(**saved[loc]["setting"])):
            _write(path, value)
        now = gpu_power.read_state(loc, driver, hw)
        same = (now.vddgfx_offset_mv, now.sclk_max_mhz) == (before.vddgfx_offset_mv, before.sclk_max_mhz)
        print(f"{loc}: " + ("restored to what ran before" if same else
                            f"NOT RESTORED: running {now}, recorded {before}"))
        if same:
            del saved[loc]
            _save(saved)
        failed += not same
    return 1 if failed else 0


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def _trigger(cards: list[tuple[str, str | None]]) -> None:
    _run(["udevadm", "control", "--reload"])
    for locator, _ in cards:
        _run(["udevadm", "trigger", "--action=change", f"{gpu_power.SYSFS_PCI}/{locator}"])
        hw = gpu_power.hwmon_of(locator)
        if hw:
            _run(["udevadm", "trigger", "--action=change", hw])
    _run(["udevadm", "settle"])


def _gid(group: str) -> int | None:
    try:
        import grp  # noqa: PLC0415 -- POSIX only
        return grp.getgrnam(group).gr_gid
    except (ImportError, KeyError):
        return None


def cmd_grant(args: Namespace) -> int:
    if not sys.platform.startswith("linux"):
        print("gpu-power grant: GPU power settings use Linux sysfs; nothing to grant on this system", file=sys.stderr)
        return 1
    # Every card with a power backend, configured or not: configuring a card later needs no second grant.
    found = gpu_power.display_cards()
    cards = [(loc, drv) for loc, drv in found if drv in gpu_power.BACKENDS]
    skipped = [(loc, drv) for loc, drv in found if drv not in gpu_power.BACKENDS]
    if not cards:
        print(f"gpu-power grant: no GPU here has a supported power driver (found: "
              f"{', '.join(f'{loc} {drv}' for loc, drv in found) or 'none'})", file=sys.stderr)
        return 1
    rules = gpu_power.udev_rules(cards, args.group)
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
    print(f"wrote {RULES}: group {args.group} may write the power files of {', '.join(loc for loc, _ in cards)}")
    for loc, drv in skipped:
        print(f"  not covered: {loc} ({drv}): no power backend for that driver")
    if args.user:
        _run(["usermod", "-aG", args.group, args.user])
    _trigger(cards)
    print(f"done; {args.user or 'members of ' + args.group} can apply the settings after logging in again "
          f"(or `newgrp {args.group}`)")
    print()
    _print_container_lines(args.group, cards)
    return 0


def _print_container_lines(group: str, cards: list[tuple[str, str | None]]) -> int:
    gid = _gid(group)
    if gid is None:
        print(f"gpu-power: group {group} does not exist on this host yet; run `sudo ... gpu-power grant` first",
              file=sys.stderr)
        return 1
    print(power_setup.container_lines(gid, [os.path.realpath(f"{gpu_power.SYSFS_PCI}/{loc}") for loc, _ in cards]))
    return 0


def cmd_container_lines(args: Namespace) -> int:
    """On the host: the lines any container adds to apply the settings (no privileges; group and rule from grant)."""
    return _print_container_lines(args.group, [(loc, drv) for loc, drv in gpu_power.display_cards()
                                               if drv in gpu_power.BACKENDS])


def cmd_revoke(args: Namespace) -> int:
    if os.geteuid() != 0:
        print("gpu-power revoke: run it with sudo", file=sys.stderr)
        return 2
    if RULES.exists():
        RULES.unlink()
        print(f"removed {RULES}; the files return to root-only at the next driver bind or reboot "
              f"(the {GROUP} group and its members are left as they are)")
    else:
        print(f"{RULES} is not installed")
    return 0


def register(sub) -> None:
    p = sub.add_parser("gpu-power", help="per-card power settings from the environment config")
    psub = p.add_subparsers(dest="power_action", required=True)
    st = psub.add_parser("status", help="what each card runs now against its configured power settings (no privileges)")
    st.add_argument("--host", help="configured [host.*] entry (default: default-host)")
    st.add_argument("--strict", action="store_true", help="exit 1 when a configured setting is not applied")
    st.set_defaults(func=cmd_status)
    ap = psub.add_parser("apply", help="record what the cards run, then apply the configured settings (root, or after grant)")
    ap.add_argument("--host")
    ap.set_defaults(func=cmd_apply)
    rs = psub.add_parser("restore", help="put back what ran before apply (root, or after grant)")
    rs.add_argument("--host")
    rs.set_defaults(func=cmd_restore)
    gr = psub.add_parser("grant", help="let a group write every supported card's power files, via a udev rule (sudo)")
    gr.add_argument("--group", default=GROUP, help=f"group given write access (default {GROUP}, created when missing)")
    gr.add_argument("--user", help="also add this user to the group")
    gr.add_argument("--print", action="store_true", help="print the udev rule instead of installing it (no privileges)")
    gr.set_defaults(func=cmd_grant)
    cl = psub.add_parser("container-lines", help="on the host: the lines any container adds to apply the settings")
    cl.add_argument("--group", default=GROUP)
    cl.set_defaults(func=cmd_container_lines)
    rv = psub.add_parser("revoke", help="remove the rule grant installed (sudo)")
    rv.set_defaults(func=cmd_revoke)
