"""PWR04: what a host still needs before an unprivileged launch can apply GPU power settings, as numbered steps.

Nothing about the host is assumed: every prerequisite is probed (Linux sysfs, a container and which engine, whether that
container runs its own udev, each card's driver and OverDrive state, the access group, the udev rule and which cards it
covers, the user's membership in the group database and in this session), and the steps list only what is missing, in
the order it must be done. A fresh install therefore gets the whole setup; a host with everything in place gets none.

``probe`` reads the system; ``steps`` is pure, so every combination is testable.
"""

from __future__ import annotations

import getpass
import os
import sys
from dataclasses import dataclass, field

from . import gpu_power


@dataclass(frozen=True)
class Card:
    locator: str
    driver: str | None
    supported: bool          # the driver has a power backend
    overdrive_on: bool       # its voltage/clock controls exist (backends that need a kernel option to expose them)


@dataclass(frozen=True)
class Probe:
    linux: bool
    container: str | None    # "docker", "podman", "container" (some other engine), None on the host itself
    container_udev: bool     # the container runs its own udevd (then a grant inside it takes effect)
    sys_read_only: bool
    cards: tuple[Card, ...]
    group: str
    group_exists: bool
    gid: int | None
    rule_installed: bool
    rule_cards: frozenset[str]          # locators the installed rule covers
    user: str
    user_in_group: bool                 # per the group database
    session_in_group: bool              # this process carries the group
    blocked: tuple[str, ...] = field(default=())   # the files this launch must write but cannot


def container_engine() -> str | None:
    if os.path.exists("/.dockerenv"):
        return "docker"
    if os.path.exists("/run/.containerenv"):
        return "podman"
    try:
        with open("/proc/1/cgroup", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    return "container" if any(k in text for k in ("docker", "containerd", "kubepods", "libpod", "lxc")) else None


def _own_udevd() -> bool:
    """A udevd in this process's mount namespace (one that would act on a rule written here). A container sharing the
    host's PID namespace sees the host's udevd, which is in another mount namespace and does not count."""
    try:
        mine = os.readlink("/proc/self/ns/mnt")
    except OSError:
        return False
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/comm", encoding="utf-8") as f:
                if f.read().strip() not in ("systemd-udevd", "udevd"):
                    continue
            if os.readlink(f"/proc/{pid}/ns/mnt") == mine:
                return True
        except OSError:
            continue
    return False


def _rule_locators(path: str) -> frozenset[str]:
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return frozenset()
    import re  # noqa: PLC0415
    return frozenset(re.findall(r'KERNELS?=="([0-9a-fA-F:.]+)"', text))


def probe(group: str, rules_path: str, blocked: list[str]) -> Probe:
    linux = sys.platform.startswith("linux")
    engine = container_engine() if linux else None
    cards = []
    for locator, driver in (gpu_power.display_cards() if linux else []):
        backend = gpu_power.BACKENDS.get(driver)
        cards.append(Card(locator, driver, backend is not None,
                          bool(backend) and backend.overdrive_on(locator)))
    gid = None
    members: list[str] = []
    try:
        import grp  # noqa: PLC0415 -- POSIX only
        entry = grp.getgrnam(group)
        gid, members = entry.gr_gid, list(entry.gr_mem)
    except (ImportError, KeyError):
        pass
    user = os.environ.get("SUDO_USER") or getpass.getuser()
    session = gid is not None and hasattr(os, "getgroups") and gid in os.getgroups()
    sys_ro = False
    try:
        sys_ro = bool(os.statvfs("/sys").f_flag & os.ST_RDONLY)
    except (OSError, AttributeError):
        pass
    return Probe(
        linux=linux, container=engine, container_udev=bool(engine) and _own_udevd(),
        sys_read_only=sys_ro, cards=tuple(cards), group=group, group_exists=gid is not None, gid=gid,
        rule_installed=os.path.exists(rules_path), rule_cards=_rule_locators(rules_path), user=user,
        user_in_group=user in members, session_in_group=session, blocked=tuple(blocked),
    )


def _bootloader_line(option: str) -> str:
    if os.path.exists("/etc/default/grub"):
        return (f"add {option} to GRUB_CMDLINE_LINUX in /etc/default/grub, regenerate the GRUB config "
                "(update-grub, or grub2-mkconfig -o <your grub.cfg>)")
    return f"add {option} to the kernel command line in your bootloader's configuration"


def steps(p: Probe, cmd) -> list[str]:
    """The numbered steps still needed, as lines. ``cmd(*words, host_side)`` renders a bigcherry command: with this
    checkout's paths when it runs here, or as run from a checkout on the host when the step happens on the host."""
    if not p.linux:
        return ["GPU power settings are applied through Linux sysfs; this system is not Linux, so they cannot be applied."]
    out: list[str] = []
    unsupported = [c for c in p.cards if not c.supported]
    supported = [c for c in p.cards if c.supported]
    if not supported:
        return [f"No GPU here has a supported power driver (found: "
                f"{', '.join(f'{c.locator} {c.driver}' for c in p.cards) or 'none'}; supported: "
                f"{', '.join(sorted(gpu_power.BACKENDS))})."]
    on_host = p.container is None or p.container_udev
    where = "" if on_host else "On the host: "
    od_off = [c for c in supported if not c.overdrive_on]
    if od_off:
        for driver in sorted({c.driver for c in od_off}):
            option = gpu_power.BACKENDS[driver].kernel_option
            out.append(f"{where}{_bootloader_line(option)} (turns on the voltage/clock controls of "
                       f"{', '.join(c.locator for c in od_off if c.driver == driver)}), then reboot.")
    missing_rule = {c.locator for c in supported} - p.rule_cards
    need_grant = not p.group_exists or not p.rule_installed or bool(missing_rule) or (on_host and not p.user_in_group)
    if need_grant:
        what = []
        if not p.group_exists:
            what.append(f"creates the {p.group} group")
        if not p.rule_installed or missing_rule:
            what.append(f"lets it write the power files of {len(supported)} card(s) through a udev rule")
        if on_host and not p.user_in_group:
            what.append(f"adds {p.user} to it")
        user = ["--user", p.user] if on_host else []
        out.append(f"{where}sudo {cmd('gpu-power', 'grant', *user, host_side=not on_host)}   ({'; '.join(what)})")
    if p.container and not on_host:
        engine = p.container if p.container in ("docker", "podman") else "docker/podman"
        gid = str(p.gid) if p.gid is not None else f"<gid of {p.group} on the host: getent group {p.group}>"
        dirs = [os.path.realpath(f"/sys/bus/pci/devices/{c.locator}") for c in supported]
        if p.sys_read_only or not p.session_in_group:
            out.append(f"Give this {engine} container the group and the cards' sysfs directories read-write, then "
                       "recreate it (mounts and groups are fixed at creation; a restart keeps the old ones):\n"
                       f"       {engine} run ... --group-add {gid} \\\n"
                       + " \\\n".join(f"         -v {d}:{d}:rw" for d in dirs) + "\n"
                       "     or in its compose service:\n"
                       f"       group_add: [\"{gid}\"]\n"
                       "       volumes:\n" + "\n".join(f"         - {d}:{d}:rw" for d in dirs))
    elif not p.session_in_group:
        out.append(f"Log out and back in (or run: newgrp {p.group}) so this session has the {p.group} group.")
    if out:
        out.append("Run the launch again.")
    if unsupported:
        out.append(f"(Not covered: {', '.join(f'{c.locator} ({c.driver})' for c in unsupported)} — no power backend "
                   "for that driver.)")
    return out


def render(p: Probe, cmd) -> str:
    """The message for a launch that cannot apply its power settings."""
    where = (f"inside a {p.container} container" + ("" if p.container_udev else "; udev runs on the host") if p.container
             else "on the host")
    head = f"gpu-power: {p.user} cannot apply the GPU power settings ({where})."
    todo = steps(p, cmd)
    if not todo:
        return head + " Setup is complete, yet these files are not writable:\n" + "\n".join(f"  {b}" for b in p.blocked)
    numbered = [f"  {i}. {s}" if not s.startswith("(") else f"  {s}" for i, s in enumerate(todo, 1)]
    if p.container and not (p.container_udev):
        alt = cmd("gpu-power", "apply", host_side=True)
        numbered.append(f"  Or, without changing the container: run `{alt}` on the host before each run "
                        "(and `... gpu-power restore` after).")
    return "\n".join([head, "To set it up:", *numbered])
