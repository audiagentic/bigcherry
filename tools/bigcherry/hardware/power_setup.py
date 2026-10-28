"""PWR04: what to tell someone whose launch cannot write the GPU power settings.

The setup is one group on the host with write access to the GPUs' power files (a udev rule per driver). On the host, the
user joins that group; in a container, the container gets that group and the GPUs' sysfs directories mapped read-write
(a container's /sys is read-only). The message lists only the steps still missing.

``probe`` reads the system; ``steps`` is pure, so each case is testable.
"""

from __future__ import annotations

import getpass
import os
import sys
from dataclasses import dataclass

from . import gpu_power

#: where the udev rule goes (udev reads /etc/udev/rules.d)
RULES = "/etc/udev/rules.d/70-bigcherry-gpu-power.rules"


@dataclass(frozen=True)
class Card:
    locator: str
    driver: str
    overdrive_on: bool   # its voltage/clock controls exist (amdgpu needs a kernel option for them)


@dataclass(frozen=True)
class Probe:
    linux: bool
    container: str | None    # "docker", "podman", "container", or None on the host
    cards: tuple[Card, ...]  # GPUs whose driver has a power backend
    group: str
    group_exists: bool
    rule_installed: bool
    user: str
    user_in_group: bool      # in the group database
    session_in_group: bool   # this process has the group (a new login picks up a new membership)


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


def probe(group: str) -> Probe:
    linux = sys.platform.startswith("linux")
    cards = tuple(Card(loc, drv, gpu_power.BACKENDS[drv].overdrive_on(loc))
                  for loc, drv in (gpu_power.display_cards() if linux else []) if drv in gpu_power.BACKENDS)
    gid, members = None, []
    try:
        import grp  # noqa: PLC0415 -- POSIX only
        entry = grp.getgrnam(group)
        gid, members = entry.gr_gid, list(entry.gr_mem)
    except (ImportError, KeyError):
        pass
    user = os.environ.get("SUDO_USER") or getpass.getuser()
    return Probe(
        linux=linux, container=container_engine() if linux else None, cards=cards, group=group,
        group_exists=gid is not None, rule_installed=os.path.exists(RULES), user=user, user_in_group=user in members,
        session_in_group=gid is not None and hasattr(os, "getgroups") and gid in os.getgroups(),
    )


def steps(p: Probe, cmd) -> list[str]:
    """The steps still needed. ``cmd(*words)`` renders a bigcherry command as this session runs it."""
    if not p.linux:
        return ["GPU power settings use Linux sysfs; this system is not Linux."]
    if not p.cards:
        return [f"No GPU here has a supported power driver ({', '.join(sorted(gpu_power.BACKENDS))})."]
    out: list[str] = []
    for driver in sorted({c.driver for c in p.cards if not c.overdrive_on}):
        out.append(f"{'On the host: ' if p.container else ''}add {gpu_power.BACKENDS[driver].kernel_option} to the "
                   "kernel command line (your bootloader's config), then reboot.")
    if p.container:
        rules = gpu_power.udev_rules([c.driver for c in p.cards], p.group)
        dirs = [os.path.realpath(f"{gpu_power.SYSFS_PCI}/{c.locator}") for c in p.cards]
        out.append("\n".join([
            f"On the host, create the group {p.group} with write access to the GPUs' power settings:",
            f"sudo groupadd --system {p.group}",
            f"sudo tee {RULES} >/dev/null <<'EOF'",
            *rules.splitlines(),
            "EOF",
            "sudo udevadm control --reload && sudo udevadm trigger",
            f"getent group {p.group}    # the number after the second colon is its id",
        ]))
        out.append("\n".join([
            "In this container's compose service, add the group and map the GPUs, then recreate the container "
            "(compose up -d; a restart keeps the old settings):",
            "  group_add:",
            f'    - "<id of {p.group}>"',
            "  volumes:",
            *[f"    - {d}:{d}:rw" for d in dirs],
        ]))
    else:
        if not p.group_exists or not p.rule_installed:
            out.append(f"sudo {cmd('gpu-power', 'grant')}    (creates the group {p.group} with write access to the "
                       "GPUs' power settings)")
        if not p.user_in_group:
            out.append(f"sudo usermod -aG {p.group} {p.user}    then log out and back in")
        elif not p.session_in_group:
            out.append(f"Log out and back in (or run: newgrp {p.group}).")
    if out:
        out.append("Run the launch again.")
    return out


def render(p: Probe, cmd) -> str:
    where = f"in this {p.container} container" if p.container else f"as {p.user}"
    todo = steps(p, cmd)
    if not todo:
        return f"gpu-power: cannot write the GPU power settings {where}, although the setup looks complete."
    return "\n".join([f"gpu-power: cannot write the GPU power settings {where}. To set it up once:",
                      *[f"{i}. {s}" for i, s in enumerate(todo, 1)]])
