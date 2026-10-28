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


#: where the udev rule goes (udev reads /etc/udev/rules.d)
RULES = "/etc/udev/rules.d/70-bigcherry-gpu-power.rules"


@dataclass(frozen=True)
class Card:
    locator: str
    driver: str | None
    supported: bool          # the driver has a power backend
    overdrive_on: bool       # its voltage/clock controls exist (backends that need a kernel option to expose them)
    name: str = ""           # model, for people (lspci when present, else the PCI vendor:device id)

    def label(self) -> str:
        return f"{self.locator} {self.name}".strip()


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


def _card_name(locator: str) -> str:
    """The card's model as lspci names it (when pciutils is installed), else its PCI vendor:device id."""
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    if shutil.which("lspci"):
        done = subprocess.run(["lspci", "-s", locator], capture_output=True, text=True)
        text = done.stdout.strip()
        if done.returncode == 0 and ": " in text:
            return "(" + text.split(": ", 1)[1].split(" (rev")[0] + ")"
    ids = [(_read_text(f"{gpu_power.SYSFS_PCI}/{locator}/{k}") or "?").replace("0x", "") for k in ("vendor", "device")]
    return f"({ids[0]}:{ids[1]})"


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="ascii") as f:
            return f.read().strip()
    except OSError:
        return None


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
                          bool(backend) and backend.overdrive_on(locator), _card_name(locator)))
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
    """The numbered steps still needed, as lines. ``cmd(*words)`` renders a bigcherry command as this session runs it
    (its checkout and environment); host steps for a container are plain shell, so the host needs no bigcherry."""
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
                       f"{', '.join(c.label() for c in od_off if c.driver == driver)}), then reboot.")
    missing_rule = {c.locator for c in supported} - p.rule_cards
    if not on_host:
        # From inside a container: the host needs no bigcherry, only these commands; the container already knows the rule
        # and the cards' paths, and the group's id is the one thing only the host can print.
        rules = gpu_power.udev_rules([(c.locator, c.driver) for c in supported], p.group)
        out.append("\n".join([
            "On the host, run (needs nothing from bigcherry; it gives one group write access to the power files of "
            f"{', '.join(c.label() for c in supported)}, and prints that group's id):",
            # unindented, so it pastes into a shell as is (a heredoc's EOF must start its line)
            f"sudo groupadd --system {p.group} 2>/dev/null || true",
            f"sudo tee {RULES} >/dev/null <<'EOF'",
            *rules.splitlines(),
            "EOF",
            "sudo udevadm control --reload && sudo udevadm trigger --action=change "
            "--subsystem-match=pci --subsystem-match=hwmon && sudo udevadm settle",
            f"getent group {p.group} | cut -d: -f3",
        ]))
        dirs = [os.path.realpath(f"{gpu_power.SYSFS_PCI}/{c.locator}") for c in supported]
        out.append("\n".join([
            "Add to this container's compose service (merge with any existing lists), with the id the last command "
            "printed, then recreate it (compose up -d <service>; a restart keeps the old mounts and groups). The same "
            "lines serve every container on that host:",
            "       group_add:",
            '         - "<that id>"',
            "       volumes:",
            *[f"         - {d}:{d}:rw" for d in dirs],
        ]))
    elif not p.group_exists or not p.rule_installed or missing_rule or not p.user_in_group:
        what = []
        if not p.group_exists:
            what.append(f"creates the {p.group} group")
        if not p.rule_installed or missing_rule:
            what.append("lets it write the power files of these cards through a udev rule: "
                        + ", ".join(c.label() for c in supported))
        if not p.user_in_group:
            what.append(f"adds {p.user} to it")
        out.append(f"sudo {cmd('gpu-power', 'grant', '--user', p.user)}   ({'; '.join(what)})")
        if not p.session_in_group:
            out.append(f"Log out and back in (or run: newgrp {p.group}) so this session has the {p.group} group.")
    elif not p.session_in_group:
        out.append(f"Log out and back in (or run: newgrp {p.group}) so this session has the {p.group} group.")
    if out:
        out.append("Run the launch again.")
    if unsupported:
        out.append(f"(Not covered: {', '.join(f'{c.label()} [{c.driver}]' for c in unsupported)} — no power backend "
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
    return "\n".join([head, "To set it up:", *numbered])


def container_lines(gid: int, dirs: list[str]) -> str:
    """What any container on this host adds so it can write the cards' power files: the group's id here, and each card's
    sysfs directory read-write (a container's /sys is read-only). The same for every container."""
    mounts = [f"{d}:{d}:rw" for d in dirs]
    return "\n".join([
        "For a container to apply the power settings, add to its compose service (merge with any existing lists):",
        "      group_add:",
        f"        - \"{gid}\"",
        "      volumes:",
        *[f"        - {m}" for m in mounts],
        "or to its run options:",
        f"  --group-add {gid} " + " ".join(f"-v {m}" for m in mounts),
        "then recreate the container (docker/podman compose up -d <service>; a restart keeps the old mounts and groups).",
    ])
