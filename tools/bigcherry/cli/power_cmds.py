"""`bigcherry gpu-power status|apply|restore|grant|revoke` (PWR04): a host's configured per-card power settings.

The settings come from the environment's device inventory (``[host.*.devices.power]``, environment.py); the sysfs work
is hardware/gpu_power.py and the setup checks hardware/power_setup.py, neither of which knows about bigcherry config.

``status`` reads what each card runs and compares it with the configuration (no privileges). ``apply`` and ``restore``
go through ``hardware.power_transaction``: ``apply --layout NAME --run-id ID`` changes ONLY the cards that layout uses,
after every requested setting passed preflight against the card's own reported ranges, the gpu lease is verified, and a
recovery journal is on disk; any failure rolls everything back and exits non-zero. ``restore --run-id ID
--processes-exited`` puts the journal's original values back and verifies them; it refuses without the assertion that
nothing is using the cards any more, and a failed restore keeps the journal. Both run as whoever launches: root, or a
member of the group ``grant`` gave write access to the cards' power files (a udev rule, installed once with sudo); a
refusal for missing access prints the setup steps still missing on this host.
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
from ..engine_adapters import radiance_layout
from ..hardware import gpu_power, power_setup
from ..hardware import lease as gpu_lease
from ..hardware.power_transaction import PowerFailed, PowerRefused, PowerTransaction

#: where grant writes its rules (udev reads /etc/udev/rules.d)
RULES = Path(power_setup.RULES)
#: the group grant gives the write access to by default: its own, so only users added to it can change voltage and power
#: (video holds every GPU user on most hosts); grant creates it when it is missing
GROUP = "bigcherry-gpu"


def _journal_dir() -> Path:
    """Where the transaction journals live: the work root's power folder (survives the launcher's process)."""
    root = os.environ.get("BC_WORK_ROOT")
    if root:
        return Path(root) / "flashnext" / "power" / "journals"
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return Path(base) / "bigcherry" / "power-journals"


def _setting(power: environment.DevicePower) -> gpu_power.Setting:
    return gpu_power.Setting(power.vddgfx_offset_mv, power.sclk_max_mhz, power.power_cap_w)


def _devices(args: Namespace) -> list[environment.Device]:
    host = environment.load_default().host(args.host)
    return [d for d in host.devices if d.locator]


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


def _command(args: Namespace, *words: str) -> str:
    """A bigcherry command line as this session runs it: this checkout's tools and environment file."""
    host = ["--host", args.host] if getattr(args, "host", None) else []
    env = os.environ.get("BIGCHERRY_ENVIRONMENT")
    cmd = [*(["BIGCHERRY_ENVIRONMENT=" + env] if env else []), f"PYTHONPATH={Path(__file__).resolve().parents[2]}",
           sys.executable or "python3", "-m", "bigcherry", *words, *host]
    return " ".join(shlex.quote(c) for c in cmd)


def _explain(args: Namespace) -> str:
    return power_setup.render(power_setup.probe(GROUP), lambda *w: _command(args, *w))


def _sys() -> dict:
    """Where the transaction reads and writes sysfs: the real one (a seam for the tests' fake tree)."""
    return {}


def _selected(args: Namespace) -> dict[str, environment.Device]:
    """The cards the layout uses, by PCI locator: the only ones a transaction may touch."""
    host = environment.load_default().host(getattr(args, "host", None))
    r = radiance_layout.resolve(args.layout, host, budgets={})
    wanted = list(r.engine_indices) + [h.index for h in r.helpers]
    by_index = {d.index: d for d in host.devices}
    out: dict[str, environment.Device] = {}
    for i in wanted:
        d = by_index.get(i)
        if d is None or not d.locator:
            raise radiance_layout.LayoutError(f"layout {args.layout}: card {i} has no PCI locator in the host inventory")
        out[d.locator] = d
    return out


def cmd_apply(args: Namespace) -> int:
    try:
        selected = _selected(args)
    except radiance_layout.LayoutError as exc:
        print(f"gpu-power apply: {exc}", file=sys.stderr)
        return 2
    requested = {loc: _setting(d.power) for loc, d in selected.items() if d.power}
    tx = PowerTransaction(args.run_id, _journal_dir(), lease_ok=gpu_lease.held_by_this_process, **_sys())
    try:
        tx.prepare(requested, sorted(selected))
        tx.apply()
    except PowerRefused as exc:
        print("gpu-power apply: refused, nothing was written:", file=sys.stderr)
        for reason in exc.reasons:
            print(f"  - {reason}", file=sys.stderr)
        if any("not_writable" in r or "not_permitted" in r or "access_denied" in r for r in exc.reasons):
            print(_explain(args), file=sys.stderr)
        return 2
    except PowerFailed as exc:
        print(f"gpu-power apply: FAILED ({exc}); " + ("everything was put back" if exc.rolled_back else
              f"NOT restored: journal {tx.journal_path} holds the original values (gpu-power restore --run-id {args.run_id})"),
              file=sys.stderr)
        return 1
    for d in tx.devs:
        print(f"{selected[d.bdf].index} {d.bdf}: applied and read back")
    return 0


def cmd_restore(args: Namespace) -> int:
    path = _journal_dir() / f"{args.run_id}.json"
    if not path.exists():
        print(f"gpu-power restore: no journal for run {args.run_id} (nothing applied by that run)")
        return 0
    if not args.processes_exited:
        print("gpu-power restore: refused: pass --processes-exited once nothing is using the cards "
              "(restoring under a running process would change its clocks)", file=sys.stderr)
        return 2
    try:
        ok = PowerTransaction.recover(path, lease_ok=gpu_lease.held_by_this_process, processes_exited=True, **_sys())
    except PowerRefused as exc:
        print("gpu-power restore: refused: " + str(exc), file=sys.stderr)
        return 2
    print("restored to what ran before and verified" if ok else
          f"NOT RESTORED: journal {path} kept with the original values; see its events")
    return 0 if ok else 1


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def _trigger() -> None:
    _run(["udevadm", "control", "--reload"])
    _run(["udevadm", "trigger"])
    _run(["udevadm", "settle"])


def _gid(group: str) -> int | None:
    try:
        import grp  # noqa: PLC0415 -- POSIX only
        return grp.getgrnam(group).gr_gid
    except (ImportError, KeyError):
        return None


def cmd_grant(args: Namespace) -> int:
    """Create the group with write access to the GPUs' power files (a udev rule per supported driver present)."""
    if not sys.platform.startswith("linux"):
        print("gpu-power grant: GPU power settings use Linux sysfs; this system is not Linux", file=sys.stderr)
        return 1
    drivers = sorted({drv for _, drv in gpu_power.display_cards() if drv in gpu_power.BACKENDS})
    if not drivers:
        print("gpu-power grant: no GPU here has a supported power driver", file=sys.stderr)
        return 1
    rules = gpu_power.udev_rules(drivers, args.group)
    if args.print:
        print(rules, end="")
        return 0
    if os.geteuid() != 0:
        print("gpu-power grant: run it with sudo (--print shows the rule without installing it)", file=sys.stderr)
        return 2
    if _gid(args.group) is None:
        _run(["groupadd", "--system", args.group])
    RULES.write_text(rules, encoding="utf-8")
    _trigger()
    print(f"group {args.group} can write the GPUs' power settings. Add a user with: "
          f"sudo usermod -aG {args.group} <user> (then log in again)")
    return 0


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
    ap = psub.add_parser("apply", help="apply the configured settings to the cards of one layout, all or nothing "
                                       "(gpu lease required; root, or after grant)")
    ap.add_argument("--host")
    ap.add_argument("--layout", required=True, help="the layout whose cards may be changed (and no others)")
    ap.add_argument("--run-id", required=True, help="names the recovery journal")
    ap.set_defaults(func=cmd_apply)
    rs = psub.add_parser("restore", help="put back what ran before apply, from its journal (gpu lease required)")
    rs.add_argument("--host")
    rs.add_argument("--run-id", required=True)
    rs.add_argument("--processes-exited", action="store_true",
                    help="assert that no process uses the cards any more (restore refuses without it)")
    rs.set_defaults(func=cmd_restore)
    gr = psub.add_parser("grant", help="create the group with write access to the GPUs' power settings (sudo)")
    gr.add_argument("--group", default=GROUP, help=f"default {GROUP}")
    gr.add_argument("--print", action="store_true", help="print the udev rule instead of installing it")
    gr.set_defaults(func=cmd_grant)
    rv = psub.add_parser("revoke", help="remove the rule grant installed (sudo)")
    rv.set_defaults(func=cmd_revoke)
