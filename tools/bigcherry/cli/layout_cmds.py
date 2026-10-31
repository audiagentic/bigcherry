"""`bigcherry layout list|show|env` (RAD23): the remote-MoE offload's card types and layouts, resolved against this host.

``show`` prints what a layout resolves to; ``env`` prints it as shell assignments for the launcher (no architecture, index or budget lives
in the launcher). Both refuse an inconsistent layout with the reason. Data: engines/radiance/archs.toml, engines/radiance/layouts.toml,
this host's device inventory.
"""

from __future__ import annotations

import json
import shlex
import sys
from argparse import Namespace

from ..core import environment
from ..engine_adapters import radiance_layout


def _budgets(items: list[str] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items or ():
        arch, sep, mib = item.partition("=")
        if not sep or not mib.isdigit() or not arch.startswith("gfx"):
            raise radiance_layout.LayoutError(f"--budget wants ARCH=MIB (gfx1030=4096), got {item!r}")
        out[arch] = int(mib)
    return out


def _resolve(args: Namespace) -> radiance_layout.Resolved:
    host = environment.load_default().host(getattr(args, "host", None))
    return radiance_layout.resolve(args.name, host, budgets=_budgets(args.budget))


def _env_lines(r: radiance_layout.Resolved) -> list[str]:
    q = shlex.quote
    lines = [f"LAYOUT_ENGINE_DEVICE={q(str(r.engine_index))}", f"LAYOUT_ENGINE_PLUGIN={q(r.engine_plugin[:-3])}", f"LAYOUT_HELPER_COUNT={len(r.helpers)}", "LAYOUT_COMMON=(" + " ".join(q(f"{k}={v}") for k, v in r.common_env.items()) + ")"]
    for i in range(len(r.helpers)):
        lines.append(f"LAYOUT_HELPER_{i}=(" + " ".join(q(f"{k}={v}") for k, v in r.helper_env(i).items()) + ")")
    lines.append("LAYOUT_SUMMARY=" + q("; ".join(f"helper{i}: card {h.index} {h.arch.name} {h.budget_mib} MiB {h.placement}" for i, h in enumerate(r.helpers))))
    return lines


def _run(args: Namespace) -> int:
    try:
        if args.layout_cmd == "list":
            for n in radiance_layout.names():
                print(n)
            return 0
        if args.layout_cmd == "card-env":
            # what a helper process on this inventory card needs (plugin, formats, few-rows kernel), as KEY=VALUE words for `env`
            host = environment.load_default().host(getattr(args, "host", None))
            archs = radiance_layout.load_archs()
            card = {d.index: d for d in host.devices}.get(args.device)
            if card is None or card.arch not in archs:
                raise radiance_layout.LayoutError(f"device {args.device} is " + ("not in the host inventory" if card is None else f"{card.arch}, which archs.toml does not describe"))
            print(" ".join(f"{k}={v}" for k, v in radiance_layout.card_env(archs[card.arch]).items()))
            return 0
        r = _resolve(args)
        if args.layout_cmd == "env":
            print("\n".join(_env_lines(r)))
            return 0
        print(json.dumps({
            "layout": r.name, "engine": {"device": r.engine_index, "arch": r.engine_arch, "plugin": r.engine_plugin},
            "helpers": [{"helper": i, "device": h.index, "arch": h.arch.name, "vram_mib": h.vram_mib, "budget_mib": h.budget_mib, "placement": h.placement,
                         "share": h.share, "plugin": h.arch.plugin, "formats": list(h.arch.formats), "rows_kernel": h.arch.rows_kernel} for i, h in enumerate(r.helpers)],
            "common_env": r.common_env}, indent=2))
        return 0
    except (radiance_layout.LayoutError, environment.EnvironmentError_) as exc:
        print(f"layout: {exc}", file=sys.stderr)
        return 2


def register(sub) -> None:
    p = sub.add_parser("layout", help="the remote-MoE offload's layouts, resolved against this host (RAD23)")
    ls = p.add_subparsers(dest="layout_cmd", required=True)
    ls.add_parser("list", help="the layouts in engines/radiance/layouts.toml")
    for name, helptext in (("show", "what a layout resolves to on this host (JSON)"), ("env", "shell assignments for the launcher")):
        c = ls.add_parser(name, help=helptext)
        c.add_argument("name")
        c.add_argument("--host", default=None, help="host in config/environment.local.toml (default: the default host)")
        c.add_argument("--budget", action="append", metavar="ARCH=MIB", help="override the VRAM budget of the group using this architecture")
    c = ls.add_parser("card-env", help="the helper environment a card of this host needs (KEY=VALUE words)")
    c.add_argument("device", type=int, help="the inventory index (the HIP/ROCR ordinal)")
    c.add_argument("--host", default=None, help="host in config/environment.local.toml (default: the default host)")
    p.set_defaults(func=_run)
