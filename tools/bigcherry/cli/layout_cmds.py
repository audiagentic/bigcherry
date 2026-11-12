"""Read-only layout inspection from bc-serve's structured KFD/PCI resolution.

Recipes describe models and preferences; they do not assign physical devices.
The C++ resolver is the only source of engine/helper BDF and HIP ordinal roles.
"""
from __future__ import annotations

import json
import shlex
import sys
from argparse import Namespace
from pathlib import Path

from ..core import environment
from ..engine_adapters import radiance_inventory, radiance_layout


def _resolve(args: Namespace) -> radiance_inventory.Layout:
    if getattr(args, "budget", None):
        raise radiance_inventory.InventoryError(
            "--budget is not supported by the C++ serving resolver; configure "
            "memory.helper_budgets_mib in a bc-serve site/profile instead"
        )
    host = environment.load_default().host(getattr(args, "host", None))
    return radiance_inventory.resolve(
        args.name, host=host, binary=getattr(args, "bc_serve", None),
        site=getattr(args, "site", None), profile=getattr(args, "profile", None),
        sys_root=getattr(args, "sys_root", None),
    )


def _env_lines(r: radiance_layout.Resolved) -> list[str]:
    q = shlex.quote
    lines = [
        f"LAYOUT_ENGINE_DEVICE={r.engine_index}",
        "LAYOUT_ENGINE_DEVICES=" + q(",".join(map(str, r.engine_indices))),
        f"LAYOUT_ENGINE_RANKS={len(r.engine_indices)}",
        "LAYOUT_ENGINE_PLUGIN=" + q(r.engine_plugin[:-3]),
        "LAYOUT_MODEL=" + q(r.model),
        "LAYOUT_TP_WIRE=" + q(r.tp_wire or ""),
        "LAYOUT_SERVER_ARGS=(" + " ".join(q(arg) for arg in r.server_args) + ")",
        f"LAYOUT_HELPER_COUNT={len(r.helpers)}",
        "LAYOUT_COMMON=(" + " ".join(q(f"{k}={v}") for k, v in r.common_env.items()) + ")",
    ]
    for i in range(len(r.helpers)):
        lines.append(f"LAYOUT_HELPER_{i}=(" + " ".join(q(f"{k}={v}") for k, v in r.helper_env(i).items()) + ")")
    ranks = f"TP ranks: {','.join(map(str, r.engine_indices))} ({r.engine_arch}), wire {r.tp_wire or 'default'}"
    helpers = "; ".join(f"helper{i}: card {h.index} {h.arch.name} "
                        f"{h.budget_mib} MiB {h.placement}" for i, h in enumerate(r.helpers))
    summary = (f"{ranks}; {helpers}" if len(r.engine_indices) > 1 else helpers) if r.helpers else ranks
    lines.append("LAYOUT_SUMMARY=" + q(summary))
    return lines


def _run(args: Namespace) -> int:
    try:
        if args.layout_cmd == "list":
            for n in radiance_layout.names():
                print(n)
            return 0
        host = environment.load_default().host(getattr(args, "host", None))
        if args.layout_cmd == "card-env":
            inventory = radiance_inventory.detect(binary=args.bc_serve, sys_root=args.sys_root)
            inventory.check_power_overrides(host)
            archs = radiance_layout.load_archs()
            card = inventory.by_ordinal(args.device)
            if card.arch not in archs:
                raise radiance_inventory.InventoryError(f"device {args.device} is {card.arch}, which archs.toml does not describe")
            if "helper" not in archs[card.arch].roles:
                raise radiance_inventory.InventoryError(f"device {args.device} is {card.arch}, which archs.toml does not allow as a helper")
            print(" ".join(shlex.quote(f"{k}={v}") for k, v in radiance_layout.card_env(archs[card.arch]).items()))
            return 0
        layout = _resolve(args)
        r = layout.resolved
        if args.layout_cmd == "env":
            print("\n".join(_env_lines(r) + [
                "LAYOUT_ENGINE_BDFS=(" + " ".join(shlex.quote(b) for b in layout.engine_bdfs) + ")"
            ]))
            return 0
        print(json.dumps({
            "layout": r.name, "model": r.model,
            "engine": {"device": r.engine_index, "devices": list(r.engine_indices),
                       "bdfs": list(layout.engine_bdfs),
                       "tp": len(r.engine_indices), "tp_wire": r.tp_wire,
                       "arch": r.engine_arch, "plugin": r.engine_plugin,
                       "server_args": list(r.server_args)},
            "helpers": [{"helper": i, "device": h.index, "bdf": layout.helper_bdfs[i],
                         "arch": h.arch.name, "vram_mib": h.vram_mib,
                         "budget_mib": h.budget_mib, "placement": h.placement,
                         "share": h.share, "plugin": h.arch.plugin,
                         "formats": list(h.arch.formats), "rows_kernel": h.arch.rows_kernel}
                        for i, h in enumerate(r.helpers)],
            "common_env": r.common_env,
        }, indent=2))
        return 0
    except (radiance_inventory.InventoryError, radiance_layout.LayoutError,
            environment.EnvironmentError_) as exc:
        print(f"layout: {exc}", file=sys.stderr)
        return 2


def register(sub) -> None:
    p = sub.add_parser("layout", help="layouts resolved by the bc-serve KFD/PCI detector (RSC11)")
    ls = p.add_subparsers(dest="layout_cmd", required=True)
    ls.add_parser("list", help="the named layouts in the layout definitions (BIGCHERRY_LAYOUTS; default engines/radiance/layouts/)")
    for name, helptext in (("show", "resolved role/BDF assignments (JSON)"),
                           ("env", "shell assignments from the serving resolver")):
        c = ls.add_parser(name, help=helptext)
        c.add_argument("name")
        c.add_argument("--host", default=None, help="host scalar settings and BDF-scoped power overrides only")
        c.add_argument("--bc-serve", type=Path, help="the serving resolver binary")
        c.add_argument("--sys-root", type=Path, help="fake filesystem root for offline tests")
        c.add_argument("--site", type=Path, help="bc-serve site configuration")
        c.add_argument("--profile", type=Path, help="bc-serve profile")
        c.add_argument("--budget", action="append", help="retired: use bc-serve site/profile helper budgets")
    c = ls.add_parser("card-env", help="helper settings for a detected HIP ordinal")
    c.add_argument("device", type=int)
    c.add_argument("--host", default=None, help="BDF-scoped power overrides only")
    c.add_argument("--bc-serve", type=Path)
    c.add_argument("--sys-root", type=Path)
    p.set_defaults(func=_run)
