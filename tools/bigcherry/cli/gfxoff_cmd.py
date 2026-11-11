"""Nonblocking PWR06 sysfs probe for the serving lifecycle and doctor."""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

from ..hardware import gfxoff
from ..hardware.power_receipt import write_json_atomic


def inspect(
    *, sys_root: Path = Path("/sys"),
    floor_path: Path | None = None,
    engine_idle: bool = False,
    bdfs: list[str] | None = None,
    sampler=gfxoff.sample_cards,
) -> dict[str, dict]:
    if bdfs is None:
        base = sys_root / "bus" / "pci" / "devices"
        bdfs = [
            p.name for p in sorted(base.glob("*"))
            if gfxoff.gfx12_card(p.name, sys_root=sys_root) is True
        ]
    floor = gfxoff.load_windows(floor_path) if floor_path is not None else {}
    if not engine_idle or not floor:
        return {
            bdf: gfxoff.Verdict(
                bdf, "needs_idle_engine_context",
                "supply_a_no_process_floor_and_explicit_idle_engine_context",
                floor[bdf].to_dict() if bdf in floor else None, None,
            ).to_dict() for bdf in bdfs
        }
    loaded = sampler(bdfs, phase="engine_loaded_idle", sys_root=sys_root)
    return {
        bdf: gfxoff.evaluate(floor.get(bdf), loaded.get(bdf)).to_dict()
        for bdf in bdfs
    }


def cmd_gfxoff(args: Namespace) -> int:
    try:
        root = args.sys_root
        if args.gfxoff_action == "floor":
            windows = gfxoff.sample_cards(args.bdf, phase="no_process", sys_root=root)
            gfxoff.store_windows(args.output, windows)
            for bdf, window in windows.items():
                print(f"gfxoff: floor {window.status} on {bdf}" +
                      (f" ({window.reason})" if window.reason else ""), file=sys.stderr)
        else:
            floor = gfxoff.load_windows(args.floor)
            loaded = gfxoff.sample_cards(args.bdf, phase="engine_loaded_idle", sys_root=root)
            verdicts = {b: gfxoff.evaluate(floor.get(b), loaded[b]) for b in args.bdf}
            write_json_atomic(args.output, {
                "schema_version": 1,
                "gfxoff": {b: v.to_dict() for b, v in verdicts.items()},
            })
            for verdict in verdicts.values():
                if verdict.status != "not_applicable":
                    print(gfxoff.format_verdict(verdict), file=sys.stderr)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Health checking and serving must not depend on this diagnostic.
        print(f"gfxoff: unavailable ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 0


def register(sub) -> None:
    parser = sub.add_parser("gfxoff", help="PWR06 read-only GFX12 idle-state probe")
    children = parser.add_subparsers(dest="gfxoff_action", required=True)
    for name in ("floor", "check"):
        p = children.add_parser(name)
        p.add_argument("--bdf", action="append", required=True, help="engine PCI BDF (repeat for TP)")
        p.add_argument("--sys-root", type=Path, default=Path("/sys"))
        p.add_argument("--output", type=Path, required=True)
        if name == "check":
            p.add_argument("--floor", type=Path, required=True)
        p.set_defaults(func=cmd_gfxoff)
