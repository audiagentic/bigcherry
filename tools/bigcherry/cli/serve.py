"""RSC11 S6 CLI surface; activation waits for S11/S12 and the S16 cutover.

Only 'explain' and durable run-record 'status' are operational without the
injected supervisor services. 'start' and 'stop' deliberately refuse rather
than silently invoking the shell launcher or bypassing lease/power checks.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path

from ..core.context import resolve_work_root
from ..engine_adapters.radiance_inventory import InventoryError
from ..hardware.power_receipt import read_json
from ..serving.supervisor import ProcessLauncher, ServeRefused, Supervisor, valid_run_id
from . import layout_cmds


def _record_dir() -> Path:
    return resolve_work_root() / "serving" / "runs"


def _journal_dir() -> Path:
    # Same transaction journal location as gpu-power. S12 unifies the selection
    # under the resolved v2 power plan; do not invent another power journal.
    from .power_cmds import _journal_dir as power_journal_dir

    return power_journal_dir()


def start_with_services(
    args: Namespace,
    *,
    planner: Callable,
    launcher: ProcessLauncher,
    wait: Callable,
    lease_ok: Callable,
    power_factory: Callable | None = None,
) -> dict:
    """Testable start entry: real integrations are supplied by S11/S12."""
    supervisor = Supervisor(
        resolve=lambda name: layout_cmds._resolve(args),
        plan=planner,
        launcher=launcher,
        wait=wait,
        lease_ok=lease_ok,
        record_dir=_record_dir(),
        journal_dir=_journal_dir(),
        power_factory=power_factory,
    )
    return supervisor.execute(name=args.name, run_id=args.run_id)


def cmd_serve(args: Namespace) -> int:
    try:
        if args.serve_action == "explain":
            layout = layout_cmds._resolve(args)
            print(
                json.dumps(
                    {
                        "layout": layout.resolved.name,
                        "engine_bdfs": layout.engine_bdfs,
                        "helper_bdfs": layout.helper_bdfs,
                        "engine_ordinals": layout.resolved.engine_indices,
                        "helper_ordinals": [h.index for h in layout.resolved.helpers],
                        "plan": "not_measured",
                        "power": "not_applied",
                    },
                    indent=2,
                )
            )
            return 0
        if args.serve_action == "status":
            if not valid_run_id(args.run_id):
                raise ServeRefused("invalid run ID")
            p = _record_dir() / f"{args.run_id}.json"
            if not p.is_file():
                raise ServeRefused(f"no run record for {args.run_id!r}")
            doc = read_json(p)
            if doc.get("run_id") != args.run_id:
                raise ServeRefused("run record identity mismatch")
            # A persisted 'running' record is not proof its process survived.
            if doc.get("state") != "stopped":
                doc["live_state"] = "unverified"
            print(json.dumps(doc, indent=2))
            return 0
        if args.serve_action == "start":
            raise ServeRefused(
                "S6 is a card-free supervisor skeleton: the measured-plan and "
                "verified process launcher are not connected (S11/S12); "
                "the existing shell launcher remains in service until S16"
            )
        raise ServeRefused(
            "cross-process stop is not activated until S11/S12; refusing "
            "to claim a shutdown or restore power without verified child exit"
        )
    except (InventoryError, ServeRefused, OSError, ValueError) as exc:
        print(f"serve: {exc}", file=sys.stderr)
        return 2


def register(sub) -> None:
    p = sub.add_parser("serve", help="single serving supervisor (S6 skeleton)")
    children = p.add_subparsers(dest="serve_action", required=True)
    for action in ("start", "explain"):
        c = children.add_parser(action)
        c.add_argument("name", help="named serving layout")
        c.add_argument("--host", default=None)
        c.add_argument("--bc-serve", type=Path)
        c.add_argument("--sys-root", type=Path)
        c.add_argument("--site", type=Path)
        c.add_argument(
            "--profile", type=Path, help="serving profile (not a power profile)"
        )
        c.add_argument(
            "--power-profile", help="reserved for the S12 v2 resolved power plan"
        )
        if action == "start":
            c.add_argument("--run-id", required=True)
    c = children.add_parser("stop")
    c.add_argument("run_id")
    c = children.add_parser("status")
    c.add_argument("run_id")
    p.set_defaults(func=cmd_serve)
