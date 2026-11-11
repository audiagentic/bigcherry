"""RSC11 S13 staged v2 supervisor CLI; v1 is the only production contract.

No serving start is permitted until the atomic v2 cutover installs genuine
native measurement, strict S10 writer and pre-weight engine validator.
Stop uses the owning supervisor's verified cross-process request, never PID
signals or a different process's GPU power restoration.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path

from ..core.context import resolve_work_root
from ..engine_adapters.radiance_inventory import InventoryError
from ..hardware import gfxoff, lease
from ..hardware.power_receipt import read_json
from ..serving.activation import V2Bindings, production_v2_bindings
from ..serving.runtime_adapters import (
    LifecycleLimits,
    StopLimits,
    request_stop,
    start_with_adapters,
)
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
    gfxoff_capture: Callable = gfxoff.sample_cards,
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
        gfxoff_capture=gfxoff_capture,
    )
    return supervisor.execute(name=args.name, run_id=args.run_id)


def _start_limits(args: Namespace) -> LifecycleLimits:
    fields = ("readiness_s", "terminate_s", "poll_s", "stop_wait_s")
    if any(getattr(args, field, None) is None for field in fields):
        raise ServeRefused(
            "v2 start requires explicit --readiness-s, --terminate-s, "
            "--poll-s and --stop-wait-s; no guessed lifecycle intervals"
        )
    return LifecycleLimits(**{field: getattr(args, field) for field in fields})


def start_v2_with_bindings(
    args: Namespace,
    bindings: V2Bindings,
    *,
    lease_ok: Callable = lease.held_by_this_process,
    **adapter_options,
) -> dict:
    """Single S11/S12/P3 route; only real S13(c) may install bindings."""
    if not isinstance(bindings, V2Bindings):
        raise ServeRefused("v2 start missing strict native S10 planning services")
    return start_with_adapters(
        name=args.name,
        run_id=args.run_id,
        resolve=lambda name: layout_cmds._resolve(args),
        source=bindings.source,
        writer=bindings.writer,
        validator=bindings.validator,
        record_dir=_record_dir(),
        journal_dir=_journal_dir(),
        lease_ok=lease_ok,
        limits=_start_limits(args),
        explicit_power_profile=args.power_profile,
        qualification_gate=bindings.qualification_gate,
        **adapter_options,
    )


def _verified_v2_record(run_id: str) -> dict:
    """Never send a supervisor stop request to a v1 or unknown run."""
    if not valid_run_id(run_id):
        raise ServeRefused("invalid run ID")
    record_path = _record_dir() / f"{run_id}.json"
    if not record_path.is_file():
        raise ServeRefused(f"no supervisor record for {run_id}")
    record = read_json(record_path)
    manifest_name = record.get("manifest")
    if record.get("run_id") != run_id or not isinstance(manifest_name, str):
        raise ServeRefused("no authenticated v2 supervisor manifest in run record")
    manifest = read_json(Path(manifest_name))
    if manifest.get("apiVersion") != "bigcherry/v2":
        raise ServeRefused("v2 stop refused: run manifest is not bigcherry/v2")
    if "power_receipt" not in record:
        raise ServeRefused("v2 stop refused: supervisor power-receipt identity missing")
    return record


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
            # This is deliberately fail-closed in the pre-cutover release:
            # no v1 launcher, no inactive v2 writer, no implicit power plan.
            record = start_v2_with_bindings(args, production_v2_bindings())
            print(json.dumps(record, indent=2))
            return 0
        if args.serve_action == "stop":
            _verified_v2_record(args.run_id)
            if args.poll_s is None or args.stop_wait_s is None:
                raise ServeRefused(
                    "v2 stop requires --poll-s and --stop-wait-s; no hidden timeout"
                )
            record = request_stop(
                args.run_id, _record_dir(),
                limits=StopLimits(
                    poll_s=args.poll_s, stop_wait_s=args.stop_wait_s
                ),
            )
            print(json.dumps(record, indent=2))
            return 0
        raise ServeRefused("unknown serve action")
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
            for key in ("readiness", "terminate", "poll", "stop-wait"):
                c.add_argument(f"--{key}-s", type=float)
    c = children.add_parser("stop")
    c.add_argument("run_id")
    c.add_argument("--poll-s", type=float)
    c.add_argument("--stop-wait-s", type=float)
    c = children.add_parser("status")
    c.add_argument("run_id")
    p.set_defaults(func=cmd_serve)
