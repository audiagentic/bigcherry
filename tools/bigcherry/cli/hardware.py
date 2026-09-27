"""Operator/agent CLI for discovered-and-accepted hardware state."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

from ..core.context import ProjectContext
from ..hardware.drift import classify_drift
from ..hardware.inventory import InventoryCatalog
from ..hardware.model import inventory_from_mapping
from ..hardware.slurm import render_gres_conf, render_node_gres


def _emit(value: object) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _catalog() -> InventoryCatalog:
    context = ProjectContext.resolve()
    root = Path(
        os.environ.get("BIGCHERRY_HARDWARE_ROOT", context.work_root / "hardware")
    ).resolve()
    return InventoryCatalog(root)


def _inventory_dict(inventory) -> dict[str, object]:
    value = dataclasses.asdict(inventory)
    value["material_hash"] = inventory.material_hash
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry hardware")
    sub = parser.add_subparsers(dest="action", required=True)

    show = sub.add_parser("show")
    show.add_argument("executor_id")
    show.add_argument("--kind", choices=("accepted", "observed"), default="accepted")

    diff = sub.add_parser("diff")
    diff.add_argument("executor_id")

    record = sub.add_parser("record-observed")
    record.add_argument("executor_id")
    record.add_argument("inventory", type=Path)

    accept = sub.add_parser("accept")
    accept.add_argument("executor_id")
    accept.add_argument("--expected-hash", required=True)
    accept.add_argument("--allow-material-change", action="store_true")

    gres = sub.add_parser("render-gres")
    gres.add_argument("executor_id")
    gres.add_argument("--kind", choices=("accepted", "observed"), default="accepted")

    args = parser.parse_args(argv)
    try:
        catalog = _catalog()
        if args.action == "show":
            inventory = (
                catalog.load(args.executor_id)
                if args.kind == "accepted"
                else catalog.load_observed(args.executor_id)
            )
            _emit({"executor_id": args.executor_id, "kind": args.kind, "inventory": _inventory_dict(inventory)})
            return 0
        if args.action == "diff":
            accepted = catalog.load(args.executor_id)
            observed = catalog.load_observed(args.executor_id)
            report = classify_drift(accepted, observed)
            _emit({
                "executor_id": args.executor_id,
                "accepted_hash": accepted.material_hash,
                "observed_hash": observed.material_hash,
                "drift": dataclasses.asdict(report),
            })
            return 2 if report.material else 0
        if args.action == "record-observed":
            value = json.loads(args.inventory.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("inventory file must contain a JSON object")
            value.pop("material_hash", None)
            inventory = inventory_from_mapping(value)
            path = catalog.record_observed(args.executor_id, inventory)
            _emit({
                "executor_id": args.executor_id,
                "observed_path": str(path),
                "material_hash": inventory.material_hash,
            })
            return 0
        if args.action == "accept":
            inventory = catalog.accept_observed(
                args.executor_id,
                expected_material_hash=args.expected_hash,
                allow_material_change=args.allow_material_change,
            )
            _emit({
                "executor_id": args.executor_id,
                "accepted": True,
                "material_hash": inventory.material_hash,
            })
            return 0
        if args.action == "render-gres":
            inventory = (
                catalog.load(args.executor_id)
                if args.kind == "accepted"
                else catalog.load_observed(args.executor_id)
            )
            _emit({
                "executor_id": args.executor_id,
                "kind": args.kind,
                "material_hash": inventory.material_hash,
                "node_gres": render_node_gres(inventory),
                "gres_conf": render_gres_conf(inventory),
            })
            return 0
        raise AssertionError(args.action)
    except Exception as exc:
        _emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
