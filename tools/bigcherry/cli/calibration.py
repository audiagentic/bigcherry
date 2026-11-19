"""Read-only CLI for platform calibration stores and native Radiance TOML."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ..core.calibration_records import (
    CalibrationError,
    CalibrationStore,
    read_native_toml,
)


def _source(path: Path) -> list[dict]:
    if path.suffix.lower() == ".toml":
        return read_native_toml(path)
    if path.is_dir() and not path.is_symlink():
        return [
            {
                "kind": r["kind"],
                "key": r["key"]["sha256"],
                "status": "valid",
                "reason": None,
                "record": r,
            }
            for r in CalibrationStore(path).list()
        ]
    raise CalibrationError(
        "store must be a native TOML file or platform JSON store directory"
    )


def cmd_calibration(args) -> int:
    """List, inspect, or check calibrated evidence; never activate a policy."""
    store = args.store or (
        Path(os.environ["BC_WORK_ROOT"]) / "live-policy-calibration.toml"
        if os.environ.get("BC_WORK_ROOT")
        else None
    )
    if store is None:
        print("calibration: --store PATH or BC_WORK_ROOT is required", file=sys.stderr)
        return 2
    try:
        rows = _source(Path(store))
        if args.calibration_action == "show":
            matches = [
                r for r in rows if r["kind"] == args.kind and r["key"] == args.key
            ]
            if not matches:
                raise CalibrationError(f"no record for {args.kind} / {args.key}")
            payload = matches[0]
        elif args.calibration_action == "validate":
            invalid = [r for r in rows if r["status"] != "valid"]
            payload = {
                "store": str(store),
                "total": len(rows),
                "valid": len(rows) - len(invalid),
                "unusable": len(invalid),
                "records": [
                    {
                        "kind": r["kind"],
                        "key": r["key"],
                        "status": r["status"],
                        "reason": r["reason"],
                    }
                    for r in rows
                ],
            }
            print(json.dumps(payload, sort_keys=True, indent=2))
            return 1 if invalid else 0
        else:
            payload = [
                {
                    "kind": r["kind"],
                    "key": r["key"],
                    "status": r["status"],
                    "reason": r["reason"],
                    "source_build": (r["record"] or {})
                    .get("provenance", {})
                    .get("build_id"),
                }
                for r in rows
            ]
        print(json.dumps(payload, sort_keys=True, indent=2))
        return 0
    except (CalibrationError, OSError, ValueError) as exc:
        print(f"calibration: {exc}", file=sys.stderr)
        return 1
