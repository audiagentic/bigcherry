"""CLI presentation handlers for deterministic product diagnostics."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, cast

from .. import __version__, check, doctor
from ..core import paths
from ..source import audit as source_audit
from ..release import records as releases


def cmd_doctor(args: Namespace) -> int:
    """Report migration assumptions without modifying source or build state."""
    return doctor.main(as_json=args.json)


def cmd_status(args: Namespace) -> int:
    root = paths.llama_root(args.llama_root)
    revision, dirty = source_audit.git_revision(root)
    print(f"bigcherry {__version__}")
    print(f"  repo:     {paths.REPO_ROOT}")
    print(f"  checkout: {root}")
    print(f"  revision: {revision[:12]}{' (dirty)' if dirty else ''}")
    print()
    records = releases.all_records()
    if not records:
        print("  no releases recorded yet")
        return 0
    print(f"  {'release':<16} {'stage':<12} {'audit':<7} manifest")
    for record in records:
        audit = (
            "pass" if record.audit.get("passed") else ("fail" if record.audit else "-")
        )
        print(
            f"  {record.slug():<16} {record.stage:<12} {audit:<7} {record.manifest_hash[:12] or '-'}"
        )
    return 0


def cmd_check(args: Namespace) -> int:
    report = cast(
        dict[str, Any],
        check.run_checks(
            root=paths.REPO_ROOT, tier=args.tier or "default", fail_fast=args.fail_fast
        ),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json:
        Path(args.json).write_text(encoded, encoding="utf-8")
    else:
        for result in report["checks"]:
            print(f"[{result['status'].upper():6}] {result['id']}: {result['detail']}")
    return 0 if report["passed"] else 1
