"""CLI for deterministic per-source patch footprint reports."""

from __future__ import annotations

import sys
from pathlib import Path

from ..analysis import patch_footprint
from ..core import paths


def cmd_patch_footprint(args) -> int:
    directory = (
        Path(args.out_dir)
        if args.out_dir is not None
        else paths.REPO_ROOT / "docs" / "evidence" / f"{args.engine}-patch-footprint"
    )
    try:
        report = patch_footprint.collect(args.engine)
        changed = patch_footprint.publish(report, directory, check=args.check)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"patch-footprint: {exc}", file=sys.stderr)
        return 1
    if args.check and changed:
        print("patch-footprint: stale/missing generated reports:\n  " + "\n  ".join(changed), file=sys.stderr)
        return 1
    print(
        f"patch-footprint: {'current' if args.check else 'generated'} "
        f"{report['totals']['composed_patches']} composed patches, "
        f"{report['totals']['anchors']} anchor edits, "
        f"{report['totals']['upstream_files']} upstream files in {directory}"
    )
    return 0
