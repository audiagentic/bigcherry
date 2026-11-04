#!/usr/bin/env python3
"""BU01: fetch and verify a stock Radiance runtime bundle from its release."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from bigcherry.core.context import ProjectContext
from bigcherry.release import runtime_assets


def fetch(*, tag: str | None) -> Path:
    changed = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no", "--",
         "engines/radiance", "config"], check=True, capture_output=True, text=True)
    if changed.stdout.strip():
        raise runtime_assets.BundleError(
            "engine/config tree has local edits; branch build must compile, not reuse a release binary")
    recipes = tomllib.loads((ROOT / "engines/radiance/recipes.toml").read_text())
    pin = recipes["pinned"]
    record = json.loads((ROOT / "releases" / f"{pin}.json").read_text())
    sdk = os.environ.get("ROCM_PATH") or os.environ.get("BC_ROCM_ROOT")
    if not sdk:
        raise runtime_assets.BundleError("ROCM_PATH must identify the configured host ROCm SDK")
    work = ProjectContext.resolve(engine="radiance", project_root=ROOT).work_root
    return runtime_assets.fetch(record, ROOT, work, Path(sdk), tag=tag)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="specific bc-radiance-<version> release; default is latest recorded")
    args = parser.parse_args()
    try:
        path = fetch(tag=args.tag)
    except (runtime_assets.BundleError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"BU01: release bundle unavailable/unverified: {exc}", file=sys.stderr)
        return 1
    print(f"BU01: verified stock release bundle installed: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
