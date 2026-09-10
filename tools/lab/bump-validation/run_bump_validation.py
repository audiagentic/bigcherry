"""Reusable post-pin-bump smoke matrix (PIN_BUMP.md step 5/6).

Runs ONE real production build (via the same campaign planner/runner
`bigcherry build` uses), then launches it through `runtime-matrix` as:
  - one lightweight cross-architecture model (tierB-qwen9b-q6k) on every
    real GPU individually (single-card smoke), plus
  - the real dual-XTX production runtime-profile (production-dual-xtx)
    with the real production model (tierL-qwen27b-q8) across both cards.

Not a benchmark -- a real-server-launches-and-completes-correctly gate,
run on the build server against real hardware after a pin bump, so a
bump that silently broke dispatch/build/runtime (as opposed to a patch
anchor mismatch, which patch-rebase-check already catches during the
bump itself) is caught before the bump is declared complete.

build_id/binary are NOT hand-writable: they are content-hash identities
that only exist once a real build has run, so this script builds first
and fills them in itself -- there is no static matrix config to maintain.

Usage (on the build server, real GPUs required):
    python3 tools/lab/bump-validation/run_bump_validation.py \\
        --output work/bump-validation/<run-id> [--dry-run]

`--dry-run` resolves and writes the immutable matrix (build still runs
for real -- there is no fake build_id/binary to substitute) without
launching any server.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BC_TOOLS = _REPO_ROOT / "tools"
if str(_BC_TOOLS) not in sys.path:
    sys.path.insert(0, str(_BC_TOOLS))

from bigcherry.campaign.planner import CampaignRequest, plan, run_campaign
from bigcherry.core import config as campaign_config
from bigcherry.core.artifacts import ArtifactStore
from bigcherry.core.context import ProjectContext

SMOKE_MODEL = "tierB-qwen9b-q6k"
PRODUCTION_MODEL = "tierL-qwen27b-q8"
SINGLE_GPU_DEVICES = (0, 1, 2, 3)


def _build_once(context: ProjectContext, store: ArtifactStore, run_id: str):
    cfg = campaign_config.load(context.config_path)
    # Inlined rather than calling a campaign.lane helper: RE15's real-
    # hardware HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES finding (only set
    # HIP_VISIBLE_DEVICES, never both) still applies to whichever build
    # this runs against -- only HIP_VISIBLE_DEVICES=0 matters for a build
    # step (the smoke cells below each set their own real device list).
    smoke_environment = tuple(sorted({
        "HIP_VISIBLE_DEVICES": "0",
        "PATH": os.environ.get("PATH", ""),
    }.items()))
    request = CampaignRequest(
        profile_name="standard",
        binary_relative_path="bin/llama-server",
        smoke_environment=smoke_environment,
    )
    lanes = plan(request, cfg)
    results = run_campaign(lanes, cfg=cfg, context=context, store=store, run_id=run_id)
    failed = {lid: r for lid, r in results.items() if isinstance(r, Exception)}
    if failed:
        for lid, exc in failed.items():
            print(f"bump-validation: build lane {lid} FAILED -- {exc}", file=sys.stderr)
        raise SystemExit(1)
    # The 'standard' profile's first/only production lane's binary is what
    # every smoke cell below launches -- same binary, different devices/model.
    lane_id = sorted(results)[0]
    result = results[lane_id]
    return result.build_plan_id, str(store.resolve(result.binary_ref.path))


def build_cells(build_id: str, binary: str) -> list[dict]:
    cells = []
    for device in SINGLE_GPU_DEVICES:
        cells.append({
            "cell_id": f"bump-validate-gpu{device}-native-smoke",
            "model_id": SMOKE_MODEL,
            "devices": [device],
            "topology": "single",
            "runtime_profile": "production-safe-single",
            "arm": "native",
            "build_id": build_id,
            "binary": binary,
        })
    cells.append({
        "cell_id": "bump-validate-dual-xtx-mtp-smoke",
        "model_id": PRODUCTION_MODEL,
        "devices": [0, 1],
        "topology": "dual-xtx",
        "runtime_profile": "production-dual-xtx",
        "arm": "native",
        "build_id": build_id,
        "binary": binary,
    })
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="directory for resolved cells/status/events")
    parser.add_argument("--llama-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true", help="resolve and write the matrix without launching")
    args = parser.parse_args()

    context = ProjectContext.resolve(
        work_root=None, upstream_repo=Path(args.llama_root) if args.llama_root else None
    )
    store = ArtifactStore(context.work_root / "artifacts-store")
    run_id = args.run_id or "bump-validation"

    build_id, binary = _build_once(context, store, run_id)
    print(f"bump-validation: built build_id={build_id} binary={binary}")

    cells = build_cells(build_id, binary)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / "matrix.json"
    matrix_path.write_text(json.dumps({"schema_version": 1, "host": "build-server", "cells": cells}, indent=2) + "\n")
    print(f"bump-validation: wrote {matrix_path}")

    # Delegate resolution/execution to the real `bigcherry runtime-matrix`
    # CLI rather than re-implementing its resolution/launch logic here.
    cmd = [
        sys.executable, "-m", "bigcherry", "runtime-matrix",
        "--config", str(matrix_path), "--output", str(output),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = str(_BC_TOOLS)
    return subprocess.call(cmd, cwd=str(_REPO_ROOT), env=full_env)


if __name__ == "__main__":
    raise SystemExit(main())
