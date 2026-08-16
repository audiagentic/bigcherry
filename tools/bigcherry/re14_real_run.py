"""RE14: CLI wrapper around campaign_lane.execute_campaign_lane() for one lane.

Not a permanent CLI command in its own right -- ``campaign-build`` (see
__main__.py) is the registered entrypoint; this module is what it and
re14_parity_run.py both invoke as a subprocess. All the actual orchestration
(materialize -> generate -> build -> runtime-smoke) lives in campaign_lane.py
(RE16) -- this file only parses args, builds a spec, calls the library
function, and renders the result. The legacy ``cmd_build`` path is
completely untouched; this writes only into ``context.work_root`` and a
dedicated ArtifactStore directory.

Usage:
    python -m bigcherry.re14_real_run \
        --upstream-repo /mnt/vault/development/bc-branch/vendor/llama.cpp \
        --inventory artifacts/campaign-gfx1100-inventory.json \
        --arch gfx1100 \
        --model /mnt/vault/llm-models/qwen3.5-0.8B/gguf/Qwen3.5-0.8B-UD-Q5_K_XL.gguf \
        --hip-visible-devices 0
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from . import config
from .artifacts import ArtifactStore
from .campaign_lane import (CampaignLaneError, CampaignLaneExecutionSpec,
                            execute_campaign_lane, smoke_environment_for_hip_devices)
from .context import ProjectContext
from .runtime_smoke import RuntimeSmokeSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--arch", required=True, help="comma-separated, e.g. gfx1100")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", default="bigcherry")
    parser.add_argument("--build", default="tune")
    parser.add_argument("--platform", default="linux-multi")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--hip-visible-devices", default="0")
    parser.add_argument("--split-mode", default="none")
    parser.add_argument("--binary-relative-path", default="bin/llama-bench")
    parser.add_argument(
        "--c-compiler", default=None,
        help="override platform.c_compiler -- e.g. to build against a "
             "specific ROCm install for a version comparison")
    parser.add_argument("--cxx-compiler", default=None)
    args = parser.parse_args(argv)

    run_id = args.run_id or uuid.uuid4().hex[:12]
    print(f"=== RE14 real run {run_id} ===")

    context = ProjectContext.resolve(work_root=args.work_root, upstream_repo=args.upstream_repo)
    cfg = config.load(context.config_path)
    store = ArtifactStore(context.work_root / "artifacts-store")

    spec = CampaignLaneExecutionSpec(
        source_name=args.source, build_name=args.build, platform_name=args.platform,
        architectures=tuple(args.arch.split(",")), inventory_path=args.inventory,
        validation=RuntimeSmokeSpec(model_path=args.model, split_mode=args.split_mode),
        binary_relative_path=args.binary_relative_path,
        c_compiler=args.c_compiler, cxx_compiler=args.cxx_compiler,
        smoke_environment=smoke_environment_for_hip_devices(args.hip_visible_devices),
    )

    try:
        result = execute_campaign_lane(spec, cfg=cfg, context=context, store=store, run_id=run_id)
    except CampaignLaneError as exc:
        print(f"BUILD CAMPAIGN FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"materialized: source_slice_id={result.source_slice_id}")
    print(f"build plan: id={result.build_plan_id} variant_set={result.build_plan.variant_set}")
    print(f"generated: workload_id={result.workload_id}")
    print(f"built: {result.binary_ref.kind} artifact published")
    smoke_result = json.loads(result.smoke_ref.path.read_text(encoding="utf-8"))
    print(f"smoke: {json.dumps(smoke_result, indent=2)}")

    # A machine-readable summary for tooling that wants to load this run's
    # artifacts back out (RE14's parity harness, re14_parity_run.py) without
    # re-deriving the store-relative paths itself.
    print("RE14_PARITY_RESULT_JSON: " + json.dumps({
        "run_id": result.run_id,
        "source_slice_id": result.source_slice_id,
        "build_plan_id": result.build_plan_id,
        "workload_id": result.workload_id,
        "store_root": str(store.root),
        "manifest_relative": str(result.manifest_ref.path.relative_to(store.root).as_posix()),
        "manifest_content_hash": result.manifest_ref.content_hash,
        "binary_relative": str(result.binary_ref.path.relative_to(store.root).as_posix()),
    }))

    print(f"=== RE14 real run {run_id}: ALL STAGES PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
