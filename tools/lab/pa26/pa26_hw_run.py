"""PA26: real two-arm HIP hardware build + replay-equivalence comparison.

Run ON Brutus (needs real GPU + the real production build pipeline), from
inside a clean bc-pa-work checkout, with PYTHONPATH=tools and the bc-pytest
venv python, e.g.:

    cd /home/audumla/bc-pa-work && PYTHONPATH=tools \
      /home/audumla/bc-pytest-venv/bin/python \
      tools/lab/pa26/pa26_hw_run.py > /home/audumla/pa26-hw-run.log 2>&1 &

Builds the PRODUCTION pair (bigcherry-native vs PA27's ephemeral
serving-core composition) and the DIAGNOSTIC companion pair (same, +0810),
drives the diagnostic pair with the real winners corpus via
make_real_hardware_runtime_runner, and prints the full receipt as JSON.

This is throwaway lab tooling for PA26's real hardware pass, not a
permanent CLI surface -- kept in tools/lab per this project's convention
of never leaving ad-hoc scripts in /tmp on Brutus.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from bigcherry.core import config as campaign_config
from bigcherry.core.artifacts import ArtifactStore
from bigcherry.core.context import ProjectContext
from bigcherry.patch import patchset
from bigcherry.campaign import replay_equivalence_hardware as hw

WORKDIR = Path("/home/audumla/pa26-hw-workdir")
WORKDIR.mkdir(parents=True, exist_ok=True)

WINNERS_CACHE = Path(
    "/home/audumla/bc-pa-artifacts/pa26-rha15-verify2/workdir/dispatch.cache"
)
INVENTORY_PATH = Path(
    "/home/audumla/bc-pa-artifacts/pa26-rha15-verify2/workdir/inventory.json"
)
MODEL_PATH = Path(
    "/mnt/vault/llm-models/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf"
)
DEVICES = "0"
PLATFORM = "linux-multi"
ARCHS = ("gfx1100",)
BIGCHERRY_REVISION = "28ff0958291ce3465fabd7bd679d4b0edd742bd9"  # current pin b10901

context = ProjectContext.resolve(work_root=None, upstream_repo=None)
cfg = campaign_config.load(context.config_path)
catalog = patchset.catalog()
store = ArtifactStore(context.work_root / "artifacts-store")

if "production-safe-single" not in cfg.runtime_profiles:
    print(f"no production-safe-single runtime profile: {sorted(cfg.runtime_profiles)}", file=sys.stderr)
    sys.exit(2)
runtime_profile = cfg.runtime_profiles["production-safe-single"]

print(f"PA26 hardware run starting. workdir={WORKDIR}", flush=True)
print(f"winners_cache={WINNERS_CACHE} exists={WINNERS_CACHE.is_file()}", flush=True)
print(f"inventory={INVENTORY_PATH} exists={INVENTORY_PATH.is_file()}", flush=True)
print(f"model={MODEL_PATH} exists={MODEL_PATH.is_file()}", flush=True)

control_spec, candidate_spec = hw.build_hardware_specs(
    platform_name=PLATFORM,
    architectures=ARCHS,
    inventory_ref=INVENTORY_PATH,
    winners_ref=WINNERS_CACHE,
)
diagnostic_control_spec, diagnostic_candidate_spec = hw.build_diagnostic_hardware_specs(
    platform_name=PLATFORM,
    architectures=ARCHS,
    inventory_ref=INVENTORY_PATH,
    winners_ref=WINNERS_CACHE,
)

runtime_runner = hw.make_real_hardware_runtime_runner(
    model_path=MODEL_PATH,
    devices=DEVICES,
    runtime_profile=runtime_profile,
    workdir=WORKDIR,
    winners_cache_path=WINNERS_CACHE,
)

receipt = hw.build_hardware_receipt(
    cfg,
    catalog,
    WINNERS_CACHE,
    bigcherry_revision=BIGCHERRY_REVISION,
    context=context,
    store=store,
    control_spec=control_spec,
    candidate_spec=candidate_spec,
    diagnostic_control_spec=diagnostic_control_spec,
    diagnostic_candidate_spec=diagnostic_candidate_spec,
    # A prior pass already wrote content under the fixed "pa26-hw1" run-scoped
    # artifact-store path with different bytes (immutable-store collision) --
    # unique per-invocation prefix avoids colliding with stale prior-run
    # output rather than deleting shared ~/.cache/bigcherry state.
    run_id_prefix=f"pa26-hw-{int(time.time())}",
    runtime_runner=runtime_runner,
)

out_path = WORKDIR / "pa26-hardware-receipt.json"
out_path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
print("RECEIPT_WRITTEN", out_path, flush=True)
print(json.dumps(receipt, indent=2, default=str))
