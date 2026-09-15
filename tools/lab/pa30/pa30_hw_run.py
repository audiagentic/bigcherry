"""PA30: real two-arm HIP hardware build + replay-equivalence comparison
for gates 1-3, against the REAL bigcherry-native / bigcherry-serving-base
sources (gate 1), with the same run's identities separately checked to
also satisfy gate 2 (pre-cutover-release vs migrated bigcherry) and gate 3
(replay-winner re-run against the final migrated serving source) per
GPT design review (dev-gpt-agent session ses_c2892cdae7f14feb,
req_92fb5a4fe596449e): one hardware execution may satisfy G1/G2/G3 if the
resolver receipts prove G1 control identity == G2 reconstructed-old-release
identity and G1 candidate identity == current migrated bigcherry identity.

Run ON Brutus (needs real GPU + the real production build pipeline), from
inside a clean bc-pa-work checkout, with PYTHONPATH=tools and the bc-pytest
venv python, e.g.:

    cd /home/audumla/bc-pa-work && PYTHONPATH=tools \
      /home/audumla/bc-pytest-venv/bin/python \
      tools/lab/pa30/pa30_hw_run.py > /home/audumla/pa30-hw-run.log 2>&1 &

Reuses the exact same RHA15-verify2 winners corpus / model / arch / devices
/ runtime-profile PA26 used, per PA30's validation requirement to use
identical inputs across paired comparisons.

This is throwaway lab tooling for PA30's real hardware pass, not a
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
from bigcherry.campaign import resolution
from bigcherry.campaign import replay_equivalence_pa30 as pa30

WORKDIR = Path("/home/audumla/pa30-hw-workdir")
WORKDIR.mkdir(parents=True, exist_ok=True)

WINNERS_CACHE = Path(
    "/home/audumla/bc-pa-artifacts/pa26-rha15-verify2/workdir/dispatch.cache"
)
CORPUS_PRODUCER_MANIFEST_PATH = Path(
    "/home/audumla/.cache/bigcherry/artifacts-store/runs/"
    "pa26-rha15-verify2-replay-cb4187d0a830/generate/hip-autotune-manifest.json"
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

print(f"PA30 hardware run starting. workdir={WORKDIR}", flush=True)
print(f"winners_cache={WINNERS_CACHE} exists={WINNERS_CACHE.is_file()}", flush=True)
print(f"inventory={INVENTORY_PATH} exists={INVENTORY_PATH.is_file()}", flush=True)
print(f"model={MODEL_PATH} exists={MODEL_PATH.is_file()}", flush=True)
print(
    f"corpus_producer_manifest={CORPUS_PRODUCER_MANIFEST_PATH} "
    f"exists={CORPUS_PRODUCER_MANIFEST_PATH.is_file()}", flush=True,
)

# --- Pre-flight identity proofs (gate 1 control/candidate == gate 2's
# reconstructed-old-release / migrated-bigcherry identities) BEFORE any
# GPU/build time is spent. ---
delta = pa30.resolve_real_composition_delta(
    cfg, catalog, control_source="bigcherry-native", candidate_source="bigcherry-serving-base",
)
pa30.require_real_expected_composition_delta(delta)
print("gate1 composition delta OK:", delta.removed, flush=True)

pre_cutover_identity = pa30.require_pre_cutover_release_identity(cfg, catalog, native_source="bigcherry-native")
print("gate2 pre-cutover-release == bigcherry-native: proven OK", flush=True)

bc_current = resolution.resolve_canonical_selection("bigcherry", cfg, catalog)
serving_base_current = resolution.resolve_canonical_selection("bigcherry-serving-base", cfg, catalog)
g1_g2_share = (
    bc_current.identity.patch_ids == serving_base_current.identity.patch_ids
    and bc_current.identity.module_hashes == serving_base_current.identity.module_hashes
)
print(f"gate1 candidate (bigcherry-serving-base) == migrated bigcherry: {g1_g2_share}", flush=True)
if not g1_g2_share:
    print("FATAL: cannot share one hardware run across gates 1/2/3 -- compositions diverge", file=sys.stderr)
    sys.exit(3)

control_spec, candidate_spec = pa30.build_real_hardware_specs(
    platform_name=PLATFORM,
    architectures=ARCHS,
    inventory_ref=INVENTORY_PATH,
    winners_ref=WINNERS_CACHE,
    control_source="bigcherry-native",
    candidate_source="bigcherry-serving-base",
)
diagnostic_control_spec, diagnostic_candidate_spec = pa30.build_real_diagnostic_hardware_specs(
    platform_name=PLATFORM,
    architectures=ARCHS,
    inventory_ref=INVENTORY_PATH,
    winners_ref=WINNERS_CACHE,
    control_source="bigcherry-native",
)

runtime_runner = pa30.pa26_hw.make_real_hardware_runtime_runner(
    model_path=MODEL_PATH,
    devices=DEVICES,
    runtime_profile=runtime_profile,
    workdir=WORKDIR,
    winners_cache_path=WINNERS_CACHE,
)

receipt = pa30.build_real_hardware_receipt(
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
    corpus_producer_manifest_path=CORPUS_PRODUCER_MANIFEST_PATH,
    control_source="bigcherry-native",
    candidate_source="bigcherry-serving-base",
    run_id_prefix=f"pa30-hw-{int(time.time())}",
    runtime_runner=runtime_runner,
)
receipt["pa30_gate_sharing"] = {
    "g1_candidate_equals_migrated_bigcherry": g1_g2_share,
    "g2_pre_cutover_equals_bigcherry_native": True,
}

out_path = WORKDIR / "pa30-hardware-receipt.json"
out_path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
print("RECEIPT_WRITTEN", out_path, flush=True)
print(json.dumps(receipt, indent=2, default=str))
