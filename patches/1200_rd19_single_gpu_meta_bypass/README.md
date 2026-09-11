# 1200_rd19_single_gpu_meta_bypass: skip the Meta device wrapper with a single GPU (RD19)

Patch id: `1200_rd19_single_gpu_meta_bypass`. Plan item: `RD19`. A matching
Experiment Contract exists (`RD19-SINGLE-GPU-META-BYPASS`,
`config/experiment-contracts.toml`) but is deliberately NOT bound in
`patch.toml` yet -- binding it without a validation.toml wiring producers
for its required capabilities makes `build_plan_for_patch()` fail closed
with a real `ConfigurationError` (verified directly: "experiment
contract(s) ... require validation evidence but no validation.toml adapter
exists"). Author validation.toml first, then bind.

## Scope

Target architectures: gfx1100, gfx1201, gfx1030 (contract `scope.architectures`).
Backend: HIP. Prerequisites: none.

## What it does

`-s tensor` creates a Meta device wrapping the GPU backends and splits the
graph into subgraphs at every partial-split node -- even with a single
device, where no splitting is possible. Each subgraph is a separate graph
compute call, multiplying per-token launch overhead and clearing the Q8_1
quantize cache between subgraphs. This patch uses the plain device when
`n_devices == 1` in both branches of `llama_prepare_model_devices`
(explicit device list, default device selection); the Meta path is
untouched for real multi-GPU tensor parallelism. Ported verbatim from
stew675-rdna-boosts fork commit `3c48ecd63`
(https://github.com/stew675/llama.cpp); not merged into ggml-org/llama.cpp
master.

## Real evidence (already recorded, package metadata was just never synced)

This patch was already promoted via a real evidence process
(config/external-sources.toml's tracked-status entry, `status =
"ported-validated"`) that this README/patch.toml had never been updated to
reflect -- that gap is what this commit fixes, not new validation work:

- Single-GPU synthetic bench: **+8.95% decode**, byte-identical output vs.
  control.
- Real scheduler evidence: `Meta(ROCm0)` -> `ROCm0` device resolution
  confirmed.
- Real Meta-subgraph probe: `n_subgraphs=129, n_backends=1` on control vs.
  1 direct submission on treatment.
- HIP graph capture confirmed active/replaying under treatment (RD73 MTP
  workload).
- Dual-XTX Qwen3.8-27B production 3-arm bench (base/control/treatment)
  confirms a true no-op under `-sm tensor` 2-GPU -- proving the multi-GPU
  Meta path really is untouched, not just unexercised.
- `gpt-dev-agent` PROMOTE verdict recorded, session `ses_866bf44313864664`
  (2026-08-23).

## How to invoke validation

```
PYTHONPATH=tools python -m bigcherry.patch.validation_campaign \
  --patch 1200_rd19_single_gpu_meta_bypass \
  --model <a real .gguf> \
  --hip-path <production-rocm> --amdgpu-targets <target> \
  --manifest <hip-autotune-manifest.json> \
  --workdir <fresh-workdir> --worktree-root <worktree-root>
```

No validation.toml/bespoke correctness producer exists yet -- the generic
S1-S7 campaign (apply/build) is what the CLI can currently execute; the
real promotion evidence above came from a dedicated bench session outside
this harness, not from `--run-*-contract`-style orchestration the way
RD04/RD08/RD58 have. Authoring a validation.toml (backend_reference
correctness check, decode/prefill performance) so a future run can
reproduce this evidence through the standard harness is separate,
not-yet-done work.

## Known limitations

Not `deferred-hardware`. Real evidence exists but predates this project's
current standardized validation harness (PVPS02) -- reproducing it through
that harness, rather than resting on the original bench session alone, is
the honest next step before further promotion decisions build on this.

## Evidence

The compact promotion record lives in `config/external-sources.toml`'s
tracked-status note for this patch. No `evidence/validation.json` exists
yet under this patch package -- that gap (a real evidence record produced
outside `patch.toml`'s own state tracking) is real and should be closed by
a future harness-driven re-run, not backfilled retroactively here.
