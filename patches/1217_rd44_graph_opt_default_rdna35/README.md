# 1217_rd44_graph_opt_default_rdna35: default GGML_CUDA_GRAPH_OPT to enabled on RDNA3.5 (RD44)

Patch id: `1217_rd44_graph_opt_default_rdna35`. Plan item: `RD44`. A
matching Experiment Contract exists (`RD44-GRAPH-OPT-DEFAULT-RDNA35`,
`config/experiment-contracts.toml`) but is deliberately NOT bound in
`patch.toml` yet -- binding it without a validation.toml wiring producers
for its required capabilities makes `build_plan_for_patch()` fail closed
with a real `ConfigurationError`. Author validation.toml first, then bind.
Last patch in the AMD-STREAM chain -- `requires = ["1215_rd394041_amd_stream_moe_overlap",
"1216_rd43_concurrent_join_fusion_guard"]` (both prerequisites are already
materialized patches in this repo).

## Scope

Target architecture for the real performance claim: gfx1151 only (contract
`scope.architectures`). Behavior-neutral on gfx1100/gfx1201/gfx1030 (the
RDNA3.5-gated branch this patch touches cannot resolve on those
architectures) -- so only the no-regression proof is currently runnable on
this project's own hardware. Backend: HIP.

## What it does

A policy/default flip: the graph-optimization pass (previously opt-in via
an environment variable) is defaulted to enabled specifically on gfx1151
(RDNA3.5); every other architecture keeps its prior behavior unchanged.
Ported from an amd-ecosystem-llama-cpp fork, PR #56, commit `6e9f948a0`.
Upstream-fork measured (gfx1151, Qwen3.6-35B-A3B UD-Q4_K_M): tg128 +7.3%,
VLM decode +8.7% -- the fork's own number, not yet independently
reproduced by this project (no gfx1151 hardware available to do so).

## Deferred hardware, genuinely

**`deferred-hardware`** (config/external-sources.toml's own tracked
status) is the correct disposition here, not a gap to route around: this
project's hardware (Brutus: gfx1100/gfx1201/gfx1030) cannot resolve the
RDNA3.5-gated branch this patch changes at all, so there is no way to
measure the real performance claim on hardware this project actually has.
What IS runnable now: a real no-regression proof on gfx1100/gfx1201/gfx1030
-- confirming the patch is genuinely behavior-neutral there, not silently
touching something it claims not to.

## How to invoke validation

```
PYTHONPATH=tools python -m bigcherry.patch.validation_campaign \
  --patch 1217_rd44_graph_opt_default_rdna35 \
  --model <a real .gguf> \
  --hip-path <production-rocm> --amdgpu-targets gfx1100 \
  --manifest <hip-autotune-manifest.json> \
  --workdir <fresh-workdir> --worktree-root <worktree-root>
```

No validation.toml/bespoke correctness producer exists yet -- only the
generic S1-S7 campaign (apply/build) is currently invocable. Authoring a
validation.toml (backend_reference correctness check on gfx1100/gfx1201/
gfx1030, proving behavior-neutrality) is separate, not-yet-done work; the
real gfx1151 performance claim cannot be qualified by this project until
that hardware becomes available.

## Known limitations

Genuinely `deferred-hardware` for the performance claim (see above) --
this is not a workaround-able gap, it requires gfx1151 hardware this
project does not have.

## Evidence

None yet. Runtime artifacts, once a real campaign runs, land under
`artifacts/patch-validation/1217_rd44_graph_opt_default_rdna35/<campaign-identity>/`.
