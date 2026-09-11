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

Not `deferred-hardware`. Real evidence now exists both from the original
2026-08-23 bench session and a fresh 2026-09-11 native-llama.cpp 3-arm
re-sweep (below) -- neither yet runs through PVPS02's standardized harness
end-to-end (both were direct `llama-bench`/manual sessions); wiring a
`validation.toml` producer that reproduces this same 3-arm comparison
through the standard CLI remains separate, not-yet-done work.

**Gap found 2026-09-11 (process audit), closed same day (real re-sweep)**:
this patch is tagged `optimization` and carries `state = "validated"`, but
its historical core performance claim (+8.95% decode) was measured only as
a BigCherry-internal control-vs-treatment A/B -- never against unmodified
native llama.cpp. Per this project's new rule
(`docs/reference/patches/PATCH_AUTHORING.md`'s "`optimization` carries a
real validation obligation"), re-ran a genuine 3-arm sweep same day:

**Real 3-arm sweep (2026-09-11, Brutus, gfx1100, single GPU
`HIP_VISIBLE_DEVICES=0`, `-sm tensor` -- the exact condition this patch's
mechanism targets, since without it no Meta wrapper is created for a
single device and the patch has nothing to bypass)**: `llama-bench`,
`tierA-qwen4b-q6k` (Qwen3.5-4B-UD-Q6_K_XL), decode only (`-n 128 -p 0`),
`-r 5`, isolated scratch clone at pin `28ff0958291ce3465fabd7bd679d4b0edd742bd9`
(controller commit `f91db8ce`), native/BC-baseline/BC+patch each built from
the same isolated materialization (`bigcherry.patch.source.resolve_source_composition`
/ `materialize_composition`, bypassing the shared tree entirely):

| arm | tg128 (t/s) | stddev |
|---|---:|---:|
| native llama.cpp (unmodified) | 103.02 | 0.17 |
| BigCherry baseline (RD19 excluded) | 102.41 | 1.55 |
| **BigCherry + RD19** | **109.18** | **0.60** |

BigCherry's baseline matches native llama.cpp within noise (102.41 vs
103.02 -- BC's framework overhead is not costing anything here). RD19
delivers a real, substantial gain clearly outside either baseline's noise
band: **+6.6% over BC-baseline, +5.9% over native llama.cpp** -- confirms
the historical +8.95% claim's direction and rough magnitude (the modest
difference is plausibly explained by model/quantization/exact config
differences between this re-sweep and the original 2026-08-23 session,
neither of which is a red flag). A same-single-GPU run WITHOUT `-sm tensor`
(i.e. `-sm none`, the default) shows RD19 within noise of baseline
(108.65-109.24 t/s across arms) -- as expected, since without `-sm tensor`
no Meta wrapper is ever created and the patch's bypass has nothing to
short-circuit; this confirms the effect is real and specifically tied to
the `-sm tensor` + single-device condition the patch's own docstring
describes, not a general fluke.

## Evidence

The compact promotion record lives in `config/external-sources.toml`'s
tracked-status note for this patch. No `evidence/validation.json` exists
yet under this patch package -- that gap (a real evidence record produced
outside `patch.toml`'s own state tracking) is real and should be closed by
a future harness-driven re-run, not backfilled retroactively here.
