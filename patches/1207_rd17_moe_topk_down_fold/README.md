# 1207_rd17_moe_topk_down_fold: fold the MoE topk-weights MUL into the down projection (RD17)

Patch id: `1207_rd17_moe_topk_down_fold`. Original plan item `RD17` is
superseded by `PRBE14` (docs/planning/active/patching-rdna-boost-experiments/PRBE14.md,
capability-rebaseline-v3-2026-09) -- PRBE14 is the authoritative tracking
item now. No Experiment Contract is bound yet -- this patch has no
`experiment-contract` field in `patch.toml` and no bespoke correctness
producer under `validation/`, unlike RD04/RD08/RD58/RD73. It has no
`validation.toml` at all yet (`apply`/`build` run via the generic default
adapter; there is no `activation` check wired -- see below).

**Composition, not a soft warning**: `CONFLICTS = ("1205_rd12_paired_mmvq_dual_output",)`
in `patch.py` is real and enforced (`patchset.resolve_exact()` raises if
both are explicitly selected). RD17's `common.cuh`/`ggml-cuda.cu` anchors
occupy the exact same insertion slots RD12 (1205) uses -- verified failing
identically on two different pinned revisions (not a rebase regression).
Selecting RD17 alone (as below) is unaffected by this; it only matters if a
future recipe ever tries to combine 1205 and 1207, which none does today
and PRBE14 explicitly requires composition-conflict validation (PKC02)
before that could change.

## Scope

No `validation-architectures` declared in `patch.toml` yet. Backend: HIP.

## What it does

The MoE output is normally scaled per token by the topk softmax routing
weights in a separate broadcast `MUL` kernel run after the down projection.
This folds that scale into the down-projection matmul's own epilogue: the
mmvq kernel multiplies each result row by the destination channel's
(token's) weight directly, via a new `x_scale_channel_dst` fusion flag that
indexes the scale array by `channel_dst` instead of the existing
NVFP4-only per-expert `channel_x` indexing, with the NVFP4-only scale-
fusion assertion relaxed to allow this second case. Detection pattern:
`[MUL_MAT_ID, MUL]` where the `MUL`'s second operand is a contiguous
per-channel F32 vector matching the matmul's row count. Ported from
stew675-rdna-boosts fork commit `5e545b7da` (https://github.com/stew675/llama.cpp);
not merged into ggml-org/llama.cpp master. The fork's own claim (not yet
independently verified): removes 40 kernel launches per decode token on a
qwen3.5-MoE-class model, with bit-identical perplexity.

**Real evaluation of this patch requires a MoE model** -- the detection
pattern only fires on a `MUL_MAT_ID`-based MoE down projection, which a
dense model never emits. `config/models.toml`'s
`tierM-qwen35b-a3b-moe-mtp` (Qwen3.6-35B-A3B) is this project's only
currently-registered MoE-family model and is the right choice for
activation/performance evidence here.

## How to invoke validation

No contract-bound qualification path or validation.toml exists yet --
only the generic default S1-S2 campaign (apply/build) is currently
invocable, and even that has no activation check wired to confirm the
fusion path actually triggered (this patch has no
`BIGCHERRY_PATCH_TRACE`-gated log marker in its `ggml_cuda_try_fuse`
detection block, unlike RD12's -- a real, separate authoring gap from the
missing README this commit fixes):

```
PYTHONPATH=tools python -m bigcherry.patch.validation_campaign \
  --patch 1207_rd17_moe_topk_down_fold \
  --model <tierM-qwen35b-a3b-moe-mtp.gguf> \
  --hip-path <production-rocm> --amdgpu-targets <target> \
  --manifest <hip-autotune-manifest.json> \
  --workdir <fresh-workdir> --worktree-root <worktree-root>
```

Bringing this patch to RD08's level of qualification (an activation trace
marker, a real correctness producer proving the fork's bit-identical-PPL
claim, a bound Experiment Contract, and real performance evidence on the
required MoE model) is separate, not-yet-done authoring work tracked under
PRBE14.

## Known limitations

Not `deferred-hardware`. No real fresh evidence exists yet for this
project's own hardware -- everything under "What it does" attributed to
"the fork" is the fork's own reported claim, not this project's
independently-measured result. No activation proof mechanism exists yet
(see above), so even a clean build+run does not yet prove the fusion path
was actually exercised rather than silently falling through unfused.

## Evidence

None yet. Runtime artifacts, once a real campaign runs, land under
`artifacts/patch-validation/1207_rd17_moe_topk_down_fold/<campaign-identity>/`.
