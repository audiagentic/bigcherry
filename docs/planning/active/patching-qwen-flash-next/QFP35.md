---
id: QFP35
order: 35
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:37.142989+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Hyper-connection kernels: no 64-bit div/mod, and audit our fused chains against the reported ones

## Description

External report: (a) replacing emulated 64-bit integer division per element with a dimensional launch in hyper-connection kernels reduced one reported class from 230 to 79 us; (b) fusing scale/sigmoid/scale/hc_post and scale/silu chains removed hundreds of launches per token. Those measurements are from different hardware and are hypotheses only.

Both residuals exist at b11402, but part of the fusion work is already covered by BigCherry. The upstream HIP `dsv4_hc_pre_f32` and `dsv4_hc_post_f32` flatten their logical dimensions and recover coordinates with runtime `int64_t / %`. BigCherry 1311's Q8_1-producing pre kernel repeats the same pattern. Separately, validated 1313 already fuses `SCALE -> SILU` and `SCALE -> SIGMOID -> SCALE`; it does **not** consume the following `DSV4_HC_POST`, so that final launch remains a concrete audit candidate.

Treat indexing and the remaining fusion as independent qualification changes.

## What we already have

### b11402 hyper-connection graph

- `src/models/qwen4exp.cpp::llama_model_qwen4exp::graph::build_hc_mix`
  - computes `lo = silu(scale(w_down @ xn, 1/hc))`;
  - computes `gate = w_up @ lo`;
  - with fused HC enabled, calls `ggml_dsv4_hc_pre_gated()`.
- `src/models/qwen4exp.cpp::llama_model_qwen4exp::graph::build_hc_combine`
  - computes `w = 2 * sigmoid(scale(inject, 1/hc))`;
  - with fused HC enabled, passes `w` to `ggml_dsv4_hc_post()`.
- `ggml/src/ggml-cuda/dsv4-hc.cu::dsv4_hc_pre_f32`
  - flat index `ir` spans `n_embd*n_tokens`;
  - executes `i0 = ir % n_embd` and `it = ir / n_embd` using runtime 64-bit dimensions.
- `ggml/src/ggml-cuda/dsv4-hc.cu::dsv4_hc_post_f32`
  - flat index spans `n_embd*hc*n_tokens`;
  - executes `i0 = ir % n_embd`, `idst = (ir/n_embd) % hc`, and `it = ir/(n_embd*hc)`.
- `ggml/src/ggml-cuda/dsv4-hc.cu::ggml_cuda_op_dsv4_hc_pre/post`
  - currently launch those flat kernels with a one-dimensional grid.
- `ggml/src/ggml-backend-meta.cpp`
  - handles DSV4 HC COMB/PRE/POST through `handle_generic(..., scalar_only=true)`; no HC-specific topology assumption exists.

### BigCherry overlap

- `patches/1311_hc_pre_q81` is validated and production-enabled.
  - Its `dsv4_hc_pre_q81_f32` combines HC pre-mix with native Q8_1 production for the following MMVQ.
  - It also flattens `n_embd_padded*n_tokens` and computes `it = ip/n_embd_padded`, `i0 = ip%n_embd_padded`.
  - Because 1311 is the production pre path for eligible decode shapes, changing only upstream `dsv4_hc_pre_f32` would leave the important production pre case untouched.
- `patches/1313_scale_act_fuse` is validated and production-enabled.
  - It fuses F32 contiguous `SCALE -> UNARY(SILU|SIGMOID) [-> SCALE]`.
  - It already targets both Qwen4Exp HC chains and preserves the 1310 Q8_1 publication path.
  - Hardware validation reduced elementwise launches and produced a production ABBA win.
- Upstream unary-mul fusion/BigCherry 1312 owns `UNARY -> MUL`; QFP35 must not duplicate it.

Finding: the external scale/activation fusion is **mostly covered by 1313**. Do not create another scale-act package. The only graph-chain extension worth qualifying is `SCALE -> SIGMOID -> SCALE -> DSV4_HC_POST`. The 64-bit indexing issue is not covered and affects both upstream HC kernels and 1311's production pre variant.

## Steps

1. Gate 0A: disassemble/profile `dsv4_hc_pre_f32`, `dsv4_hc_pre_q81_f32`, and `dsv4_hc_post_f32` on production decode/prefill to prove integer divide/remainder instructions and their time share.
2. Qualify a dimensional-index version of PRE/POST with identical memory addressing.
3. Gate 0B: kernel census with 1313 enabled; count remaining HC `scale/sigmoid/scale` and HC_POST launches. If 1313 already removes all relevant launch overhead except HC_POST, qualify only the four-node HC_POST extension.
4. Keep the indexing and fusion toggles independently switchable for ABBA/bisect.
5. Close either sub-change if its isolated timing is not material.

## Detailed Solution & Technical Design

### A. Dimensional HC indexing

New base-kernel mechanism if Gate 0A is positive:

`patches/1347_hc_dimensional_index`

Qualification flag:

`BIGCHERRY_HC_3D_INDEX=0|1`, default `0`.

#### PRE

Replace flat `ir` coordinate recovery with:

- `grid.x = ceil(n_embd/block_size)`;
- `grid.y = n_tokens`;
- thread coordinate `i0 = blockIdx.x*blockDim.x + threadIdx.x`;
- `it = blockIdx.y`.

Bounds check only `i0 >= n_embd`. The weighted HC reduction and all strides remain unchanged.

#### POST

Use:

- `grid.x = ceil(n_embd/block_size)`;
- `grid.y = hc`;
- `grid.z = n_tokens`;
- `i0 = blockIdx.x*blockDim.x + threadIdx.x`;
- `idst = blockIdx.y`;
- `it = blockIdx.z`.

This removes all variable 64-bit div/mod from coordinate recovery. It does not change tensor layout or arithmetic.

CUDA/HIP grid dimension limits must be checked from runtime dimensions before dispatch; if a dimension cannot be represented safely, fall back to the original flat kernel. Do not truncate `int64_t` model dimensions to an unchecked 32-bit grid coordinate.

#### 1311 Q8_1 PRE

This is an improvement to an existing patch and therefore belongs **inside `patches/1311_hc_pre_q81`**, not a duplicate package. Under the same qualification concept, launch its padded PRE as:

- x over `n_embd_padded`;
- y over `n_tokens`;
- derive `ip = it*n_embd_padded + i0` only when needed for the output/Q8_1 offset;
- derive Q8_1 lane/block from power-of-two `QK8_1` and padded-row layout, avoiding variable division by `n_embd_padded`.

Keep 1311's Q8_1 cache key/layout exactly unchanged. The indexing change must not alter cache generation, row padding, zero padding or quantization math.

Qualification can use `BIGCHERRY_HC_3D_INDEX` in both the base package and 1311 so PRE/POST can be enabled as one experiment; 1311 remains owned by its existing `BIGCHERRY_HC_Q81` mechanism.

### B. Fuse HC gate production into HC_POST

Do **not** create a new package. Extend `patches/1313_scale_act_fuse`, because it already owns the exact scale/sigmoid/scale chain.

Add a default-off qualification subflag:

`BIGCHERRY_HC_POST_GATE_FUSE=0|1`, default `0`.

In `ggml_cuda_try_fuse`, after the existing 1313 three-node matcher, recognize only the exact chain:

`SCALE(inject) -> UNARY(SIGMOID) -> SCALE -> DSV4_HC_POST`

with:

- F32 contiguous scale/activation tensors;
- each intermediate single-use as required by `ggml_can_fuse`;
- final scale tensor is exactly `hc_post->src[2]`;
- HC_POST input/residual/comb contracts already accepted by the CUDA DSV4 implementation;
- no additional user of the final weight tensor.

The fused HC_POST kernel receives the original injection tensor plus both scale/bias parameter pairs and computes, per `(idst,it)`, exactly:

`post = s1 * sigmoid(s0 * inject + b0) + b1`

before the existing `x * post + residual[/comb]` calculation. Do not materialize the post-weight tensor.

This should consume four graph nodes in one CUDA/HIP launch. Preserve 1313's existing standalone three-node fusion for all non-HC_POST consumers.

### Architectures

- gfx1100 and gfx1201: primary target for target-model HC PRE/POST.
- gfx1030: explicitly relevant because the Qwen4Exp MTP sidecar executes the same HC graph; qualify dimensional indexing and the HC_POST fusion there too.
- No device-ordinal checks and no assumption that the draft remains single-GPU.

These are ordinary scalar F32 kernels; architecture-specific code is unnecessary unless profiling proves a launch/block-size crossover. Start with the existing block size.

### Meta split and fusion interaction

Meta's HC operations remain generic/scalar. The CUDA/HIP kernel receives the local physical tensor and derives its grid from those local dimensions, so arbitrary tensor/attention splits remain valid.

The HC_POST graph fusion occurs after graph construction and must not cross Meta-device boundaries. Only fuse when all four nodes are present in the same per-backend graph and `ggml_cuda_can_fuse` proves the chain. Do not alter collective placement or delayed-branch semantics.

1311's Q8_1 PRE publication remains compatible with 1313 because they operate at different HC sites.

## Code Samples & Guidance

Likely indexing anchors:

- `ggml/src/ggml-cuda/dsv4-hc.cu::dsv4_hc_pre_f32` at:
  `const int64_t ir = ...`, `i0 = ir % n_embd`, `it = ir / n_embd`.
- `dsv4_hc_post_f32` at its flat `ir/nr` block.
- `ggml_cuda_op_dsv4_hc_pre/post` at construction of `grid_dims`.

1311 anchor:

- `patches/1311_hc_pre_q81/patch.py` raw C++ kernel `dsv4_hc_pre_q81_f32` and `qgrid`.

1313 anchors:

- existing matcher comment `bigcherry 1313: SCALE -> UNARY(SILU|SIGMOID) [-> SCALE]`;
- existing `bc_scale_act_fused` ownership;
- CUDA HC POST implementation in `dsv4-hc.cu` for the specialized fused entry point.

Activation evidence:

- `BIGCHERRY_PATCH_HIT patch=1347_hc_dimensional_index op=pre|post ...`;
- extend 1311 marker with `index=3d` during qualification;
- a distinct 1313 marker suffix such as `hc_post=1` for the four-node fusion.

Never log every HC invocation.

## Files

If Gate 0A is positive:

- create `patches/1347_hc_dimensional_index/{patch.py,README.md,SUMMARY.md}`;
- add `tools/tests/patch/test_1347_hc_dimensional_index.py`;
- production target: `ggml/src/ggml-cuda/dsv4-hc.cu`;
- update **existing** `patches/1311_hc_pre_q81/patch.py` and its test/docs for the Q8_1 PRE variant.

If Gate 0B is positive:

- update **existing** `patches/1313_scale_act_fuse/patch.py`, README/SUMMARY and its existing test;
- production targets remain its fusion matcher files plus `ggml/src/ggml-cuda/dsv4-hc.cu` if the specialized HC_POST entry point is added.

Do not create a second scale-act fusion package.

## Validation

### Indexing offline

- Patch-lint and exact/idempotent mechanics.
- Direct PRE/POST op tests over token counts 1, 2, 8, 16 and a prefill width.
- Compare flat vs dimensional outputs bit-for-bit: indexing changes only coordinates, not arithmetic order.
- 1311 Q8_1 path: F32 output and Q8_1 bytes/cache key must be identical.
- Exercise nontrivial strides and both HC_POST `comb == nullptr` / non-null variants.

### Fusion offline/equivalence

- Synthetic exact four-node chain plus near-misses: shared intermediate, noncontiguous tensor, wrong src[2], unsupported type.
- Existing 1313 two/three-node tests remain unchanged.
- Prefer bit-identical output because operation order can be preserved; if HIP compiler contraction changes a last bit, document explicit backend tolerance and require greedy identity.

### Activation/timing

- Kernel census must show dimensional PRE/POST launches replacing only their opted-in counterparts.
- Disassembly or instruction metrics prove variable 64-bit divide/remainder removed.
- Four-node fusion census should remove one HC_POST-adjacent materialization launch chain, not merely rename kernels.
- Record per-op GPU time on gfx1100/gfx1201 and MTP gfx1030.

### Hardware ABBA

Use `tools/lab/flash-next/queue-env-ab.sh` with complete process separation and test each mechanism independently before combined B:

1. `BIGCHERRY_HC_3D_INDEX=0/1`, fusion held fixed.
2. `BIGCHERRY_HC_POST_GATE_FUSE=0/1`, indexing held fixed.
3. Combined only if both isolated changes win.

Run decode at ~8K and ~24K; include ~98K for target decode and a short prefill check. Record kernels/token, HC kernel time, total ms/token, MTP acceptance/depth and greedy target identity.

No multi-session contract campaign.

## Effort & Risk

Effort: M.

Risk:
- dimensional indexing: low correctness / low-medium integration;
- HC_POST four-node fusion: medium because it changes graph consumption/lifetime and must compose with Meta and 1313.

Expected gain on our topology: medium. 1313 already captured much of the external launch-count opportunity, so do not assume the reported full gain remains. The indexing change is attractive because the defect is visible in actual pin code and affects all target HC layers plus the gfx1030 MTP graph.

## Standards

- Preserve arithmetic and tensor layout for indexing.
- Existing fusion owner 1313 is extended in-place.
- Existing Q8_1 PRE owner 1311 is extended in-place.
- No fixed topology or device ordinal.
- No q4 KV.
- No compatibility shim.
- Default off for new qualification switches.
- External timings are hypotheses.

## Acceptance Criteria

- Profile/disassembly proves the current HC kernels execute costly variable 64-bit div/mod, or the indexing sub-item closes.
- Dimensional PRE/POST and 1311 PRE are bit-identical to flat indexing.
- Four-node fusion activates only on the exact single-use HC_POST chain.
- Existing 1311 Q8_1 and 1313 standalone fusions remain valid.
- Activation/census shows the intended launch/instruction removal on gfx1100/gfx1201 and, where used, gfx1030.
- Fully separated ABBA shows repeatable decode benefit with greedy identity and no prefill/MTP regression.

## Notes

Execution order: sixth. Relative final priority is high among kernel items because both residual mechanisms are visible in the pin, but expected fusion upside is lower than the external report because 1313 already removes most scale/activation launches.

## Change Log

- 2026-10-07T00:39:37.142989+00:00 (created-by): Created by agent
- 2026-10-07: grounded at b11402 and validated 1311/1313; identified flat 64-bit HC indexing plus the remaining scale/sigmoid/scale/HC_POST fusion boundary and assigned changes to existing owners.
