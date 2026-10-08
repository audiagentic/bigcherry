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



## Status 2026-10-08

- Part (a), hyper-connection kernels without 64-bit div/mod: **done** by 1344_dsv4_hc_grid_index (validated, production, default on; prefill +1.3-1.6% on Flash-Next, text identical).
- Part (b), fusion audit: open, decode-side. Now also carries PRBE11 / 37 / 38 / 39 / 40 (GEMV epilogue, residual-add, K+V projection fusions); already covered pieces are 1307-1313 (Q8_1 chain, scale-act fuse) and native GLU. No code this round. Needs the decode kernel census before choosing a fusion.

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

## Consolidated decode-fusion follow-up (PRBE11, PRBE37-40)

Keep this **separate** from QFP35's already-covered `1313_scale_act_fuse` and HC index work; use the existing decode profiling Gate 0 before writing new fusion code. This section is the single planning owner for canonical GLU gaps, literal unary-gated GEMV, residual view ADD and paired MMVQ/K+V.

### Already implemented / measured

- Native b11474 HIP `ggml_cuda_should_fuse_mul_mat` + `ggml_cuda_mm_fusion_args_host/_device` already embeds supported canonical `GGML_OP_GLU` SWIGLU/GEGLU-family cases into MMVQ/MMVF vector epilogues. Native `ggml_cuda_op_unary_mul` fuses literal pointwise unary+MUL as a separate path, but does **not** by itself eliminate the preceding GEMV output store.
- Validated production `1307_q81_activation_cache_mmvq`, `1309_rms_norm_mul_q81`, `1310_act_q81`, `1311_hc_pre_q81`, `1312_mul_q81` eliminate quantified Q8_1 launch boundaries; `1313_scale_act_fuse` handles SCALE/SILU and SCALE/SIGMOID/SCALE. Do not implement these again.
- `1206_rd13_mul_mat_add_view_fusion` is **untested** and already implements the RESHAPE/eligible zero-offset contiguous VIEW + residual `ADD` vector epilogue using `x_bias`. gfx1100 recorded favorable isolated decode tests; gfx1201 and adversarial layout/alias negative fixtures remain (PRBE12, PRBE39). Requalify the same patch, not another residual fusion.
- `1205_rd12_paired_mmvq_dual_output` is **untested**, shared-activation dual-output kernel/matcher for K/V-like pairs (PRBE11/PRBE40). Its matcher must require exact activation identity, equal output `ne[0..3]` and `nb[0..3]`, disjoint output ranges and unchanged GLU precedence. gfx1100 positive evidence exists; gfx1201 incomplete, gfx1151 unavailable.
- `1245_gp11_mmvq_fusion_ncols_gate` is **rejected** (patch.toml); widening 1245 to MTP verify widths cannot be treated as a validated fusion. Preserved as a negative control.

### Remaining scoped work and ranking

1. **Gate 0: profile first (PRBE113 linkage).** On the 27B Q8_0 and Flash-Next composed production traces, count per-token launch pairs/triples `MUL_MAT[/ID] -> UNARY(SILU|SIGMOID|SOFTPLUS) -> MUL`, `MUL_MAT -> VIEW|RESHAPE -> ADD`, and paired GEMV from the same activation; report actual graph names, exact consumer identities, N/quant/strides, GPU time, total launch-gap ms/token. If zero trace-proven unfused cases, do not add a patch.
2. **Literal gated GEMV (PRBE38) vs canonical GLU (PRBE37):** only implement an extra epilogue when the exact literal `UNARY -> MUL` sequence is still separate after `1312/1313` and native GLU fusion. Preserve canonical GLU precedence, type/shape/split guards, `GLU` family non-regression, wrong-wiring and multi-consumer fallbacks. No speculative generic GEMM/prefill claim.
3. **Residual ADD (PRBE39):** complete 1206's negative `VIEW` offset/strided/alias and graph capture/replay checks; retake the incomplete gfx1201 E2E lane before promotion, without duplicating 1206.
4. **Paired K/V MMVQ (PRBE11/PRBE40):** complete 1205 output-layout/overlap/adversarial controls, marker proving true K/V weight pair and full output identity; compare standalone 1205 against native current composition. Preserve its conflict with rejected 1207 as metadata, do not combine implicitly.
5. One independently gated A/B per surviving candidate with `tools/lab/flash-next/queue-env-ab.sh`: 8K/24K/98K decode, verify widths, launch-count reductions, F32 reference/tolerance or bit identity as appropriate, VRAM and <=1% unaffected regression. Rank only by measured eliminated time; stop on no activation or no E2E improvement.

`PRBE11/37/38/39/40` are closed as duplicated **planning** scope under this open QFP35 follow-up. `1205/1206` retain their actual untested states; closure does not promote them.

## Change Log

- 2026-10-08 (triage): Consolidated PRBE11, PRBE37-40 decode fusion; separated validated 1307-1313 from untested 1205/1206 and rejected 1245.

- 2026-10-07T00:39:37.142989+00:00 (created-by): Created by agent
- 2026-10-07: grounded at b11402 and validated 1311/1313; identified flat 64-bit HC indexing plus the remaining scale/sigmoid/scale/HC_POST fusion boundary and assigned changes to existing owners.

## Ledger-events

- chg_20261007_051800_flash-next-prefill-is-about-1_8190
- 2026-10-07T05:18:06.429391+00:00 (updated-by): Updated: section:ledger-events

## Code-level review (2026-10-07)

### 1. Verified facts and corrections

Checked against llama.cpp \`d89651a7b205\` and the current validated-enhancements composition. This review covers **only the remaining fusion half**. The dimensional-index half is obsolete as a plan item: it landed as validated/default-on \`1344_dsv4_hc_grid_index\` and measured +1.3% to +1.6% prefill.

- \`src/models/qwen4exp.cpp::llama_model_qwen4exp::graph::build_hc_mix(...)\` constructs the low-rank HC activation exactly as:
  \`ggml_tensor * lo = build_lora_mm(w_down, xn);\`
  \`lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));\`.
  The residual \`SCALE -> SILU\` mentioned in the old QFP35 text is therefore **already implemented by 1313**; there is no second scale/silu kernel to build.
- \`src/models/qwen4exp.cpp::llama_model_qwen4exp::graph::build_hc_combine(...)\` constructs:
  \`ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));\`
  \`w = ggml_scale(ctx0, w, 2.0f);\`
  followed, when fused HC post is enabled, by
  \`cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);\`.
  This is the exact remaining four-node chain \`SCALE -> UNARY(SIGMOID) -> SCALE -> DSV4_HC_POST\`.
- The target graph calls \`build_hc_combine\` twice per transformer layer: once after attention and once after FFN. \`graph_mtp::graph_mtp(...)\` also calls the same pair of \`build_hc_mix/build_hc_combine\` sequences in its single MTP block, so gfx1030 is a real execution target for the sidecar, not a hypothetical architecture.
- Native HC POST is
  \`template <bool has_comb> static __global__ void dsv4_hc_post_f32(...)\`
  in \`ggml/src/ggml-cuda/dsv4-hc.cu\`. Its arithmetic is
  \`float sum = x[...] * post[idst*sp0 + it*sp1];\`
  then either the comb reduction or \`sum += residual[...]\`, then dst write.
- The host entry is
  \`void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst)\`.
  It binds \`dst->src[0]\`=block output, \`src[1]\`=residual, \`src[2]\`=post weight, \`src[3]\`=optional comb; all are asserted F32.
- \`ggml/src/ggml-cuda/dsv4-hc.cuh\` declares only \`ggml_cuda_op_dsv4_hc_comb/pre/post\` at the pin. A fused HC-post-gate entry therefore needs one explicit declaration there.
- \`ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i)\` first honors \`GGML_CUDA_DISABLE_FUSION\`; any new graph fusion must live below that check.
- Fusion memory safety is performed by
  \`static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph, const int node_idx, const int node_count, const int * out_nodes, const int out_count, const bool is_topk_moe = false)\`.
  It compares actual allocation address ranges. FKE01 therefore applies directly.
- \`1313_scale_act_fuse\` currently inserts matcher edit \`scale-act-match\` after upstream's softcap matcher. Its three-node arm requires contiguous F32 SCALE/UNARY/SCALE and \`ggml_can_fuse(cgraph, i, ops3, 3)\`, calls \`bc_scale_act_fused(..., post)\`, and returns 2. Thus in today's production composition it materializes \`w\` in one fused scale/sigmoid/scale launch, then HC_POST launches separately.
- \`1313::bc_scale_act_kernel\` computes \`scaled = s0*x + b0\`, \`v = op(scaled)\`, and, when the second SCALE exists, \`v = s1*v + b1\`. For the HC chain this exactly produces the current materialized \`post\` value. Its Q8_1 publication path is only taken when \`scale1 == nullptr\`; the HC three-node path does **not** publish Q8_1.
- Correction to the old design: using only \`ggml_can_fuse\` for the new four-node match is insufficient for this project. The final-output index must also pass \`ggml_cuda_check_fusion_memory_ranges\` explicitly, because HC_POST introduces external src0/src1/src3 reads and FKE01 makes address-overlap admission observable.

### 2. Composition and post-patch anchors

Relevant validated patches:

- \`1313_scale_act_fuse\`
  - \`scale-act-kernel\`: inserts \`bc_scale_act_kernel/bc_scale_act_fused\` in \`unary.cu\`;
  - \`scale-act-decl\`: declares it in \`unary.cuh\`;
  - \`scale-act-match\`: inserts the SCALE/UNARY[/SCALE] matcher in \`ggml_cuda_try_fuse\`.
  The new four-node matcher must run **before** the existing three-node HC gate arm, otherwise 1313 consumes the first three nodes and makes HC_POST unreachable to the extension.
- \`1311_hc_pre_q81\`
  - \`hc-pre-q81-kernel\` and \`hc-pre-q81-dispatch\` add the Q8_1-producing PRE path in \`dsv4-hc.cu\`.
  It is a different HC site and has no matcher collision, but it already inserts code before the native HC_POST kernel.
- \`1344_dsv4_hc_grid_index\`
  - \`hc-grid-include\`, \`hc-grid-kernels\`, \`hc-grid-pre-launch\`, \`hc-grid-post-launch\`.
  It inserts the 2-D/3-D kernels immediately before \`void ggml_cuda_op_dsv4_hc_comb(...)\` and replaces the PRE/POST launch blocks. A new fused HC_POST kernel should use the same 3-D coordinate convention directly; it should **not** alter \`hc-grid-post-launch\`.
- \`1310_act_q81\`, \`1312_mul_q81\`, \`1309_rms_norm_mul_q81\`, and \`1307_q81_activation_cache_mmvq\` own other activation/Q8_1 publication paths. The HC gate's existing three-node 1313 arm has \`scale1 != nullptr\`, so none of those cache publications is part of this chain.
- \`1281_moe_mul_mat_id_range\` and \`1283_qwen4exp_expert_parallel\` do not edit these HC files/functions; they affect later MoE ID/range matmuls and Meta placement. No direct anchor collision exists.

Recommended ownership remains **inside \`1313_scale_act_fuse\`**, with a separate default-off subflag. This is not a duplicate scale-act implementation: 1313 already owns the exact producer chain, and the change extends its consumer boundary by one node.

Post-composition anchors:

- Matcher: keep edit id \`scale-act-match\` anchored on the same upstream softcap block, but extend its inserted payload so the four-node HC_POST check occurs before the current three-node SCALE/SIGMOID/SCALE call.
- Fused kernel: in \`dsv4-hc.cu\`, anchor \`insert_before\` on the exact
  \`void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {\`
  signature. 1311 inserts before the native POST kernel; 1344 later inserts its grid kernels before the same COMB function, so this anchor composes without replacing either patch's text.
- Declaration: in \`dsv4-hc.cuh\`, anchor after
  \`void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);\`.
- Do not anchor on 1344's generated \`bc_grid\` launch block; that would couple this feature to 1344's switch implementation unnecessarily.

### 3. Gaps and risks

- **Exact chain only.** The matcher must require ops \`SCALE, UNARY, SCALE, DSV4_HC_POST\`, unary exactly SIGMOID, \`hc_post->src[2] == scale1\`, and the normal edge chain \`act->src[0] == scale0\`, \`scale1->src[0] == act\`. Do not identify HC from shapes alone.
- Require F32 contiguous \`scale0->src[0]\`, \`scale0\`, \`act\`, and \`scale1\`; require HC_POST's existing F32 contracts and supported strides. The original injection tensor is \`scale0->src[0]\`; the fused kernel must read that, not the unmaterialized intermediate nodes.
- Use \`ggml_can_fuse(cgraph, i, ops4, 4)\` for graph-use/single-use eligibility **and** \`ggml_cuda_check_fusion_memory_ranges(cgraph, i, 4, &out_idx, 1)\` for allocation-overlap safety. This is the FKE01-critical missing condition.
- The current model calls HC_POST with \`comb == nullptr\`, but the CUDA op supports non-null comb. V1 should support both at kernel level because the arithmetic is trivial to preserve, while the graph matcher should not assume a model name. Tests must cover both.
- Meta: DSV4 HC ops are scalar/generic Meta operations; matching occurs only after a backend-local subgraph exists. Derive \`n_embd/hc/n_tokens\` and strides from the physical local tensors. Never assume all three devices see the same token extent.
- gfx1100/gfx1201: use the 1344-style 3-D grid with the existing block size; no architecture-specific arithmetic is justified. gfx1030 executes the same HC combine in \`graph_mtp\`, but its enablement must be based on measured benefit rather than a family-wide predicate.
- 1281/1283: the fusion is upstream of expert routing and must not inspect or alter range op params, expert ids, or delayed all-reduce state. The only interaction is global performance/allocator layout through the containing graph.
- Q8_1 cache: do not publish any Q8_1 entry from the four-node fused HC_POST. The existing 1313 three-node HC gate path does not publish one, and HC_POST's output has different semantics/shape. Preserve 1311 PRE publication unchanged.
- CUDA graphs: the fused kernel requires no new pool allocation or host copy and is therefore capture-friendly. Flag values must be process-stable/cached. Marker logging must be one-shot and outside stream synchronization.
- FKE01/numerics: replacing a materialized F32 gate tensor with a register value can produce last-bit differences through compiler contraction even if source operation order is written identically. Do not claim bit identity for the fusion. The only valid bit/greedy identity isolation is patch A/B with \`GGML_CUDA_DISABLE_FUSION=1\` on both arms, where the new code is inert. Fusion-on equivalence is a separate tolerance/greedy probe.
- The feature removes one small producer launch for every matched HC_POST; it does not remove the HC_POST launch itself, and it does not improve the already-landed 1344 coordinate indexing. External “hundreds of launches” numbers therefore substantially overstate remaining opportunity.
- No host thread is useful; this is a single-stream graph-fusion boundary.

### 4. Concrete implementation outline

Keep package ownership in \`patches/1313_scale_act_fuse\`.

New subflag:
\`BIGCHERRY_HC_POST_GATE_FUSE=1\` enables; **default 0**. Existing \`BIGCHERRY_SCALE_ACT_FUSE\` remains independently default-on. Require both for the extension so disabling 1313 restores the original graph behavior.

Add in \`dsv4-hc.cu\`:

\`template <bool has_comb> static __global__ void bc_dsv4_hc_post_gate_f32(...)\`

with inputs:
- \`x\`, \`residual\`, **original \`inject\`**, optional \`comb\`, dst;
- \`n_embd, hc, n_tokens\`;
- x/residual/inject/comb/dst strides in float elements;
- \`s0, b0, s1, b1\`.

Launch geometry must match 1344 POST:
- x: \`ceil(n_embd/256)\`;
- y: \`hc\`;
- z: \`n_tokens\`;
- \`i0 = blockIdx.x*blockDim.x + threadIdx.x\`, \`idst=blockIdx.y\`, \`it=blockIdx.z\`.

For each output:
\`const float post = s1 * op_sigmoid(s0 * inject[idst*si0 + it*si1] + b0) + b1;\`
then execute the native HC_POST arithmetic unchanged.

Add host entry:
\`void bc_cuda_op_dsv4_hc_post_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * scale0, const ggml_tensor * scale1, ggml_tensor * hc_post);\`

It reads scale/bias via \`ggml_get_op_params_f32\`, validates grid-dimension conversion, derives all physical strides, selects \`has_comb\`, and launches the fused kernel. It must allocate nothing.

Matcher, before 1313's current three-node arm:

1. require \`BIGCHERRY_HC_POST_GATE_FUSE=1\`;
2. require \`i+3 < n_nodes\`;
3. exact ops/edges: SCALE -> SIGMOID -> SCALE -> DSV4_HC_POST, with \`hc_post->src[2] == scale1\`;
4. existing 1313 contiguous/F32 checks for producer tensors plus HC_POST type/shape/grid checks;
5. \`ggml_can_fuse(cgraph, i, ops4, 4)\`;
6. \`const int out_idx=i+3; ggml_cuda_check_fusion_memory_ranges(cgraph, i, 4, &out_idx, 1)\`;
7. call \`bc_cuda_op_dsv4_hc_post_gate(*cuda_ctx, scale0, scale1, hc_post)\`; return 3.
8. Otherwise fall through unchanged to 1313's current three-node/two-node logic.

Activation marker, one-shot per process/device:
\`BIGCHERRY_PATCH_HIT patch=1313_scale_act_fuse path=hc_post_gate cc=<cc> n_embd=<...> hc=<...> tokens=<...> comb=<0|1>\`.

Edit list inside 1313:
- \`hc-post-gate-kernel\`: \`ggml/src/ggml-cuda/dsv4-hc.cu\`; anchor exact COMB host-function signature above; mode \`insert_before\`, \`expect_matches=1\`.
- \`hc-post-gate-decl\`: \`ggml/src/ggml-cuda/dsv4-hc.cuh\`; anchor exact existing POST declaration; mode \`insert_after\`, \`expect_matches=1\`.
- \`scale-act-match\`: modify the existing edit payload, preserving its upstream softcap anchor and \`insert_after\` mode; put the four-node check before the current three-node branch.
- Add \`EnvDoc('BIGCHERRY_HC_POST_GATE_FUSE', '0|1', '0 (off)', ...)\`.
- Do **not** edit 1344, 1311, Q8_1 cache code, model graph construction, or Meta.

Offline test should extend 1313's existing mechanics test and compare exact generated kernel/matcher text, including near-miss graph predicates. 1344's independent test remains the owner of PRE/POST indexing text.

### 5. Gate 0 and lightweight validation

Gate 0 must be positive **with 1313 and 1344 already enabled**.

Run production Flash-Next, f16 KV, ub512, current three-card tensor split at 8K and 24K prefill with rocprof kernel trace plus \`BIGCHERRY_PATCH_TRACE\`/a fusion census. Count matched HC-combine sequences and correlate:
- current 1313 \`bc_scale_act_kernel<sigmoid,...>\` with \`post=1\`;
- immediately following 1344 \`dsv4_hc_post_grid_f32\`;
- per Agent_Id call count and aggregate GPU duration;
- expected structural opportunity: two HC combines per target layer, using the actual built graph count rather than a hardcoded layer number.

Gate is positive only if:
- >=95% of fused HC_POST graph sites present as the exact 1313 three-node producer + 1344 POST pair (otherwise there is a composition/matcher issue to solve first); and
- the removable **1313 producer launch alone** accounts for **>=0.35% of critical-rank prefill wall time or >=0.70 ms per 512-token ubatch** at either 8K or 24K.

Also census the gfx1030 MTP sidecar during depth-3 drafting. It may be enabled there only if the same exact chain is present and the removable producer launch is >=0.5% of drafter step GPU time; do not enable gfx1030 merely because the source graph contains the chain.

Validation:
- extend 1313 offline mechanics test + patch-lint;
- synthetic exact four-node graph plus near misses: wrong unary, shared intermediate, wrong \`src[2]\`, noncontiguous producer, overlapping output/external source, unsupported type, out-of-range grid;
- activation marker must fire only on the four-node path; existing 1313 marker/path remains for two/three-node fusions;
- backend-op comparison of fused vs materialized HC POST over token counts 1,2,8,16,512, both comb variants and physical strides; use explicit F32 tolerance and compare greedy model output;
- fully separated performance ABBA via \`tools/lab/flash-next/queue-env-ab.sh\`, A=flag 0, B=flag 1, with 1313/1344 otherwise fixed and fusion **on** so the feature can execute;
- separate identity/control ABBA with \`GGML_CUDA_DISABLE_FUSION=1\` on **both** A and B: outputs/generated text must be identical and the new marker must not fire;
- fusion-on probes required by FKE01: record accepted/rejected fusion counts, memory-range rejection count if instrumented, greedy md5, per-rank kernel totals, prefill t/s, CUDA-graph capture/recapture count, and MTP acceptance/depth;
- no q4 KV.

### 6. Verdict

**GO AFTER GATE 0.** The exact residual chain exists twice per target layer and in the MTP block, and implementation is small when owned by 1313; however 1313 already removed the two scale/activation launches and 1344 already accelerated HC_POST, so only one small gate-materialization launch remains.

Expected gain on the production three-card Flash-Next topology if Gate 0 passes: **+0.1% to +0.5% prefill**. Treat any larger result as requiring a fusion-census explanation rather than assuming the external report transfers.
- 2026-10-08T09:46:12.268171+00:00 (updated-by): Updated: section:notes
