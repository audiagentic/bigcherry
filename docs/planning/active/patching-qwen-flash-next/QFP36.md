---
id: QFP36
order: 36
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:40.654749+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# MoE router: split-K router GEMM and multi-warp routing helper

## Description

External report: (a) a small F32 MoE router GEMM split across eight K partitions plus a deterministic reduction; (b) a 16-warp routing helper replacing the single-warp token scan. The reported timings came from a different 4-GPU build and are hypotheses only.

Both target mechanisms are identifiable at b11402:

- Qwen4Exp's `ffn_gate_inp` is a 2D `[n_embd, n_expert]` router weight and `build_moe_ffn()` computes router logits with an ordinary MUL_MAT before top-k. The production model has 512 experts.
- `ggml_cuda_launch_mm_ids_helper()` launches **one physical warp per expert**, scans all tokens/routes for that expert, stores matches in shared memory sized by `n_tokens`, then writes compact row maps and expert bounds.
- BigCherry 1281 translates range-MUL_MAT_ID global ids to local ids (or `INT_MAX` for a nonlocal route) before calling that same helper. A replacement helper must preserve this exact contract.

Gate each optimization separately. The router GEMM changes F32 accumulation order and is higher numerical risk; the helper can and should be output-identical.

## Steps

1. Gate 0: profile ub512 prefill and record router MUL_MAT + `mm_ids_helper` call count/time per layer and per physical device in row-split and, if enabled, expert-parallel mode.
2. Qualify the helper first because it is exact and has a clean API boundary.
3. Qualify router split-K separately using an explicit router marker; do not dispatch by device ordinal or by “512” alone.
4. Run helper adversarial identity tests; run router score/top-k probes.
5. ABBA each change independently before a combined run.

## Detailed Solution & Technical Design

### A. Multi-warp ids helper

New mechanism:

`patches/1348_moe_ids_multiwarp`

Flag:

`BIGCHERRY_MOE_IDS_MULTIWARP=0|1`, default `0`.

Keep the signature of `ggml_cuda_launch_mm_ids_helper()` unchanged so ordinary MMQ and 1281 range MMQ share the implementation.

For the production top-10 case, add a specialized block with e.g. 16 physical warps per expert. Do not assume 16 is universally optimal; make the compiled candidate explicit and benchmark gfx1100/gfx1201/gfx1030 separately.

#### Two-pass stable algorithm

The current helper's observable ordering is expert-major, then ascending token order within each expert, with the original route-slot index attached. Preserve that ordering exactly.

Pass 1, inside one expert block:

1. Divide `[0,n_tokens)` into contiguous token ranges, one per warp. Contiguous ranges are important for stable global order.
2. Within each warp use the existing top-k padded-slot technique (top-10 -> 16 slots) to test whether each token uses this expert.
3. Accumulate per-warp:
   - number of tokens matching this expert;
   - number of route assignments whose expert id is lower than this expert (`nex_prev` contribution).
4. Store only O(number_of_warps) counts in shared memory.
5. Warp 0 computes exclusive prefixes of match counts and the block-wide `nex_prev`.

Pass 2:

1. Rescan the same contiguous token range.
2. Compute each matching token's stable rank inside its warp.
3. Write at `nex_prev + warp_prefix + local_rank`:
   - `ids_dst`;
   - forward `ids_src1`, or inverse `ids_src1` when `write_inverse=true`.
4. One thread writes `expert_bounds[expert]`; the final expert also writes `expert_bounds[n_experts]`.

The second read pass is intentional: it removes the `n_tokens`-sized shared store and exposes enough parallelism without introducing a global temporary or atomics.

#### Range compatibility

1281's `INT_MAX` sentinel must retain current behavior:

- never equals a local expert;
- contributes false to `expert_used < expert`;
- generates no compact row.
- when inverse mode is requested, nonlocal slots remain at the caller's preinitialized `-1`.

Do not move range translation into the helper. Keeping translation separate preserves 1281 ownership and lets the same helper serve ordinary/range ids.

Fallback to the current one-warp helper for unsupported top-k, very small token counts where extra warps lose, or resource-limit failures.

### B. Split-K router GEMM

New mechanism, same QFP but separate qualification package to keep bisectability:

`patches/1349_moe_router_splitk`

Flag:

`BIGCHERRY_MOE_ROUTER_SPLITK=0|1`, default `0`.

#### Explicit router marking

Do not identify the router by tensor name or exact dimensions. In `src/llama-graph.cpp::build_moe_ffn`, mark only the ordinary MUL_MAT that produces router logits using an unused BigCherry op-param marker. The marker is copied with the graph node through Meta transformation.

The CUDA dispatch checks:

- marker present;
- `GGML_OP_MUL_MAT`;
- src0/src1/dst F32;
- compatible contiguous/aligned layout;
- AMD HIP backend;
- dimensions inside the qualified range;
- no conflicting fusion;
- flag on.

All other F32 MUL_MATs retain normal dispatch/QFP34 behavior.

#### Split-K layout

For a physical local router matrix:

- K = local/full inner dimension `src0->ne[0]`;
- M = local output expert rows `src0->ne[1]`;
- N = activation token columns.

Start with eight K partitions only because that is the external hypothesis. Require K divisibility/alignment needed by vector loads; otherwise fall back.

Use a temporary F32 workspace from the existing CUDA/HIP pool:

`[split_k, M, N]`.

Kernel 1 computes one partial dot per `(partition, output_expert, token)`. Kernel 2 reduces partitions in a fixed ascending partition order into `dst`. Avoid floating atomic accumulation: it would make run-to-run order nondeterministic and would weaken the routing test.

If profiling shows workspace or second-launch overhead dominates at smaller N, gate by measured token width rather than forcing split-K.

### Architectures

- gfx1100/gfx1201: primary router/helper prefill targets.
- gfx1030: helper/router may execute in the MTP sidecar; qualify rather than assume the same thresholds.
- Runtime CC and physical shape decide; no ROCm device ordinal is embedded.

### Meta split

Router split-K operates on the physical matrix passed to the backend and does not alter split state. If Meta has split the router output rows, only local M changes; existing Meta reduction/gather semantics remain authoritative.

For expert parallelism, the routing decision still refers to global 512 expert ids. 1283/1281 then translate/group local expert work. The helper optimization must accept the local `n_experts` it is passed and must not assume 512 experts per device.

No collective behavior changes.

### Fusion

The router marker applies only to the unfused router MUL_MAT; if a future fusion consumes it, split-K must fail closed until that fusion explicitly carries the contract. The helper sits inside MMQ and composes with current gate/up GLU fusion and QFP30 range dedup because its output buffers are unchanged.

## Code Samples & Guidance

Helper anchors:

- `ggml/src/ggml-cuda/mmid.cu::mm_ids_helper`;
- `launch_mm_ids_helper`;
- `ggml_cuda_launch_mm_ids_helper`;
- preserve `ggml/src/ggml-cuda/mmid.cuh` public signature if possible.

Router marker anchor:

- `src/llama-graph.cpp::llm_graph_context::build_moe_ffn` at router logits construction from `gate_inp`.

Router dispatch anchors:

- `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat`;
- new implementation in a dedicated CUDA/HIP compilation unit/cuh if the patch framework can add it cleanly; do not bury split-K inside generic rocBLAS code.

Suggested activation markers:

- `BIGCHERRY_PATCH_HIT 1348 moe_ids_multiwarp experts=<...> tokens=<...> used=<...> inverse=<0|1>`;
- `BIGCHERRY_PATCH_HIT 1349 moe_router_splitk m=<...> n=<...> k=<...> split=8`.

Emit once per representative configuration, not per layer.

## Files

### Helper

Create:

- `patches/1348_moe_ids_multiwarp/{patch.py,README.md,SUMMARY.md}`;
- `tools/tests/patch/test_1348_moe_ids_multiwarp.py`.

Production targets:

- `ggml/src/ggml-cuda/mmid.cu`;
- `mmid.cuh` only if a private helper declaration is required.

Do not modify 1281 if the public helper contract remains unchanged.

### Router

Create:

- `patches/1349_moe_router_splitk/{patch.py,README.md,SUMMARY.md}`;
- `tools/tests/patch/test_1349_moe_router_splitk.py`.

Production targets:

- `src/llama-graph.cpp` for the router marker;
- `ggml/src/ggml-cuda/ggml-cuda.cu`;
- dedicated CUDA/HIP source/header and build-list edit if required.

## Validation

### Helper mechanics/identity

- Patch-lint and exact/idempotent mechanics.
- Standalone old-vs-new helper test with random/adversarial ids:
  - token counts 1, 2, 31, 32, 33, 128, 512;
  - top-k 2/4/6/8/10/16/32 plus generic;
  - duplicate/unsorted expert selections where the API permits them;
  - expert ids at 0 and `n_experts-1`;
  - `INT_MAX` sentinel patterns matching 1281 range translation;
  - forward and inverse modes.
- Require byte-identical `ids_src1`, `ids_dst`, `expert_bounds`.
- Range QFP30 test must still pass with identical active/inactive mapping.

### Router mechanics/equivalence

- Compare split-K scores against current F32 router over random/production-shaped inputs.
- Because K reduction order changes, use explicit F32 tolerance rather than bit identity for scores.
- Separately require **top-k ids unchanged** on captured production probes and adversarial near-tie probes. If near ties can change selected experts under normal inputs, reject rather than calling that harmless floating-point noise.
- End-to-end greedy target output must be identical for promotion.

### Activation/timing

- Kernel census shows one-warp helper replaced on qualified shapes and lower total helper GPU time.
- Router marker proves only router GEMMs use split-K; rocBLAS/generic F32 census changes only for those nodes.
- Record workspace peak.

### Hardware ABBA

Use `tools/lab/flash-next/queue-env-ab.sh`, full process separation, ub512.

1. helper A/B with router split-K off;
2. router split-K A/B with helper fixed;
3. combined B only after isolated wins.

Run pp4096 and a 24K production prefill; add ~98K only if the per-layer overhead remains material. Record prefill t/s, helper/router GPU time, VRAM peak, router top-k probes and greedy identity.

No multi-session contract campaign.

## Effort & Risk

Effort: M for helper, M/H for split-K depending on workspace/kernel integration.

Risk:
- helper: medium implementation risk, low numerical risk;
- router split-K: medium/high numerical/performance risk because expert selection is discontinuous near score ties.

Expected gain on our topology: helper medium-confidence for ub512 because the current one-warp/all-token implementation is visibly serial; router split-K is lower-confidence until its per-layer aggregate is measured. The slow R9700 PCIe link is not directly relevant because neither mechanism adds host/P2P traffic.

## Standards

- Exact helper ordering and maps.
- No atomics in router score reduction.
- No hard-coded 512-expert/device assumption.
- No device-ordinal/topology assumption.
- No q4 KV.
- No legacy shim.
- Default off during qualification.
- Preserve 1281 range sentinel/inverse-map semantics.
- External timings are hypotheses.

## Acceptance Criteria

- Gate 0 attributes material time to the router/helper on our ub512 workload.
- Multi-warp helper is byte-identical to the existing helper, including 1281 range/inverse cases.
- Router split-K affects only explicitly tagged router MUL_MAT nodes.
- Router score error stays within stated tolerance and production top-k ids are unchanged.
- Greedy target output is identical.
- Fully separated ABBA shows a repeatable prefill benefit without VRAM or decode regression.
- If either submechanism is not material, close it independently.

## Notes

Execution order: seventh. Within QFP36, qualify the exact multi-warp helper before the numerically riskier router split-K. In final gain/effort ranking, the helper can move ahead of QFP34 if Gate 0 confirms ~layer-per-layer serialization at ub512.

2026-10-07 Gate 0 (prefill kernel profile, production build b-prod-1343 with 1343 + 1344, 38.7K-token fill at 977 t/s under rocprofv3, run gate0-d24576). Kernel time per target card 28.0 s over a 99 s span (28% busy). mm_ids_helper<10>: 3.23 s summed over devices = about 3.8% of all kernel time on the three target cards - the multi-warp helper is worth doing. Float matmul family (rocBLAS SGEMM, largest kernel Cijk_..._MT64x64x8 7.24 s): 12.4-16.0% of kernel time per card; the profile does not split the router GEMM from other F32 matmuls, a dispatch census by shape is still needed for split-K (and for QFP34). Family shares per XTX: collectives 30-31% (ncclDevKernel_Generic_4 27.6 s over devices; R9700 41.6%), MMQ 27%, float matmul 16%, flash attention 8.7%, norm/activation 4%, rows/concat/mask 3.3%, GDN 2.5%. Collectives and the 72% idle span are the largest prefill levers (QFP39, QFP41).

2026-10-09 larger-ubatch results on the released build (Flash-Next, 245,760 ctx, f16 KV). Plain ub768: prefill +5.7/+5.0/+6.2% at 8K/24K/98K, decode -2/-2/+3.6%, ub896 does not fit (87.5 MiB short). 1330 (in-place QSA mask) alone at ub512: neutral, text identical. 1330+ub1024: prefill +11/+13/+13%, decode -3% at 8K, level at 24K, -18% at 98K (acceptance drops to ~317/577); not caused by deferred catch-up or look-ahead. 1332 chunking (BIGCHERRY_QSA_CHUNK=256)+ub1024: prefill only +2.3..4.7%. Probe fidelity at 8K against the CPU f32 reference (24 probes; production = 23/24 top-1, TV mean 0.0764): 1330+ub1024 23/24, TV 0.0804; chunk256+ub1024 24/24, TV 0.1103 (max 0.506); plain ub768 22/24, TV 0.0949 (max 0.409). 1330+ub1024 is the closest to production of the three; plain ub768 is not obviously safer than it. Open: the 98K decode loss with ub1024 is unexplained; no larger-batch setting is adopted until it is.

2026-10-09 ub1024 + 1330 at 98K (b-1330d, same binary, ABBA): prefill 1364 / 1372 vs 1172 / 1212 t/s, decode 56.4 / 51.2 vs 69.2 / 69.5 t/s, acceptance 315/583 and 316/580 vs 344/500 and 343/502, greedy text differs. Probe comparison at 98K against production ub512 (no CPU reference at this depth): top-1 21/24, TV mean 0.1234, max 0.884; production against itself is 24/24, TV 0. The same comparison at 8K was 24/24, TV mean 0.0655. The target's own next-token distributions move about twice as far from production at 98K as at 8K, and three of 24 probes change their top token, so the decode loss is not only a draft effect. Conclusion: ubatch 1024 is not adopted for Flash-Next at long context. 1330 itself stays neutral and text-identical at ub512. Open question for whoever picks this up: which side is closer to the CPU reference at depth (a reduced-depth CPU reference at ~32K is affordable).

2026-10-09 router Gate 0 from the released-build kernel census (run census-rel1-d24576, b-metamem-rel3, Flash-Next 38.7K-token fill, 75 chunks, 1203.5 t/s under rocprofv3, ~32 s prompt wall). One rocBLAS F32 GEMM runs exactly once per layer per chunk on every target card (3600 calls = 75 x 48), which is the call pattern of the router: on each XTX `Cijk_Alik_Bljk_SB_MT64x64x8` grid 2048x8, 0.165 ms/call, 0.595 s (7.9 ms per chunk, ~1.9% of prompt wall); on the R9700 a different Tensile kernel, `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT16x16x16` grid 2048x32, 0.432 ms/call, 1.557 s (20.7 ms per chunk, ~4.9% of prompt wall, 2.6x the XTX time for the same work). Both exceed the >=1.0 ms per ubatch gate. Not yet shown: that this kernel is the router and not another once-per-layer F32 matmul (needs the dispatch census by callback name and shape), the weight type, and whether the R9700 is on the critical path at that point. All F32 GEMM together is 6.9% of kernel time. Next: dispatch census, then a router-marked custom F32 kernel aimed first at the gfx1201 case, where rocBLAS picks the slow fallback kernel; bit-identical top-k ids required as already stated. Other shares in this census: collectives 37.9%, MMQ 28.0%, flash attention 7.5%, GDN 3.3%, multi-warp ids helper 1.0% (was 3.8% before 1345).

2026-10-09 (same census) the once-per-layer GEMM is the router: on the R9700 3300 of its 3600 calls are followed directly by topk_moe_cuda > bc_mm_ids_helper_mw, the other 300 by soft_max > argsort (the non-fused top-k path); it is always preceded by dsv4_hc_pre_grid. Router Gate 0 is therefore positive on time: 7.9 ms per 512-token chunk on each XTX, 20.7 ms on the R9700. Caution on the R9700 figure: that card has the largest collective share (it waits for the XTXs), so its router time is probably not on the critical path; the XTX 7.9 ms (~1.9% of prompt wall) is the realistic ceiling, and a custom kernel would recover part of it. Weight type still to be read from the model before any code.

## Status 2026-10-08

- Part (b), multi-warp routing helper: **done** by 1345_moe_ids_multiwarp (validated, production, default on; 8 warps per expert for >= 128 tokens; prefill +1.6-2.1% on Flash-Next, +2.0% on Gemma, text identical).
- Part (a), split-K router GEMM (`ffn_gate_inp`, `[n_embd, 512]` F32): open. Neither 1347 (2..8-row F32 weights) nor 1350 (Q8_0 few-tile Stream-K) covers a 512-row F32 router. No measurement yet of its share of prefill time: the kernel census of the released build (queued 2026-10-08, `queue-prefill-profile.sh b-metamem-rel3 census-rel1 24576`) decides whether it is worth a patch.

## What we already have

### Router at b11402

- `src/models/qwen4exp.cpp::llama_model_qwen4exp::load_arch_tensors`
  - creates `layer.ffn_gate_inp` as `{n_embd, n_expert}`.
- `src/models/qwen4exp.cpp::graph::build_layer_ffn`
  - passes that tensor to generic `build_moe_ffn(..., n_expert, n_expert_used, ...)`.
- `src/llama-graph.cpp::llm_graph_context::build_moe_ffn`
  - owns the generic router-logit MUL_MAT, top-k ids/weights and routed expert graph.
- `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat`
  - routes an ordinary F32 router matmul through MMVF/MMF/rocBLAS according to width/shape; there is no router-specific split-K path.
- The logical router output is `n_expert x n_tokens`; with Meta tensor split, the CUDA backend must use the **physical local tensor dimensions** and leave Meta's existing split/reduction semantics unchanged.

The external “512x512” label must not be hard-coded: 512 experts is true for our production model, but `n_embd`, local split dimensions and token width must come from tensors.

### Routing helper at b11402

- `ggml/src/ggml-cuda/mmid.cu::mm_ids_helper<n_expert_used_template>`
  - one block per expert;
  - `__launch_bounds__(physical_warp_size, 1)`;
  - one physical warp per block;
  - specialized cases for top-k 2/4/6/8/10/16/32;
  - top-10 pads route slots to 16 and packs multiple token slots into a warp;
  - scans all `n_tokens` for every expert;
  - tracks `nex_prev` = number of route assignments to lower-numbered experts;
  - stores matched `(token, route_slot)` in shared memory in token order;
  - writes `ids_dst`, forward/inverse `ids_src1`, and `expert_bounds`.
- `launch_mm_ids_helper`
  - allocates `n_tokens*sizeof(mm_ids_helper_store)` dynamic shared memory per expert block;
  - asserts that this fits the device shared-memory-per-block limit.
- `ggml_cuda_launch_mm_ids_helper`
  - is the public entry point used by MMQ.

### BigCherry overlap

- `patches/1281_moe_mul_mat_id_range`
  - adds `ggml_cuda_mm_ids_range_translate()`;
  - range ids become local expert indices or `INT_MAX`;
  - then the unchanged public helper groups only valid local experts;
  - QFP30 additionally uses the helper's inverse map for broadcast activation dedup with `-1` preinitialization for nonlocal routes.
- `patches/1283_qwen4exp_expert_parallel` can reduce the local expert count per device, so helper qualification must include both upstream row-split and whole-expert range modes.
- QFP34 targets thin ordinary F32 MUL_MAT generally. Router split-K must be separately tagged/qualified so these two experiments do not accidentally intercept one another.

Neither split-K routing nor a multi-warp ids helper is currently implemented.

## Change Log

- 2026-10-07T00:39:40.654749+00:00 (created-by): Created by agent
- 2026-10-07: grounded at b11402 plus 1281/QFP30; documented stable two-pass multi-warp grouping and explicit-marker deterministic split-K router design.

## Code-level review (2026-10-07)

### 1. Verified facts and corrections

Checked against llama.cpp `d89651a7b205` and the current validated production composition.

- Qwen4Exp creates `layer.ffn_gate_inp = create_tensor(..., { n_embd, n_expert }, flags)` in `src/models/qwen4exp.cpp::load_arch_tensors`, and `graph::build_layer_ffn` passes it to `build_moe_ffn(...)`.
- The generic router path is in `src/llama-graph.cpp::llm_graph_context::build_moe_ffn`. The exact construction is `logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]`, followed by `cb(logits, "ffn_moe_logits", il)`, then softmax/sigmoid and `ggml_argsort_top_k(..., n_expert_used)`.
- `build_lora_mm` first does `ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);`, then can wrap that result in SCALE/MUL/ADD when weight scaling or LoRA is active. Therefore a router marker added after `build_lora_mm` must mark only when `logits->op == GGML_OP_MUL_MAT && logits->src[0] == gate_inp`; otherwise fail closed. The old plan's statement that the router node is unconditionally the returned ordinary MUL_MAT is too broad.
- `ggml_mul_mat` creates an F32 result and leaves op params zeroed. `ggml_mul_mat_set_prec` uses i32 op-param index 0 and `ggml_mul_mat_set_hint` uses index 1. A BigCherry router marker may therefore use a high unused slot such as i32[6] without touching precision/hint; use a distinct ordinary-MUL_MAT magic, not 1281's MUL_MAT_ID semantic.
- The router weight's runtime storage type is **not guaranteed by source**. The UD-IQ4_XS GGUF may still store this tensor as F32, but Gate 0 must record the actual `gate_inp->type`; do not code split-K if production src0 is quantized.
- Important Meta correction: `ffn_gate_inp.weight` is not matched by the FFN up/gate/down split rules in `src/llama-model.cpp`; it falls through to `GGML_BACKEND_SPLIT_AXIS_MIRRORED`. In `ggml-backend-meta.cpp::handle_mul_mat`, mirrored src0 + mirrored src1 remains mirrored; mirrored src0 + AXIS_1 src1 returns the src1 split. Thus router M stays the full `n_expert` (512 here); local token N may be split. The existing plan's “if Meta has split router output rows, local M changes” is wrong for this production pin/composition.
- The current helper is exactly `template <int n_expert_used_template> __launch_bounds__(ggml_cuda_get_physical_warp_size(), 1) static __global__ void mm_ids_helper(...)` in `ggml/src/ggml-cuda/mmid.cu`. One block is launched per expert, block size is one physical warp, and dynamic shared memory is `n_tokens*sizeof(mm_ids_helper_store)`.
- The helper's observable contract is explicit in code: `ids_dst[nex_prev + itc] = it*n_expert_used + iex_used`; `ids_src1` is either the forward map `it*sis1 + iex_used % nchannels_y` or inverse map `ids_src1[it*n_expert_used + iex_used] = nex_prev + itc`; `expert_bounds[expert] = nex_prev`, and the last block writes `expert_bounds[gridDim.x]`.
- Top-k specializations are 2/4/6/8/10/16/32; top-10 pads to 16 lanes through `mm_ids_pow2<10>::value`. On RDNA3/RDNA4 `ggml_cuda_get_physical_warp_size()` is 32 at this pin, so top-10 handles two token slots per warp iteration.
- 1281 range translation produces local id or `INT_MAX`. In the native helper, `INT_MAX < expert` is false and `INT_MAX == expert` is false, so nonlocal routes neither contribute to `nex_prev` nor create rows. Preserve exactly.
- The helper subdesign is valid in principle, but a 16-warp candidate means 512 threads/block on gfx1100/gfx1201; occupancy/resource limits must be measured, not assumed.

### 2. Composition and exact anchors

- `1281_moe_mul_mat_id_range` edits `mmid.cu` with `mmid-range-translate`: its anchor is the public `ggml_cuda_launch_mm_ids_helper(...)` definition and it inserts `mm_ids_range_translate` immediately before that definition. It also edits `mmq.cu` with `mmid-range-mmq-setup`, `mmid-range-mmq-helper`, and `mmid-range-mmq-dedup-call`. The new helper package must anchor on the **post-1281** public helper signature/body and keep that signature unchanged.
- For the multi-warp package, avoid replacing the public signature (that would collide semantically with 1281). Insert the new kernel/launcher immediately before the composed `void ggml_cuda_launch_mm_ids_helper(...)`, then replace only the `case 10:` call with an opt-in chooser.
- `1283_qwen4exp_expert_parallel` edits `src/llama-model.cpp` (`moe-ep-axis`, `moe-ep-granularity`, `moe-ep-shares`) and `ggml-backend-meta.cpp` (`moe-ep-split-state`, `moe-ep-range-node`, `moe-ep-delay`). It does not edit `build_moe_ffn` or `mmid.cu`, but it changes the local `n_experts` presented to range MMQ.
- `1237_rd30_moe_mmq_compact_grid` and `1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2` change MMQ launch-grid construction after grouping. They consume `expert_bounds`; byte-identical helper outputs are therefore a hard composition requirement.
- No current validated patch reviewed here edits the `logits = build_lora_mm(gate_inp, cur);` anchor in `build_moe_ffn`. Router marking should insert immediately after this line and before the SQRT_SOFTPLUS precision override/callback.
- No current validated patch replaces the ordinary `ggml_cuda_mul_mat` MMVF/MMF/MMQ/fallback chain. A future QFP34 implementation would share this function; router split-K must insert before generic MMVF and key strictly on the router marker so both patches compose without order-dependent interception.

### 3. Gaps and risks

- **Router type/shape risk:** source does not prove F32 router weights. Gate 0 must record src0/src1/dst types and local `M,N,K`. Split-K is a no-op unless all three are F32 and the observed layout is contiguous/aligned.
- **Meta:** router weight is mirrored, not expert-sharded. Do not use local expert count from 1283 as router M; the router still scores global experts. With token-split activation, each rank sees local N.
- **Expert parallel:** 1281/1283 affect the later `MUL_MAT_ID` path, not router logits. The helper sees local expert ids after 1281 translation in range mode. It must accept both full `n_experts=512` row-split mode and smaller local `n_experts` range mode without a fixed topology assumption.
- **Q8_1 activation work:** 1281 QFP30 uses `write_inverse=true` maps for range MMQ dedup/scatter. A helper mismatch can silently corrupt the activation ordering even if `expert_bounds` looks plausible. Test all three outputs byte-for-byte.
- **Fusion/FKE01:** helper replacement allocates no global workspace and should not alter allocation layout. Router split-K does require an F32 workspace if implemented as two kernels; pool allocation can shift addresses and therefore alter unrelated fusion selection. Identity must be shown with `GGML_CUDA_DISABLE_FUSION=1` on both arms, then fusion-on probes separately.
- **CUDA graphs:** helper launch geometry is deterministic from tensor shape and is capture-safe if it introduces no host sync/allocation. Router workspace from `ctx.pool()` is a capture/layout risk; reserve before launch through the normal pool and do not grow any persistent slab during capture. Record graph-capture hit/miss/recapture behavior in probes.
- **Numerics:** deterministic ascending split reduction is deterministic but not rocBLAS-equivalent. Top-k is discontinuous near ties. Score tolerance alone is insufficient: selected expert ids must match on production probes. 1294 deterministic tie handling does not make different floating scores safe.
- **gfx1030:** the target router/helper work is on the main target GPUs during target prefill; the 6900 XT is the drafter. Do not enable gfx1030 in V1 unless Gate 0 shows the same mechanism executing materially in the built-in-MTP 27B lane or sidecar path.
- A host thread would not help either mechanism; keep both GPU-local.

### 4. Concrete implementation outline

#### 1348 multi-warp helper

Flag: `BIGCHERRY_MOE_IDS_MULTIWARP=1`, default `0`.

Add a top-10-only V1 kernel:
`template <int n_warps> static __global__ void bc_mm_ids_helper_multiwarp_10(const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds, int n_tokens, int nchannels_y, int si1, int sis1, bool write_inverse)`.

Use `blockDim.x = n_warps * physical_warp_size`; derive `warp_id` and lane. Each warp owns a contiguous token interval. Pass 1 counts (a) matching tokens for this expert and (b) all route assignments with id lower than this expert. Shared memory is O(n_warps): per-warp match/lower counts plus prefixes. Warp 0 computes exclusive match prefixes and total `nex_prev`. Pass 2 rescans and writes exactly the native expert-major/token-ascending order. Do not use atomics for output positions.

Launcher:
`static void bc_launch_mm_ids_helper_multiwarp_10(..., int n_warps, cudaStream_t stream)`.
Start with candidates 4/8/16 warps compiled; runtime selection is a measured threshold by `n_tokens` and exact cc, not an ordinal.

Public `ggml_cuda_launch_mm_ids_helper` keeps its signature. In `case 10`, if flag + exact qualified cc + `n_tokens >= threshold`, launch candidate; else call native `launch_mm_ids_helper<10>`.

Marker:
`BIGCHERRY_PATCH_HIT patch=1348_moe_ids_multiwarp path=top10 cc=<cc> experts=<n_experts> tokens=<n_tokens> warps=<n>`.

Patch edits:
- `ids-mw-includes-flag`: `mmid.cu`, insert-after includes for cached getenv/trace support.
- `ids-mw-kernel`: insert-before the **post-1281** public `void ggml_cuda_launch_mm_ids_helper(...)` definition.
- `ids-mw-case10`: replace only the composed `case 10: launch_mm_ids_helper<10>(...); break;` block.
No `mmid.cuh` or `mmq.cu` change.

#### 1349 router split-K

Flag: `BIGCHERRY_MOE_ROUTER_SPLITK=1`, default `0`.

Marker in `build_moe_ffn`: after
`logits = build_lora_mm(gate_inp, cur);`
set i32 op-param[6] to a unique router magic only when `logits->op == GGML_OP_MUL_MAT && logits->src[0] == gate_inp`. Leave op-param[0] precision and [1] hint untouched.

CUDA helpers:
`static bool bc_is_moe_router_splitk(const ggml_tensor * dst);`
`static bool bc_should_use_moe_router_splitk(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);`
`static void bc_cuda_moe_router_splitk(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);`

V1 kernel pair:
`bc_moe_router_splitk_partial(..., float * partial, int M, int N, int K, int split_k)`
and
`bc_moe_router_splitk_reduce(const float * partial, float * dst, int M, int N, int split_k)`.
Reduction order is partition 0..split_k-1. Start with `split_k=8` only after Gate 0 proves the observed K divisibility/alignment. Workspace is `split_k*M*N*sizeof(float)`; record peak and fail closed above a measured cap.

Dispatch: insert before generic MMVF in `ggml_cuda_mul_mat`; require marker, F32/F32/F32, exact qualified cc, supported strides, and measured `M/N/K` window. Do not identify by M==512 alone.

Marker:
`BIGCHERRY_PATCH_HIT patch=1349_moe_router_splitk path=router cc=<cc> m=<M> n=<N> k=<K> split=8`.

Patch edits:
- `router-mark`: `src/llama-graph.cpp`, insert-after exact `logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]`.
- `router-kernels`: dedicated CUDA source if build integration is clean; otherwise insert private implementation adjacent to existing matmul dispatch support, not inside rocBLAS implementation.
- `router-build-source`: build-list insert only if a new compilation unit is used.
- `router-dispatch`: `ggml-cuda.cu`, insert-before the ordinary MMVF branch.
- `router-trace`: contained in the dispatch/launcher, one-shot per representative shape.

### 5. Gate 0 and lightweight validation

Run current production composition with f16 KV, ub512, target three-card topology using `tools/lab/flash-next/long-ctx-profile.sh ... prefillprof` at 8K and 24K. Parse `kernel_trace.csv` per Agent_Id.

**1348 gate:** sum `mm_ids_helper<10>` duration and calls on each participating rank, separately for normal row-split and `BIGCHERRY_MOE_EP=1` range mode if that mode is in the production recipe. Positive only if helper time is **>=0.50% of critical-rank prefill wall time or >=1.0 ms per 512-token ubatch**, and the top-10 specialization accounts for >=90% of helper calls. Otherwise close 1348.

**1349 gate:** add a no-code-change host dispatch census at `ggml_cuda_mul_mat`/debug build that records graph callback name, src types, marker candidate, local M/N/K, chosen backend and per-device call count; correlate the 48-layer `ffn_moe_logits` events with rocprof. Positive only if the production router is F32/F32/F32 and its aggregate GPU time is **>=0.50% of critical-rank prefill wall time or >=1.0 ms per 512-token ubatch**. If src0 is quantized, or opportunity is below threshold, close 1349.

Validation:
- 1348 offline mechanics + patch-lint; old/new standalone helper comparison over token counts 1,2,31,32,33,128,511,512 and full/range ids; byte-identical `ids_src1`, `ids_dst`, `expert_bounds`; include QFP30 `-1` inverse-map sentinel preinitialization.
- 1349 offline mechanics + patch-lint; marker only on unwrapped router MUL_MAT; backend-op score tolerance; production and adversarial near-tie top-k-id probes.
- One-shot activation marker required for each subject arm.
- Fully separated ABBA through `tools/lab/flash-next/queue-env-ab.sh`: helper alone, router alone, combined only after both win.
- Identity arm: `GGML_CUDA_DISABLE_FUSION=1` on A and B. Fusion-on runs are probes required by FKE01.
- Record CUDA-graph recapture count/behavior, workspace peak, per-agent kernel time, prefill t/s, greedy md5. No q4 KV.

### 6. Verdict

**1348: GO AFTER GATE 0.** Exact-output boundary and current one-warp serialization make it the stronger half. Expected production prefill gain if positive: **+0.3% to +1.2%**.

**1349: GO AFTER GATE 0, lower priority.** The router is explicitly identifiable but its type/time are not yet proven and split-K risks top-k changes plus workspace/fusion-layout effects. Expected gain if positive: **+0.1% to +0.6%**.

Combined QFP36 expectation on this topology if both gates pass: **+0.4% to +1.5%**, not the external reported gain.
- 2026-10-07T06:35:55.547181+00:00 (updated-by): Updated: section:notes

## Ledger-events



- chg_20261007_073613_flash-next-prefill-is-about-2_3220
- 2026-10-07T07:36:17.133419+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-08T09:46:17.213927+00:00 (updated-by): Updated: section:notes
- 2026-10-08T15:22:57.433370+00:00 (updated-by): Updated: section:notes
- 2026-10-08T15:59:24.829040+00:00 (updated-by): Updated: section:notes
- 2026-10-09T04:16:40.210507+00:00 (updated-by): Updated: section:notes
- 2026-10-09T04:17:11.105956+00:00 (updated-by): Updated: section:notes
- chg_20261010_015740_optional-faster-moe-router-mat_2362
- 2026-10-10T01:57:44.055374+00:00 (updated-by): Updated: section:ledger-events
- chg_20261010_114843_flash-next-prompt-processing-i_7051
- 2026-10-10T11:48:51.010778+00:00 (updated-by): Updated: section:ledger-events
