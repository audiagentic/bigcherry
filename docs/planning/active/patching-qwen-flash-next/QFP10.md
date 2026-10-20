---
id: QFP10
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:22:48.527804+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Flash-Next IQ expert decode/MTP path: remove activation tax before VDR micro-tuning


## 2026-10-20 BCOP339 — gfx1100 Q4_K MoE 32-element dot: reject direct fork port, gate residual

**Disposition: HOLD / no patch or hardware queue.** This is a *typed Q4_K routed-expert comparator* under QFP10's existing whole-expert-service qualification, **not** a replacement for the IQ4_XS/IQ3_XXS activation-tax work below. It applies only if a concrete BigCherry Flash-Next/other MoE GGUF and composed HIP binary actually execute Q4_K `MUL_MAT_ID` on gfx1100; the external Qwen3-30B-A3B result does not prove that route for BigCherry. Do not change QFP10's IQ dispatch, the current 1273 IQ patch, 0600 geometry, 1237/1265 MMQ block maps or the active RAD14/RAD13 worker/MTP lanes.

### Pinned source, dispatch and work accounting

- Pinned llama.cpp **b11552** (`23b0202a189c44a54625aadcb37a946dd1d6278d`) `ggml/src/ggml-cuda/mmvq.cu::get_vec_dot_q_cuda` (line 53) maps Q4_K to `vec_dot_q4_K_q8_1`; `get_vdr_mmvq` (line 82) reads `VDR_Q4_K_Q8_1_MMVQ=2` from `vecdotq.cuh:504`. The Q4_K dot at `vecdotq.cuh:918-965` consumes 16 weight values per vector-dot invocation, with separate integer dot/scaling for two 8-value chunks. **Neither** `mmvq-vecdot.cuh`, `vec_dot_q4_K_q8_1_vdr4` nor `vec_dot_q4_K_q8_1_vdr4_dense` exists in this pin. A one-line port of the fork selector cannot compile.
- `ggml_cuda_mul_mat_vec_q` (mmvq.cu:1434+) allocates Q8_1 temporary storage from `ctx.pool()`, calls `quantize_row_q8_1_cuda` on the same stream and then dispatches. `mul_mat_vec_q_switch_ncols_dst` (around 1170) sends **`ids && ncols_dst>1`** to `mul_mat_vec_q_moe_launch` before the generic dense-width switch. The dedicated `mul_mat_vec_q_moe<type,2,has_fusion>` has fixed `rows_per_block=2`, grid `(ceil(nrows_x/2), expert_channels [+ shared expert])`, block `(physical_warp_size,ncols_dst)`, and `get_vec_dot_q_cuda(type)` in its kernel body. The gfx1100 Q4_K MMVQ `MUL_MAT_ID` cap is **4**: width 5+ can select another operator and must not be counted as this kernel. Width 1 is the ordinary MMVQ path, not this multi-token MoE launch.
- External [RDNA fork #167](https://github.com/stew675/llama-cpp-rdna-boosts/pull/167), merged downstream **2026-10-17**, builds on [#162](https://github.com/stew675/llama-cpp-rdna-boosts/pull/162)'s new VDR4 helper. It combines each 16-value Q4_K sub-block's integer DP4A chain before one F32 scale/convert/FMA, and selects that helper for gfx1100 MoE; gfx1201 is separately **build-time opt-in**, default off. This changes FP32 summation order. It does **not** remove expert rows, K elements, Q8_1 conversion, launches, required global loads/stores, or Q8_1 scratch; it is an instruction/codegen candidate. For a K256 block the idealized per-lane VDR2/VDR4 partition is 16×16 versus 8×32 elements, both covering 256. Actual VALU/VMEM, VGPR/SGPR, LDS, spills, wave32/64, occupancy and bandwidth require ISA/rocprof evidence, not this arithmetic count. Fork gfx1201 reports 70→67 VGPR (8-row fused) and 57→54 (2-row) with zero scratch; **not BigCherry codegen**.
- The fork's gfx1201 build flag needs host-preprocessor visibility to avoid a ccache hit on unchanged HIP device objects. Do not introduce a new BigCherry runtime knob or second selector. If ever admitted, the existing patch/architecture capability system must own the typed compile-time specialization, with the pinned VDR2 native fallback for every unqualified type/arch/width. Keep fused gate/up/GLU, expert-ID, per-rank output and graph lifetimes intact.

### Evidence, cheap discriminator and terminal gate

**External only (not BigCherry):** on one gfx1100 WSL2 Qwen3-30B-A3B Q4_K_M, #167 reports Q4_K fused gate/up/GLU 24.8→22.3 µs (−10%), Q4_K down 19.4→16.0 µs (−18%), untouched Q6_K ~18 µs, total GPU request time **4167→4087 ms (−1.92%)**, and alternating decode ~131.7→133.6 t/s (+1.4%, noisy). Greedy text **changes**; KLD 0.0044, top-token agreement 97.0% in a different r67 comparison. On gfx1201 opt-in: +0.7% decode, 2-row kernel −3.6%, 8-row neutral, greedy text changes; no gfx1201 KLD. None meets QFP10's existing >=3% E2E plus deterministic greedy-identity promotion contract. Do not present kernel-only −18% as E2E gain.

**Host-only discriminator actually run:** 50,000 random 16-weight Q4_K/Q8 signed-dot fixtures had exact `dot8a+dot8b==dot16` and bounded integer products; **9,198/50,000** differed in F32 bits when two separately rounded scaled partials were combined versus one scaled combined dot (max absolute synthetic difference 0.0625). This is not a GGUF unpack, HIP ISA, KLD or model-output test; it demonstrates why integer equivalence does **not** establish bit-identical logits. No HIP build, GPU benchmark or BigCherry patch run occurred.

**Ownership and alternative engines:** QFP10 owns the typed MoE expert-service admission only; PRBE20 owns decode-vs-verify *bit-identity* diagnosis, PRBE111/1273 owns IQ VDR, PRBE47 Q6_K lowering, PRBE21's already-native MoE rpb2 remains retired, QFP44 MMQ and QFP37 Stream-K remain separate. No neighbour is edited. [vLLM GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin) has a separate GGUF MoE/Triton layout with routed-token padding, not a direct HIP VDR4 port; [SGLang AMD support](https://sgl-project.github.io/platforms/amd_gpu.html) does not qualify GGUF on AMD. Do not copy either dispatch or benchmark result.

**Bounded decision sequence (read-only first):**
1. Confirm no independent same-capability work/queued hardware in the prior 12 h. On an *idle* hardware lane, only reuse existing BigCherry trace/evidence to establish exact GGUF tensor type, source/patch hash, rank, `MUL_MAT_ID` vs MMQ route, width 1/2/4/5/8, routed expert count, fused/unfused state, K, activation-quantize/launch time and exposed E2E wall. No RAD14 isolation benchmark changes or overlapping run.
2. **Terminal reject** `NO_Q4K_MOE_MMVQ_ROUTE` if no gfx1100 Q4_K width 2–4 route; `BELOW_CEILING` if even removing all exposed candidate kernel latency is <2.9126% of E2E; `NUMERIC_CONTRACT_FAIL` if the required greedy/bit identity cannot be preserved. The external −1.92% aggregate GPU time is a warning, not a BigCherry upper bound.
3. Only if *all* gates survive: prototype on the pinned source, **not** a direct #167 patch. Port the helper and selector together; preserve host/device VDR agreement, 32-element scale/min index and K256/tail coverage, `rows_per_block=2`, graph allocation and fallback. Compile gfx1100 baseline/candidate with identical flags; check generated DP4A counts, VGPR/SGPR, scratch/spills, LDS, wave occupancy and physical bytes. gfx1201 stays stock unless independently admitted; gfx1030/Q6_K/dense Q4_K/IQ are controls.
4. Require `test-backend-ops -o MUL_MAT_ID` Q4_K, adversarial expert IDs/ties/empty experts, widths 1/2/4/5/8, multi-ubatch and multi-request same-process, long-context graph replay/reallocation, full logits/KLD/greedy/MTP acceptance and *unchanged physical work*. Then >=4 independent sessions/arch, >=10 paired ABBA rounds at 8K/24K/98K with CI95-low >=3% E2E and <=1% control regression. Otherwise `NO_GAIN`/`REGRESSION` and retire the candidate without a production patch.

## Description

1273_iq_mmvq_rdna_tuning tunes VDR/nwarps for IQ4_XS/IQ3_XXS only. The existing QFP10 proposal extended that selector to IQ3_S/IQ4_NL, but this is too narrow: Flash-Next MTP verification uses `MUL_MAT_ID` with multiple destination columns and enters `mul_mat_vec_q_moe_launch -> mul_mat_vec_q_moe`, while the current path can also pay F32->Q8_1 activation conversion and temporary-buffer/launch costs before the expert dot product. Optimizing VDR first risks accelerating a sub-kernel that is no longer the dominant expert service cost.

New priority: measure and remove the activation/materialization tax for the actual routed-expert types and MTP shapes, then tune VDR only on the surviving hot path. Keep decode (`ncols=1`) and MTP (`ncols=4`, plus 2/8 controls) as separate selector lanes.

## Steps

1. Confirm routed expert tensor types from the exact Flash-Next GGUF; record expert width, expert count, top-k and physical `ncols` for decode/MTP. Do not assume IQ3_S/IQ4_NL.
2. Instrument expert service into activation conversion/Q8_1 materialization, MMVQ/MMQ kernel, routing/grouping/scatter, launch/wait and temporary bytes. Measure gfx1100/gfx1201 at ncols `{1,2,4,8}` and routed-token buckets `{1,2,4,8,16}`.
3. Prototype an IQ MMVDQ-style path only for confirmed types: consume F32 activations directly and dequantize expert weights in the dot path, avoiding the standalone Q8_1 activation conversion/temp when profitable. Reuse the existing CUDA/HIP quant traits and architecture selector; do not add a Flash-Next-specific dispatcher.
4. Compare three arms: stock MMVQ/MoE path, VDR1-only extension, and direct-F32/MMVDQ-style path. Select by measured total expert-service wall time, not kernel-only throughput.
5. Only after the winning activation path is known, thread `iq_vdr` through `mul_mat_vec_q_moe<>`/launch and add IQ3_S/IQ4_NL variants where the confirmed GGUF needs them. Tune nwarps/VDR independently on gfx1100/gfx1201.
6. Rebase safety check on llama.cpp b11390+ / merge `dd266785`: upstream #29941 fixes MMQ Q8_1 temporary padding for MoE by bounding the tile-width padding from `src1->ne[2]` rather than dense-layout `src1->ne[1]`. Any BigCherry MMQ/MoE branch or copied workspace calculation must inherit this invariant before performance qualification.
7. Track upstream #29953 before any MMQ qualification. #29941 still leaves allocation and execution choosing tile width through different rounding paths. #29953 computes `J_best` once, before allocating Q8_1 scratch, and passes that exact width to the launch selector. Rebase/cherry-pick the final upstream fix (or reproduce the invariant locally for testing) before trusting MMQ memory or performance results.

## Detailed Solution & Technical Design

### Measure the whole expert service

For each routed expert invocation collect:
- `activation_quant_us`: F32 -> Q8_1 conversion/materialization;
- `expert_kernel_us`: MMVQ/MMQ body;
- `routing_us` / `scatter_us`;
- Q8_1 temporary bytes and allocation high-water mark;
- launches, stream waits, VGPR/SGPR/spills and occupancy;
- total critical-path expert service time.

The optimization decision is `min(total_service_us)`, not `max(vec_dot throughput)`. A VDR variant is rejected when its kernel win is consumed by activation conversion or occupancy loss.

### Direct-F32 IQ lane

Use the same design principle as upstream AMD MMVDQ work for K-quants: for small-row decode/MTP shapes, load F32 activation values directly while unpacking/dequantizing quantized expert weights in-register. This removes a separate quantize launch, synchronization point and Q8_1 temporary traffic. Do not mechanically port K-quant templates to IQ layouts: implement only after confirming each IQ type's block/index tables and proving full element coverage.

Pseudo-selector boundary:

```cpp
const bool small_moe = ncols <= iq_mmvdq_max_cols && routed_rows <= iq_mmvdq_max_rows;
if (arch_supports_iq_mmvdq && small_moe && iq_mmvdq_supported(type)) {
    launch_mul_mat_vec_iq_f32_moe(...);
} else {
    launch_existing_mmvq_or_mmq(...);
}
```

Thresholds belong in the existing architecture/autotune mechanism. QFP10 must not create a model-name gate or a second tuning registry.

### Upstream MMQ allocation/dispatch invariant (#29941 + #29953)

Merged #29941 fixed a MoE layout error in Q8_1 scratch sizing, but fresh upstream #29953 identifies a second correctness/design defect: allocation rounded the candidate MMQ column count down while runtime tile selection rounded it up. That can allocate scratch for a narrower `J` than the kernel actually launches.

#29953 removes `ggml_cuda_mmq_get_J_max` as an independent allocation estimator. It computes the viable `J_best` once using the same `ggml_cuda_mmq_get_config(...)`, shared-memory limit and RDNA3/RDNA4 MoE average-columns heuristic used for execution, allocates padding for exactly `J_best`, and passes `J_best` in `mmq_args` to `mul_mat_q_switch_J`. It also deduplicates fallback selection through `ggml_cuda_mmq_needs_fallback()`.

This is the correct ownership model for BigCherry too: **one selector result must drive both resource sizing and launch**. Do not add a second QFP10 tile-width estimator, padding formula, or architecture table. If BigCherry needs different RDNA thresholds, extend the shared selector and consume its returned config everywhere.

Required regression matrix before tuning: column counts immediately below/at/above each supported J boundary (7/8/9, 15/16/17, 31/32/33, 63/64/65, 127/128/129), dense and `MUL_MAT_ID`, plus Flash-Next real routed shapes. For each case assert `allocated_J >= launched_J` (ideally equality), no pool overwrite/illegal access, and identical fallback/config choice between sizing and launch. Run gfx1100 and gfx1201. This gate precedes performance comparisons because a falsely small scratch allocation can make a fast arm invalid.

### Consolidation / Ownership

- QFP10 owns Flash-Next IQ expert decode/MTP qualification and the decision between MMVQ/VDR and direct-F32 IQ paths.
- Generic MMQ config selection, scratch sizing and fallback selection are upstream/PKC infrastructure. QFP10 consumes one resolved config; it must not fork these policies.
- Generic quant kernels/traits and architecture dispatch remain in the existing 1273/PKC-style infrastructure; improvements proven generic should move there rather than remain model-specific.
- QFP17 owns prefill/QSA and its MoE-MMQ prefill follow-up; QFP10 must not duplicate prefill tiling work.
- Generic sparse `MUL_MAT_ID` dispatch/range work remains outside QFP10. Reuse it if present.
- #29941/#29953 are baseline correctness behavior, not independent BigCherry performance patches.

## Code Samples & Guidance



## Files

Expected llama.cpp touch points after current-pin inspection:
- `ggml/src/ggml-cuda/mmvq.cu` / quant trait helpers used by HIP;
- `ggml/src/ggml-cuda/mmq.cu` / `mmq.cuh` to verify #29941/#29953-compatible single-source config/scratch sizing;
- `MUL_MAT_ID` MoE launch/selector path containing `mul_mat_vec_q_moe_launch` / `mul_mat_vec_q_moe`;
- existing BigCherry architecture tuning table used by 1273.

Do not add a new patch package until the microbench establishes which path wins.

## Validation

Matrix: gfx1100 + gfx1201; ncols `{1,2,4,8}`; routed-token buckets `{1,2,4,8,16}`; confirmed expert types plus one control quant. Record total service and component timing, temp bytes, launches, occupancy, registers/spills.

MMQ safety matrix: J-boundary columns `{7,8,9,15,16,17,31,32,33,63,64,65,127,128,129}`, dense + MoE layouts, #29941 lopsided expert shape and real Flash-Next shapes. Instrument selected J at allocation and launch and require equality. Repeated HIP runs must show no OOB/illegal access.

Model ABBA: fixed prompt at representative 10K/80K context, MTP on/off controls, profile-v2/current composed profile, warm clocks. Record tg128/tg512, MTP acceptance, accepted tokens/step, ms/step and per-rank arrival time. Greedy output identity is required; add raw-logit/KLD comparison for direct-F32 accumulation changes.

## Effort & Risk

Medium. The direct-F32 IQ path can reduce launches/traffic but may increase register pressure or repeat activation loads enough to lose against Q8_1 reuse, especially as ncols/routed rows grow. IQ formats also have nontrivial lookup/index layouts, so coverage proofs are mandatory. The selector must preserve MMQ for shapes where batching amortizes Q8_1 conversion. Until #29953-equivalent single-source tile selection is present, MMQ results are correctness-blocked.

## Standards

Fail closed to existing MMVQ/MMQ for unsupported type/layout/shape. No model-name dispatch. No output-tolerance widening to hide arithmetic errors. Keep architecture thresholds in the existing tuning facility. One resolved MMQ config must determine scratch sizing and launch geometry.

## Acceptance Criteria

Correctness prerequisite: #29953-equivalent allocation/launch consistency passes the full J-boundary and MoE regression matrix on gfx1100/gfx1201. No performance candidate may be promoted before this passes.

Then promote only if one candidate improves total expert-service wall >=10% on the limiting rank and yields >=3% end-to-end decode/MTP throughput on at least one representative lane, with another representative lane non-negative within 1%; no lane may regress >2%. Direct-F32 promotion additionally requires no material acceptance-rate loss, greedy identity for the deterministic lane, acceptable raw-logit/KLD parity, and no spill/occupancy pathology. If activation conversion is <10% of expert-service wall, drop MMVDQ work and return to VDR-only tuning.

## Notes

Original RV4214 estimate was +1-3%, but it considered VDR rather than the full expert-service path. Treat that as an Amdahl warning, not expected performance.

2026-10-05 upstream scan: b11390 remains the latest release visible in the release feed. Fresh PR #29953 (`CUDA: fix inconsistent MMQ ncols selection`, opened 2026-10-04 15:22 UTC) is a direct follow-up to #29941. Its author reports master can round columns down for allocation but up for actual config selection. The proposed fix resolves J once and uses it for both Q8_1 padding and launch, while retaining the RDNA3/RDNA4 MoE average-token heuristic. This is a higher-priority prerequisite than new VDR/MMVDQ tuning because it affects validity of the measured path. #29948 FFN gate/up+GLU fusion remains dense-only; do not duplicate it into routed experts without routed-MMQ evidence.

2026-10-05 upstream state: #29941 (CUDA: fix MMQ memory fault if n_expert >> n_ubatch) MERGED 2026-10-04T12:10Z, about three hours after our pin commit (050439614, 09:24Z) - not in the pin. It is a correctness fix that matches Flash-Next's shape (512 experts), so it is a candidate for an upstream-fix backport patch (10xx) or the next pin bump. #29953 (MMQ out-of-bounds reads) is still OPEN.

## Change Log

- 2026-10-03T15:22:48.527804+00:00 (created-by): Created by agent
- 2026-10-05: Reframed around full expert-service cost, direct-F32 IQ qualification, and merged #29941 MoE MMQ safety invariant; consolidated dispatch ownership and added explicit promotion gates.
- 2026-10-05: Added fresh #29953 allocation/launch tile-consistency gate; consolidated MMQ config ownership and added J-boundary HIP regression matrix.
- 2026-10-05T04:56:48.444740+00:00 (updated-by): Updated: section:notes
