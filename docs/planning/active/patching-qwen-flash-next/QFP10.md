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

## Change Log

- 2026-10-03T15:22:48.527804+00:00 (created-by): Created by agent
- 2026-10-05: Reframed around full expert-service cost, direct-F32 IQ qualification, and merged #29941 MoE MMQ safety invariant; consolidated dispatch ownership and added explicit promotion gates.
- 2026-10-05: Added fresh #29953 allocation/launch tile-consistency gate; consolidated MMQ config ownership and added J-boundary HIP regression matrix.
