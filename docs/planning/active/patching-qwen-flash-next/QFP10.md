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

### Upstream #29941 invariant

Merged upstream #29941 demonstrates a layout trap directly relevant to this work. MMQ's temporary Q8_1 padding needs enough extra columns for maximum tile width `J`. Dense tensors bound that from `src1->ne[1]`; `MUL_MAT_ID`/MoE tensors are laid out differently and require the physical-batch dimension `src1->ne[2]`. With many experts relative to ubatch, using the dense dimension can under-allocate and cause an illegal memory access. The release containing the fix is b11390 (`dd26678`, 2026-10-04).

Before carrying any local expert-kernel patch, diff its workspace/padding logic against b11390. Add a focused regression lane equivalent to upstream's lopsided shape (`Q4_0/F32`, width 512, 10 rows, 640 experts, 508 selected/layout dimension, 2560 output dimension) plus Flash-Next's real expert shape. Run under HIP memory checking where practical. This is a correctness prerequisite, not a performance lever.

## Consolidation / Ownership

- QFP10 owns Flash-Next IQ expert decode/MTP qualification and the decision between MMVQ/VDR and direct-F32 IQ paths.
- Generic quant kernels/traits and architecture dispatch remain in the existing 1273/PKC-style infrastructure; improvements proven generic should move there rather than remain model-specific.
- QFP17 owns prefill/QSA and its MoE-MMQ prefill follow-up; QFP10 must not duplicate prefill tiling work.
- Generic sparse `MUL_MAT_ID` dispatch/range work remains outside QFP10. Reuse it if present.
- #29941's padding invariant is upstream baseline behavior; do not create a BigCherry patch whose sole purpose is to duplicate a merged fix.

## Files

Expected llama.cpp touch points after current-pin inspection:
- `ggml/src/ggml-cuda/mmvq.cu` / quant trait helpers used by HIP;
- `ggml/src/ggml-cuda/mmq.cu` / `mmq.cuh` only to verify #29941-compatible workspace sizing;
- `MUL_MAT_ID` MoE launch/selector path containing `mul_mat_vec_q_moe_launch` / `mul_mat_vec_q_moe`;
- existing BigCherry architecture tuning table used by 1273.

Do not add a new patch package until the microbench establishes which path wins.

## Validation

Matrix: gfx1100 + gfx1201; ncols `{1,2,4,8}`; routed-token buckets `{1,2,4,8,16}`; confirmed expert types plus one control quant. Record total service and component timing, temp bytes, launches, occupancy, registers/spills.

Model ABBA: fixed prompt at representative 10K/80K context, MTP on/off controls, profile-v2/current composed profile, warm clocks. Record tg128/tg512, MTP acceptance, accepted tokens/step, ms/step and per-rank arrival time. Greedy output identity is required; add raw-logit/KLD comparison for direct-F32 accumulation changes.

Safety: lopsided `MUL_MAT_ID` regression shape derived from #29941 plus real Flash-Next shape; repeated runs on both AMD architectures; no OOB/illegal access.

## Effort & Risk

Medium. The direct-F32 IQ path can reduce launches/traffic but may increase register pressure or repeat activation loads enough to lose against Q8_1 reuse, especially as ncols/routed rows grow. IQ formats also have nontrivial lookup/index layouts, so coverage proofs are mandatory. The selector must preserve MMQ for shapes where batching amortizes Q8_1 conversion.

## Standards

Fail closed to existing MMVQ/MMQ for unsupported type/layout/shape. No model-name dispatch. No output-tolerance widening to hide arithmetic errors. Keep architecture thresholds in the existing tuning facility. Current upstream MMQ padding semantics are mandatory.

## Acceptance Criteria

Promote only if one candidate improves total expert-service wall >=10% on the limiting rank and yields >=3% end-to-end decode/MTP throughput on at least one representative lane, with another representative lane non-negative within 1%; no lane may regress >2%. Direct-F32 promotion additionally requires no material acceptance-rate loss, greedy identity for the deterministic lane, acceptable raw-logit/KLD parity, and no spill/occupancy pathology. If activation conversion is <10% of expert-service wall, drop MMVDQ work and return to VDR-only tuning.

## Notes

Original RV4214 estimate was +1-3%, but it considered VDR rather than the full expert-service path. Treat that as an Amdahl warning, not expected performance.

Fresh upstream scan 2026-10-05: llama.cpp b11390 (released 2026-10-04) merged #29941, `CUDA: fix MMQ memory fault if n_expert >> n_ubatch`. The bug was incorrect Q8_1 temporary padding under MoE tensor layout and is directly relevant to any local MMQ workspace code. #29927 (`__builtin_amdgcn_perm` q1_0 unpack) and #29910 (Q2_K VGPR spills) remain generic quant-kernel candidates, not QFP10 ownership. #29943 selective expert-copy belongs to expert residency/materialization work.

## Change Log

- 2026-10-03T15:22:48.527804+00:00 (created-by): Created by agent
- 2026-10-05: Reframed around full expert-service cost, direct-F32 IQ qualification, and merged #29941 MoE MMQ safety invariant; consolidated dispatch ownership and added explicit promotion gates.
