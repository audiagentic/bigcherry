---
id: QFP13
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T17:26:47.771591+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Decode kernel-count reduction for Flash-Next (launch-gap bound: ~3.5 us gap per ~1.7 us kernel)

## Description

Inter-kernel gap census on production profile v2 decode (flashnext-v2-profile/d8192, rocprofv3): median gap between consecutive in-graph kernels 3.1 us (XTX) / 3.7 us (R9700), p99 5-6.5 us; median kernel duration only 1.6-1.8 us. ~1,300 kernels per generated token per tensor-split GPU, so gaps cost ~4.7-5.0 ms/token per GPU vs ~7.6-7.9 ms/token of kernel time - this is the bulk of the measured ~58-61% GPU idle. The decode is launch/dispatch-gap bound (HIP graphs are already on). Every kernel removed saves ~3-5 us. Per-token counts (rank-census.py): elementwise ~370-400, quantize ~180-200, other ~180, mmvq ~150-166, norm/rope ~80-95, get/set_rows/cpy ~55-65, qsa/topk ~60, AllReduce produce/consume ~60.

## Steps

1. Fusion census: from the kernel trace, list the most frequent adjacent kernel pairs/triples per GPU (name n-grams within a segment) and estimate removable launches. 2. Map each top pattern to an existing fusion item (PRBE37 GEMV+activation, PRBE38 act x mul, PRBE39 GEMV->view->residual, PRBE40 paired K/V, PRBE05/1235 q8_1 activation reuse, RNX04 router/hyper-connection, RNX08 norm+rope/transpose, RNX10 shared expert) or a new one. 3. Implement in order of launches removed per token. 4. Re-run the census after each; ms/step ABBA.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Kernels/token down (rank-census.py), gap time/token down, ms/step ABBA on profile v2, greedy identical.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

This re-ranks the fusion backlog: individually each looked like 1-3%, but together they attack the single largest measured cost (~4.7 ms/token of gaps). Also check whether HIP graph launch latency itself can be reduced (e.g. hipGraph dispatch tuning / fewer nodes per graph). Earlier: AMD-Ecosystem Set 4 MMV epilogue fusions cut 1263->1103 launches/token for +1.4% tg on a different model. Related: QFP06 (graph working set), QFP11 (AR boundaries are only ~25 us), QFP09 (split balance at diminishing returns).

2026-10-04 fusion census (XTX0, ~10K, per generated token): top kernels quantize_q8_1 183, mul_mat_vec_q 154, unary_op 102, k_bin_bcast 98, scale_f32 98, mul_mat_vec_f 82, rms_norm 79, __amd_rocclr_copyBufferRectAligned 55 + __amd_rocclr_copyBuffer 53 (HIP runtime copy kernels inside the graph: 108/token, origin unknown - find via apitrace), unary_gated 30, dsv4_hc_pre/post 30 each, cpu_root produce/consume 30 each, mmvq_moe 30. Top adjacent pairs: quantize_q8_1->mul_mat_vec_q 154 (dedupe/reuse q8_1 across consumers of the same activation, PRBE05/1235, or F32-act), scale_f32->unary_op 60, copyBufferRect<->copyBuffer 80, rms_norm->quantize 41 (PRBE06), mmvq->scale 30 / mmvf->scale 30 (PRBE37/38 epilogues), scale->dsv4_hc_post 30, mmvq->dsv4_hc_pre 30 (RNX04), unary_gated->quantize 30. Next: apitrace to attribute the 108 runtime copies; then quantize dedupe.

## Change Log

- 2026-10-03T17:26:47.771591+00:00 (created-by): Created by agent
- 2026-10-03T17:27:37.409987+00:00 (updated-by): Updated: section:notes
