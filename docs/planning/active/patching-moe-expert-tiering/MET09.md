---
id: MET09
order: 9
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-06T14:53:14.519034+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: S
---

# Why expert split, row split and expert order give different numerics (top-k ties, range MMQ path)

## Description

Expert split vs row split differ by TV 0.055-0.113 on next-token probes (top-1 19-23 of 24); changing only the expert partition changes the greedy text and MTP acceptance (55% to 77%); the row split on a consistently reordered file gives a different text than on the original. GPT analysis (req_6aaf0e15985e4ba0): the two splits are different arithmetic groupings (row split sums down-projection row slices; expert split sums whole experts after weighting), all in F32; 1281 disables MMQ's dedup_bcast for range nodes so the splits take different activation-packing paths (inferred strongest candidate); CUDA bitonic argsort has no expert-id tie-break (ggml-cuda/argsort.cu), and the final expert sum runs in top-k slot order, so an expert permutation can change selection among equal scores and the sum order. Measured: forcing the plain AllReduce (GGML_CUDA_ALLREDUCE=none) changes nothing (same md5s, TV 0.1132) - a compressed cross-card sum is ruled out.

## Steps

1. Single-token batches (-b 1 -ub 1) so MUL_MAT_ID stays on MMVQ: if the E/R difference collapses, the range-MMQ packing path dominates. 2. Dump selected experts + weights (1338 trace output, or a small dump) for original and reordered files under the row split, inverse-map ids, compare at the first divergent layer / token: different top-k means ties / router, identical top-k means summation order. 3. If ties: add a deterministic expert-id tie-break to the top-k (or sort the selected ids before the sum) behind a flag and re-measure identity across the reorder. 4. If the MMQ path: make range nodes use the same dedup path where valid.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

patches/1281_moe_mul_mat_id_range, vendor ggml-cuda/argsort.cu + mmq.cu (read only), tools/lab/flash-next/flash-fidelity.sh

## Validation

A stated cause with one experiment that removes the difference; if a fix follows, identical text for the row split on original vs reordered file.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Matters because decode comparisons between expert layouts are dominated by acceptance changes caused by these numeric differences (MET08).

## Change Log

- 2026-10-06T14:53:14.519034+00:00 (created-by): Created by agent
