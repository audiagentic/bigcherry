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

External report: (a) the 512x512 F32 router GEMM as 8 K-chunks plus a sum, 158 -> 69 us; (b) a 16-warp routing helper sorting tokens per expert in two passes, 99 -> 25 us per call. Ours: ggml_cuda_launch_mm_ids_helper (plus 1281's range translate) and the generic F32 GEMM for ffn_gate_inp.

## Steps

1. Time the router GEMM and the ids helper per call at ub512. 2. Split-K router (float order changes: needs equivalence, not identity). 3. Multi-warp helper (must give the same grouping). 4. ABBA.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Helper output identical to the current helper on random and adversarial ids; router within tolerance and top-k selection unchanged on probes; prefill ABBA.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-07T00:39:40.654749+00:00 (created-by): Created by agent
