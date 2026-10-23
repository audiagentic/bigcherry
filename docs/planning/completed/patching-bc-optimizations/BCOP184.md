---
id: BCOP184
order: 184
plan: patching-bc-optimizations
state: completed
created-at: '2026-10-13T11:11:47Z'
created-by: agent
priority: P2
work: S
---

# PRBE20: per-type MMVQ width fence before determinism patching

## Description

**Discovery:** b11552 and current llama.cpp `mmvq.cu::get_mmvq_mmid_max_batch_rdna3` cap Q4_K/Q5_K/Q6_K MoE MMVQ at width 4, while `ggml_cuda_mul_mat_id` enforces the per-type cap. Patch 1210 normalizes `calc_nwarps` through width 8 but not this eligibility check. Historical gfx1100 width-5 early logit divergence (and separate late width-2/3/4 divergence) remains real; the route is **not yet causally attributed** to the tested GGUF. The former “global MMVQ max=8 rules out a width-5 route change” argument is invalid.

**Owners / consolidation:** PRBE20 is sole authoritative design; patch 1210's existing validation producer and lab bisection fixture own checks. README/SUMMARY corrected from obsolete two-hunk scope. No new dispatcher, graph callback framework, telemetry, allocator, or benchmark queue. PRBE52 adaptive MTP is separately rejected; recent Radiance/RAD01/1340/release work is excluded and untouched.

**Action:** inspect actual GGUF quant types and `MUL_MAT_ID` per-type width eligibility; if applicable, one disposable matched dispatch/first-divergent-op diagnostic with pinned `cb_eval` (diagnostic sync only), then classify `MMVQ_BOUNDARY_CONFIRMED`, `OTHER_OP_CONFIRMED`, `NOT_REPRODUCED`, or `UNATTRIBUTED`. Keep early width-5 and late width-2/3/4 mechanisms separate. Do not alter 1210 or queue GPUs before attribution.

**Terminal gate:** no eligible boundary or no first-op causality => close this hypothesis, no code. Proven causal route => one opt-in fixed-route discriminator with full-vocabulary/KLD/greedy/MTP correctness, multi-request/replay, >=4 independent paired sessions/architecture, <=1% decode control regression, and positive end-to-end MTP gain; otherwise reject. PRBE20 remains pending; BCOP184 is terminal as an audit/disposition record.

**Evidence:** actual-source checks 14/14; source-shaped host selector 9/9 and Python compilation; historical 2026-09-28 gfx1100 raw-logit mismatch. No new GPU benchmark, repository pytest, HIP build, patch edit, or speedup. [vLLM #48391](https://github.com/vllm-project/vllm/pull/48391) is a verified fixed-reduction-order precedent, not an AMD port. [llama.cpp #16016](https://github.com/ggml-org/llama.cpp/pull/16016) remains an open opt-in determinism proposal.

## Steps


## Detailed Solution & Technical Design


## Code Samples & Guidance


## Files


## Validation



Source-static and host-only checks recorded in this item's disposition/evidence text; no hardware run was required for this closure.

## Effort & Risk


## Standards


## Acceptance Criteria



The disposition and ownership decisions described above are recorded; no further implementation is owed by this item.

## Notes


## Change Log

- 2026-10-23: state 'done' (invalid) corrected to completed
