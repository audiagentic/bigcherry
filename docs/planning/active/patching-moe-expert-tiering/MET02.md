---
id: MET02
order: 2
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-02T04:44:50.160922+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# 1281 moe_mul_mat_id_range: range-aware MUL_MAT_ID primitive (CPU + HIP)

## Description

Generic ggml primitive: ggml_mul_mat_id_range(ctx, weights[K,M,n_local], act, ids, id_base). For each selected id g: local=g-id_base; if 0<=local<ne02 compute weights[...,local]*input, else output lane is zero and the GEMM/GEMV is skipped. Implemented as an op-param variant of GGML_OP_MUL_MAT_ID; plain ggml_mul_mat_id keeps strict 0<=id<ne02. No model behaviour change.

## Steps

1. ggml.h/ggml.c: ggml_mul_mat_id_range + op param.
2. ggml-cpu: range-base handling, inactive lane zero.
3. ggml-cuda/HIP: same, covering Q6_K, IQ4_XS, Q8_0, F32 MMVQ/MMQ/MUL_MAT_ID paths; zero then skip (never map inactive lanes to expert 0).
4. test-backend-ops cases.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

patches/1281_moe_mul_mat_id_range/ (ggml/include/ggml.h, ggml/src/ggml.c, ggml/src/ggml-cpu/ops.cpp, ggml/src/ggml-cuda/mmid/mmq/mmvq)

## Validation

test-backend-ops: normal MUL_MAT_ID unchanged; range containing all ids == baseline; out-of-range lanes exactly zero; mixed 10 ids across ranges; F32/Q8_0/Q6_K/IQ4_XS on gfx1100/gfx1201/gfx1030 and CPU.

## Effort & Risk

Touches hot HIP MoE kernels; must not regress normal MUL_MAT_ID (A/B on Flash-Next IQ4_XS baseline).

## Standards



## Acceptance Criteria



## Notes

2026-10-05, folded in from BCOP15 (audit backfill) - compact active-lane dispatch: range-aware zero/skip stays dispatch-heavy when a tier owns few routed experts. Follow-up once range semantics are confirmed on the live branch: (1) profile 0/10/25/50/75/100% local ownership across ub1-512 on gfx1100/gfx1201; (2) prototype global->local translation plus a compact (output lane, local expert) list in the existing backend workspace, scattering results back to the original lanes; (3) keep current precision selection and upstream MMQ tail/allocation safety; (4) gate through the existing HIP autotune, no second dispatcher or router. Promote only for >=5% end-to-end or >=10% MUL_MAT_ID kernel gain on a <=25%-local lane with no dense lane >2% slower.

## Change Log

- 2026-10-02T04:44:50.160922+00:00 (created-by): Created by agent
- 2026-10-05T04:54:43.149052+00:00 (updated-by): Updated: section:notes
