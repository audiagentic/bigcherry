---
id: PRBE26
order: 0
plan: patching-rdna-boost-experiments
state: in_progress
created-at: '2026-09-09T10:55:14.091235+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-MMV-001: Decode matvec without Q8_1 activation quantization

## Description

IMPLEMENTED-AS-PATCH, needs extension. Patch 1241_rd33_mmvq_q8_0_f32_decode (state=untested) adds a defaulted f32_act template parameter to mul_mat_vec_q + vec_dot_q8_0_f32, gated dense/Q8_0/ncols_dst==1/gfx1100/forced-only. Existing ncols_dst=1 evidence is correctness-positive but null (no gain). Real b11126 source (verified this batch): mmvq.cu's mul_mat_vec_q kernel is ALREADY templated on ncols_dst as a compile-time parameter (`template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k, bool halve_iters, int nwarps_explicit, int rows_per_block_explicit>`), and the fused-gate/GLU path (SWIGLU/GEGLU/SWIGLU_OAI/SWIGLU_CLAMP) already exists generically for any ncols_dst -- so widening f32_act to ncols_dst 5/6 (where real production MTP on Qwen3.8-27B dominates) is a template-instantiation/dispatch-site change, not a kernel rewrite.

## Steps

1. Verified via direct read of patches/1241_rd33_mmvq_q8_0_f32_decode/patch.py (CORRECTED, was not fully read before): the current launcher is NOT ncols-generic -- it hardcodes `constexpr int c_ncols_dst = 1;` (patch.py:288) and its gate is `if (!ids && src0->type == GGML_TYPE_Q8_0 && ne1 == 1 && !forced.requested())` (patch.py:322, anchor at patch.py:422). The description's 'forced-only' framing is inverted: `!forced.requested()` means the path is currently gated to the NON-forced (default) case, not forced-only. Merely widening the `ne1 == 1` gate would launch the wrong template specialization (still instantiated only for c_ncols_dst=1), so this alone would be incorrect.
2. Replace the hardcoded launcher with a `template<int N>` helper plus a runtime switch on `ne1` for N=1..8, each branch calling `calc_launch_params<type>(N, ...)` and `mul_mat_vec_q_switch_fusion<type, N, ..., f32_act=true>` (mirroring the existing per-N instantiation pattern already used for `mul_mat_vec_q_switch_fusion<type, c_ncols_dst, ...>` at patch.py:302).
3. Replace the gate anchor `if (!ids && src0->type == GGML_TYPE_Q8_0 && ne1 == 1 && !forced.requested())` with `if (!ids && src0->type == GGML_TYPE_Q8_0 && ne1 >= 1 && ne1 <= 8 && !forced.requested())`; keep forced/autotune requests on the legacy path unchanged.
4. Add requires=["0600_mmvq_geometry"] to patch 1241's patch.toml (CORRECTED -- currently missing despite anchoring on nwarps_explicit/rows_per_block_explicit signatures introduced by that patch, not raw b11126).
5. Add test-backend-ops tolerance-based correctness cases for ncols 1..8, including fused-gate/GLU combinations.
6. Confirm forced-candidate and non-target (MoE, non-Q8_0, non-gfx1100) cases remain native/unaffected.
7. Design and run an INTERLEAVED paired A/B on real production Qwen3.8-27B MTP where ncols_dst=5/6 dominates.
8. If the widened path still shows no statistically significant gain at ncols=5/6, close as a 'validated null' with the widened-shape coverage explicitly documented.

## Detailed Solution & Technical Design

No new kernel/device-code design needed -- vec_dot_q8_0_f32 and the surrounding kernel body are ncols_dst-generic already (the fused-gate/GLU switch block at mmvq.cu ~lines 845-870, verified this batch, branches on `active_glu` not on ncols_dst). The change is entirely at the host-side dispatch/instantiation layer: which (type, ncols_dst, f32_act) combinations get compiled and which one is selected at runtime for a given batch size. This should follow the same catalog-driven instantiation approach identified for PRBE21/PRBE22 (tools/bigcherry/tuning/catalog.py) if patch 1241 is catalog-integrated, or a direct Edit widening the existing forced-dispatch condition if it is not (must be confirmed by reading patch 1241's patch.py, which was not fully read this batch beyond patch.toml/SUMMARY.md).

## Code Samples & Guidance

Real b11126 anchor (verified), ggml/src/ggml-cuda/mmvq.cu, generic fused-gate switch (ncols_dst-independent):
```
switch (active_glu) {
    case GGML_GLU_OP_SWIGLU: result *= ggml_cuda_op_silu_single(gate_value); break;
    case GGML_GLU_OP_GEGLU: result *= ggml_cuda_op_gelu_single(gate_value); break;
    case GGML_GLU_OP_SWIGLU_OAI: result = ggml_cuda_op_swiglu_oai_single(gate_value, result); break;
    case GGML_GLU_OP_SWIGLU_CLAMP: result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit); break;
    default: result = result * gate_value; break;
}
```
and the kernel template signature at ~line 616-618 confirming ncols_dst is already a template int, not a runtime branch requiring a rewrite. Patch 1241's actual forced-dispatch restriction to ncols_dst==1 was NOT directly read this batch (only patch.toml/SUMMARY.md) -- implementer must read patch.py's Edit() anchors before widening, per the brief's anchor-verification requirement.

## Files

patches/1241_rd33_mmvq_q8_0_f32_decode/patch.py (widen dispatch); ggml/src/ggml-cuda/mmvq.cu (reference only, no edit expected -- kernel already generic); new test-backend-ops ncols 1..8 tolerance + GLU cases; interleaved production A/B harness/campaign config for Qwen3.8-27B MTP.

## Validation

1. patch-lint + rebase-check on the widened package. 2. test-backend-ops tolerance correctness for ncols 1..8, forced-candidate, ncols>8 fallback, fused-gate/GLU, non-target (MoE/non-Q8_0/non-gfx1100) controls. 3. rocprof/resource checks. 4. Interleaved paired A/B on production Qwen3.8-27B MTP (Brutus, not run here) where ncols=5/6 dominates -- explicitly NOT the noisy first sequential round. 5. Quality/non-inferiority guard alongside any throughput claim.

## Effort & Risk

S-M: dispatch-widening only, kernel body already generic; main risk is in correctly interpreting a marginal/noisy A/B result, which is why the interleaved-trial design matters more here than new code risk.

## Standards

Shape-aware eligibility; tolerance not bit identity; interleaved evidence; no noisy sequential conclusion.

## Acceptance Criteria

Correctness holds for widened gate and unsupported paths remain native; only a real interleaved production gain promotes. If effect remains null, close as validated null with dominant-shape coverage documented; do not claim the earlier n=1 result covered MTP.

## Notes

Supersedes: RD33
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd33

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH (1241, untested), needs ncols 5/6 widening. Verified kernel body is already ncols_dst-generic (fused-gate/GLU switch has no ncols_dst dependence); the restriction to ncols_dst==1 lives at the dispatch layer in patch.py, which was not directly read this batch -- implementer must verify that specific anchor before editing. GPT design request for this item hit a queue-saturated gateway and was not obtained in-session; plan authored directly from verified mmvq.cu source.

2026-09-24 GPT review req_2b717df095b44703 applied: verified via direct read of patch.py that the launcher hardcodes c_ncols_dst=1 (patch.py:288) and the gate is ne1==1 && !forced.requested() (patch.py:322/422) -- merely widening the ne1 gate would not select the correct template instance. Replaced with a required template<int N> launcher + runtime switch for N=1..8, corrected gate replacement text, added requires=[0600_mmvq_geometry] (missing), and corrected the inverted 'forced-only' description.

2026-09-25 (975375b4): 1241 widened to ncols_dst 1..8 -- launcher templated on ncols with a runtime switch; the f32 activation read now honours stride_col_y (it assumed ncols 1); requires 0600_mmvq_geometry; per-ncols WARN marker. New contract RD33-MMVQ-Q8_0-F32-DECODE (backend_reference, positive = real MTP decode mtp_wall_tps on tierL-qwen27b-q8 dual gfx1100, control = dense Q6_K tg128; improvement_no_regression_v1, 4 sessions). Producer: test-backend-ops Q8_0 MUL_MAT (n=1..9) on both arms + markers for every ncols 1..8, shared producer_support.mtp_server_lane (RD73's lane generalised), dense control lane. Hardware: needs a dual-gfx1100 slot (GPU0+1), not yet queued.

## Change Log

- 2026-09-09T10:55:14.091235+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:27.752011+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.244503+00:00 (updated-by): Updated: section:ledger-events
- chg_20260909_143622_cleaned-up-the-tooling-registr_2452
- 2026-09-09T14:36:22.965188+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.975621+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:55:15.763944+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025542_rdna-successors-prbe2628-now_5552
- 2026-09-10T02:55:42.877445+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:32:47.756138+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:45:20.608430+00:00 (updated-by): Updated: section:steps, section:notes
- 2026-09-24T15:40:42.696090+00:00 (state-transition): State: pending → in_progress
- 2026-09-24T15:40:45.627605+00:00 (updated-by): Updated: section:notes
