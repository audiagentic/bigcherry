---
id: PRBE53
order: 0
plan: patching-rdna-boost-experiments
state: superseded
created-at: '2026-09-09T10:57:10.449932+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-FA-001: Mixed K/V FlashAttention dispatch for asymmetric KV types

## Description

UPSTREAM-ABSORBED. Mixed K/V FlashAttention dispatch for asymmetric KV types (f16/q8_0, q8_0/q4_0, q4_0/q8_0, etc.) already exists in the CUDA/HIP FA vector-kernel eligibility path at b11126. The item's premised 'early type-equality rejection' does not exist in ggml-cuda; eligibility is per-type-independent, not a K==V gate.

## Steps

1. (Verification only, no implementation) Re-run the grep evidence above against the pin in force at qualification time, since this is pin-relative: `git -C work/upstream/llama.cpp.git grep -n "ggml_cuda_fattn_kv_type_supported\|FATTN_VEC_CASES_ALL_D" <pin> -- ggml/src/ggml-cuda/fattn.cu`.
2. If a future pin bump removes or narrows this table, reopen this item as TODO with the new pin's evidence.
3. Optional low-cost validation (not required for disposition): a test-backend-ops FLASH_ATTN_EXT case with type_K=F16/type_V=Q8_0 (and the reverse) at head_size 128, confirming it selects BEST_FATTN_KERNEL_VEC and produces reference-parity output on XTX/R9700 -- purely confirmatory, not gating.

## Detailed Solution & Technical Design

Evidence (ggml/src/ggml-cuda/fattn.cu, b11126):
- `ggml_cuda_fattn_kv_type_supported(ggml_type)` (fattn.cu:513-527) is called independently on K->type and V->type at fattn.cu:613 (`if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type))`) -- no K==V equality check anywhere in this file or fattn-common.cuh (grepped `K->type != V->type` / `type_k == type_v`: zero hits in ggml-cuda; the only such equality check found repo-wide is in a DIFFERENT backend, ggml/src/ggml-opencl/ggml-opencl.cpp:9157, which does not apply to HIP).
- `ggml_cuda_get_fattn_vec_case(head_size, type_K, type_V)` (fattn.cu:423-489) is a literal cross-product table via the `FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case)` macro, instantiated for all combinations of {F16,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,BF16} x {F16,Q4_0,Q4_1,...} (fattn.cu:424-450+, table continues symmetrically) -- this explicitly includes F16/Q8_0, Q8_0/Q4_0, and Q4_0/Q8_0, the exact combinations PRBE53 asks for. Each compiled instantiation is a real MMA/tile-eligible kernel template (`ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>`), not a conversion shim.
- Fallback: `ggml_cuda_flash_attn_ext_get_alloc_size` (fattn.cu:706-733) only forces an F16 staging conversion (`need_f16_K`/`need_f16_V`) when `ggml_cuda_get_fattn_vec_case(...) == nullptr` (fattn.cu:729, the 'f16_fallback' case) -- i.e. real native mixed-type dispatch is preferred, and conversion is only the fallback for UNSUPPORTED combos, not a blanket policy.
Conclusion: PR #27150's substance is already present at this pin. No further code change proposed.

## Code Samples & Guidance

No BigCherry patch proposed -- nothing to change. Reference anchors only (see detailed_solution).

## Files

No files touched. Reference: ggml/src/ggml-cuda/fattn.cu (read-only verification).

## Validation

N/A for promotion (nothing to promote). If the optional confirmatory test-backend-ops case above is added later for regression-guard purposes, it belongs in tests/test-backend-ops.cpp as a new FLASH_ATTN_EXT case, not a patch package.

## Effort & Risk

None -- disposition only.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Require reference-quality mixed-K/V outputs, verified non-fallback MMA/tile execution, and no material PP/TG/VRAM regression across specified long-context controls.

## Notes

2026-09-24 relevance at b11126: UPSTREAM-ABSORBED, confirmed via ggml/src/ggml-cuda/fattn.cu:513-527 (ggml_cuda_fattn_kv_type_supported checked independently per K/V) and fattn.cu:423-489 (ggml_cuda_get_fattn_vec_case full cross-product table incl. F16/Q8_0, Q8_0/Q4_0, Q4_0/Q8_0). No K==V equality gate exists in ggml-cuda (only an unrelated one in ggml-opencl.cpp:9157). GPT design request: none needed, disposition made directly from source evidence before the batched GPT request (req_bf8959fb36d248e4, submitted covering PRBE53+PRBE54) returned -- see PRBE54 for that request's use.

2026-09-24 GPT review req_2b65d50ebe9547fd: READY

## Change Log

- 2026-09-09T10:57:10.449932+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:23.119898+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.367626+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.160338+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:11:52.617299+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031217_repaired-four-more-active-succ_7909
- 2026-09-10T03:12:17.949987+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:28:53.607363+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T02:29:15.651094+00:00 (state-transition): State: pending → superseded
- 2026-09-24T04:34:06.656744+00:00 (updated-by): Updated: section:notes
