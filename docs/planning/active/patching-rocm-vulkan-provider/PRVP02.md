---
id: PRVP02
order: 0
plan: patching-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:01:11.838320+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# Qualify native Vulkan CM1 A-prefetch bounds and exact route

## Description

The old "integrate PR #27952 as an available-but-unselected 1246 candidate" premise is superseded: #27952 merged on 2026-09-24 and is an ancestor of the b11474 pin. Native CM1 is already auto-eligible on AMD RDNA3/RDNA4 with cooperative int8 support. Do not backport 1246 or introduce a second selector/registry. The remaining narrow safety question is a source-visible A-operand prefetch past end_k for partial K; upstream issue #29342 remains open. The pinned shader has an end_k guard on B loads but not A loads. An observable GPU fault/corruption is NOT established by this audit.

## Exact implementation / data flow

- Pin: llama.cpp b11474 = b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea. ggml/src/ggml-vulkan/ggml-vulkan.cpp:1887-1949 gates int8 CM1 on coopmat_int_support + AMD_RDNA3/AMD_RDNA4 and ggml_vk_matmul_cm1_int_shmem_support() on physical LDS size. Lines 2400-2568 construct CM1 pipelines with arch/quant negative gates. ggml_vk_get_mul_mat_mat_pipeline_map() (near 5976) resolves the bound pipeline; quantize_y (near 6330) selects Q8_1 activation preparation only when a matching pipeline exists, else falls back to F16. These are native controls, not new BigCherry features.
- vulkan-shaders-gen.cpp near 634 emits mul_mmq_cm1.comp for 13 formats: q4_0, q4_1, q5_0, q5_1, q8_0, iq4_nl, iq4_xs, mxfp4, q3_k, q4_k, q5_k, q6_k, nvfp4. Q2_K is not a CM1 shader. RDNA4 disables q4_1/q5_1/q4_k/q5_k/nvfp4 dense CM1 where native fallback is faster; do not override these without measured evidence.
- mul_mmq_cm1.comp defines BK=32 and BK_STEP=4 for MUL_MAT, BK_STEP=2 for MUL_MAT_ID. In mul_mmq_cm1_funcs.glsl::PREFETCH_BLOCK (pinned line ~557), A uses block_a_load(ib + ks, loadr_a) unconditionally for all ks. The adjacent B path uses ((blk)+ks*BK < end_k) ? (ib+ks) : ib, and B-to-shmem explicitly zeros out-of-range contributions. If the final K tile is partial and the A descriptor is not padded, the A load may address beyond the logical row/tensor. Driver-specific zero returns are not a correctness contract.
- Minimal proposed safety fix (upstream-first, only if still needed): use the SAME end_k predicate for the A block index, choosing ib for invalid ks, leaving B's zeroing intact. Keep descriptor, scale, quant format, tile shape, and pipeline selector unchanged. This is a correctness guard, not an optimisation claim. Re-evaluate against an upstream fix before any local patch.

## Cheapest discriminator / tests

1. Host boundary fixture (run in this audit): nine BK=32 index cases with step 4/2; dense K=160/960/5184/2880 show 3/2/2/2 unsafe final-tile prefetches, dense K=5120/6144 show zero, ID K=96 shows one and K=128/5184 zero. Guarded indices remain in the final valid tile. Note: upstream issue #29342's claim "K=5184 multiple of 128" is arithmetically false; it is a multiple of 64, so record the actual pipeline and BK_STEP before interpreting a run. This host model proves address arithmetic only, not actual GPU OOB behavior.
2. When Vulkan hardware work resumes (RRVP02 pause remains authoritative), run a bounded pinned-source shader-instrumentation/descriptor-range probe on gfx1100 RADV first, then gfx1201 separately. Force actual CM1/Q8_1 binding with route proof; include dense K=160,960,5184,2880 and aligned K=5120/6144; MUL_MAT_ID K=96/128/5184; split-K final slice; contiguous and view-backed A; guard-page/poison controls. Capture end_k, blk, BK_STEP, descriptor range, source byte bounds, shader route, work counts, and GPU validation diagnostics. No second run/queue if an existing lane is active.
3. Compare stock pin, one minimal A-guard prototype, and reference CPU/known-good Vulkan outputs in the same process. Require backend-ops, deterministic greedy/logits or KLD, NaN/Inf, multiple requests, ubatches, long context, and MTP draft acceptance. If acceptance shifts (issue #29342 reports 88% to 100% externally), treat it as a correctness signal until logits/verification prove otherwise. Preserve graph/descriptor lifetimes.
4. If upstream merges an equivalent guard, update the pin and close local action. Otherwise accept only the smallest guarded-source correction with complete correctness and no >1% E2E regression; safety fixes do not need a speculative +3% speedup. Reject an independent kernel clone.
5. For performance claims, TRVP14/15 must identify the *actual bound shader* and all Q8_1 prep/split/reduce costs. GGML_VK_DISABLE_COOPMAT is NOT an isolated CM1-off control (it disables the extension for other pipelines); a valid A/B requires a narrowly isolated route or matched native baseline. Only per-device/driver/signature CI95-low >=3% E2E gains justify a performance promotion.

## Ownership / dependencies / evidence

PRVP01: retired source transplant. PRVP02: native CM1 safety and capability reconciliation only. TRVP14: optional strict route/telemetry; TRVP15: performance qualification. RRVP02: Vulkan implementation pause. PRVP03: separate Vulkan AllReduce SPI; PKC04: lifecycle ownership decision only. Reuse native Vulkan pipeline/selector and shared runtime identity, not a parallel scheduler, cache, allocator or dispatch registry.

Upstream #27952 merged 2026-09-24; #28440 already supplied IQ4_XS in pinned code. #29342 documents a gfx1100/RADV OOB-trigger experiment but no reproduced wrong result under that driver. Its external 27B A/B reports +6.2% long-context prompt and +1.7% decode; upstream #27952 reports ~1.29x q4_0 operator timing on Strix Halo. Neither is BigCherry evidence. vLLM/AITER and MLC LLM use different quant/kernel/driver paths: capability-gating and correctness methodology transfer, not speed ratios or shader code. BigCherry has no measured CM1-vs-isolated-control result in this audit.

## Files

Pinned ggml/src/ggml-vulkan/ggml-vulkan.cpp; ggml/src/ggml-vulkan/vulkan-shaders/{mul_mmq_cm1.comp,mul_mmq_cm1_funcs.glsl,vulkan-shaders-gen.cpp}; existing Vulkan backend-ops/route tests and RRVP/TRVP evidence. No 1246 or 1247 package was created.

## Validation

Completed in this audit: seven source assertions, 13-format generator check, nine deterministic host boundary fixtures, pinned/master shader equality check, merged-commit ancestry and branch eligibility/novelty inspection. Not run: BigCherry repository pytest, Vulkan compilation, SPIR-V validation, GPU execution or hardware benchmarking.

## Acceptance Criteria

Native source provenance and route are truthful; a real OOB risk is either guarded with source+GPU correctness evidence or explicitly closed by a newer upstream fix; no false CM1-only control, duplicate patch family or fabricated speedup. Keep all implementation/hardware activity deferred while the existing Vulkan pause remains in force.

## Notes

Original PRVP02 2026-09-10 source-capture design is superseded only as to integration; its capability, fallback, and negative-control intent is retained under native Vulkan and TRVP14/15. BCOP77 records the thin disposition.

## Change Log

- 2026-09-09T11:01:11.838320+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:18:09.468597+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.596999+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.462256+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:23:25.498614+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032340_repaired-the-cm1-source-and-in_5642
- 2026-09-10T03:23:40.374809+00:00 (updated-by): Updated: section:ledger-events
