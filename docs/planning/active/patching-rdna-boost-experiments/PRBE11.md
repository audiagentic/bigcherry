---
id: PRBE11
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:15.041193+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate paired MMVQ matmuls over shared activation

## Description

TODO. Patch patches/1205_rd12_paired_mmvq_dual_output exists (state=untested, experiment-contract=RD12-PAIRED-MMVQ-DUAL, conflicts=["1207_rd17_moe_topk_down_fold"]) but has never had a real hardware qualification run as a standalone identity. PRBE11 qualifies the paired K/V MMVQ graph rewrite: exact pattern detection, GLU-precedence preservation, unfused fallback, then isolated-first correctness/performance, then (only if explicitly declared) the 1205-vs-1207 composition recipe -- never an implicit combination.

## Steps

1. Read patches/1205_rd12_paired_mmvq_dual_output/patch.py in full to confirm the exact pattern-matching predicate (CORRECTED per source audit: the real `_DETECT_BLOCK` requires exact `mid->src[1] == mm_a->src[1]` (shared-activation identity, not overlap/allocation-sharing) and, as written, contains NO `ggml_cuda_check_fusion_memory_ranges()` call for the two destinations -- add explicit output-range safety before dispatch, e.g. `out_nodes[] = { i, j }` with `ggml_cuda_check_fusion_memory_ranges(cgraph, i, j-i+1, out_nodes, 2)` (or an equivalent exact interval check), in patches/1205_rd12_paired_mmvq_dual_output/patch.py before writing any test against the corrected predicate.
2. Author exact-pattern fixtures matching the real predicate: two MMVQ ops with identical src1 (exact pointer/tensor identity), same output shape, compatible op/type gates, and now the added disjoint-output-range check passing -- and negative fixtures: src1 merely overlapping/allocation-sharing but not identical (must NOT match), overlapping output ranges (must be rejected once the range check is added), mismatched output shape, incompatible types.
3. Verify GLU fusion precedence is untouched: ggml-cuda.cu:1691's `is_mul_mat_id` GLU-fusion check and the mul_mat_id_bias_glu_ops pattern at ggml-cuda.cu:3195 must still fire exactly as before.
4. Run isolated-first: native baseline, then baseline+1205 alone (no 1207) -- correctness (paired vs unfused numerical equality) against the CORRECTED predicate and output-range check, graph capture/replay, then causal performance.
5. Only after isolated 1205 passes, define the declared 1205-vs-1207 composition recipe; 1205 and 1207 remain composition-conflicting (per patch.toml `conflicts`) until this recipe is authored and validated.
6. Record resolved patch identity, shape matrix, repetitions, and promotion/reject decision in this item's notes.

## Detailed Solution & Technical Design

This is a graph-rewrite qualification, not new kernel design -- the kernel-side implementation (shared quantized-X launch, grid adjustment) already exists in patch 1205's patch.py. The qualification's job is proving the pattern-matcher only fires on exact, safe patterns (never a false positive that silently changes output), that it composes safely (or is correctly blocked from composing) with 1207, and that it has a measurable, causal performance benefit in isolation before any composition claim.

## Code Samples & Guidance



## Files

patches/1205_rd12_paired_mmvq_dual_output/{patch.toml,patch.py} (read, not modified by this item); ggml/src/ggml-cuda/ggml-cuda.cu (graph rewrite/fusion pass, GLU precedence cross-check at ~1691/3195); new exact-pattern/negative/GLU-precedence test-backend-ops or graph-construction fixtures; paired reshape/view interval tests; composition-recipe fixture and PKC02-style conflict-validation evidence for the declared 1205-vs-1207 arm.

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay 1205_rd12_paired_mmvq_dual_output --source bigcherry-tuning`. Hardware (Brutus, not run here): gfx1100/gfx1201/gfx1030; K/V-pair correctness vs unfused reference; graph capture/replay stability; GLU-precedence fixture; isolated causal performance (balanced repeats); composition recipe validated separately once declared, never implicitly combined with 1207.

## Effort & Risk

L effort, advanced skill -- graph-pattern-matching correctness is unforgiving (false positives silently corrupt output); no real hardware evidence exists yet for this exact identity.

## Standards

Exact graph pattern; dependency-aware comparison; fallback; isolated-test-first.

## Acceptance Criteria

Exact patterns are correct under the declared PRBE11 composition; unsafe/near-miss graphs fall back; isolated performance and any composition result are separately attributable; 1207 is not silently combined; RD25/PRBE19 is not a prerequisite.

## Notes

Supersedes: RD12
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd12
Supersedes: RD12 (closed historical predecessor). Preserve source identities 44b51c663... / upstream ba9e339ea... as provenance. PRBE19 is explicitly non-applicable; do not apply raw RD25 or describe it as a prerequisite. PRBE11 may consume PRBE05 only after a real source audit establishes that dependency.

2026-09-24 relevance at b11126: TODO, no hardware evidence yet. GPT design request submitted (req_a8361cdd54af4bd5, batched with PRBE12); gateway was congested at submission time -- authored directly against patches/1205.../patch.toml and the project's own composition-conflict convention (declared in patch.toml `conflicts`) as a fallback.

2026-09-24 GPT review req_7f4dea253b7247f0 applied: corrected the documented matcher predicate -- 1205's real `_DETECT_BLOCK` requires exact `mid->src[1] == mm_a->src[1]` identity, not shared-allocation/overlap detection, and has no output-range safety check. Added a required output-range safety check (`ggml_cuda_check_fusion_memory_ranges`) to patch 1205 before dispatch and updated the negative-fixture matrix to match the real predicate.

## Change Log

- 2026-09-09T10:54:15.041193+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:22.576195+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.177979+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.874477+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:47:35.743786+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024759_three-rdna-fusion-successors-n_3469
- 2026-09-10T02:47:59.451164+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T16:41:04.517699+00:00 (updated-by): Updated: section:notes
- chg_20260911_164110_investigated-why-an-experiment_2332
- 2026-09-11T16:41:10.876945+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T09:52:15.950084+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:notes
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
- 2026-09-12T09:55:06.207567+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T10:09:36.168667+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260912_101015_fixed-the-remaining-plan-taxon_2574
- 2026-09-12T10:10:15.542955+00:00 (updated-by): Updated: section:ledger-events
- chg_20260913_064911_rd12s-fork-claim-paired-mmvq_5827
- 2026-09-13T06:49:14.580560+00:00 (updated-by): Updated: section:ledger-events
- chg_20260914_093904_fixed-rd12s-activation-valida_1312
- 2026-09-14T09:39:07.007863+00:00 (updated-by): Updated: section:ledger-events
- chg_20260914_103034_the-adversarial-review-of-the_7845
- 2026-09-14T10:30:37.707782+00:00 (updated-by): Updated: section:ledger-events
- chg_20260914_124427_run-rd12-contract-is-now-a-r_9981
- 2026-09-14T12:44:30.679907+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:34:22.636649+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:38:53.320058+00:00 (updated-by): Updated: section:steps, section:notes
