---
id: PRBE40
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:15.028499+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-KVPROJ-001: Fuse attention K and V projection into one MMVQ dispatch

## Description

IMPLEMENTED-AS-PATCH (re-dispositioned 2026-09-24, correcting this plan's own earlier TODO draft above). patches/1205_rd12_paired_mmvq_dual_output/ (state="untested", plan-ids=["RD12"]) already implements exactly what AMD-KVPROJ-001 asks for: its own SUMMARY.md states 'K and V projections in an attention layer run two mmvq matmuls over the identical activation; fusing them into one launch avoids re-reading the activation and issuing a second kernel... bit-identical output vs the unfused path plus small positive tg64 gains on gfx1201', validation-architectures = ["gfx1100", "gfx1201", "gfx1030"] (covers this item's gfx1151/gfx1201 target, plus more). This plan's disposition changes from TODO to: validate/qualify patch 1205, do not author a new package. Note 1205 is broader than PRBE40's original AMD-KVPROJ-001 scope (any two adjacent shared-activation MUL_MAT nodes, not model-size-gated to 0.5B-4B) and uses a different mechanism than either this plan's original sketch or GPT's load-time-concatenation design (a fusion.gate + dst_gate epilogue on the FIRST matmul computing the SECOND weight's result too, no load-time weight concatenation) -- 1205's mechanism is lower-risk (no new load-time tensor, no VRAM duplication) and should be preferred over authoring a new package.

## Steps

1. Read patches/1205_rd12_paired_mmvq_dual_output/patch.py in full to confirm its matcher covers the K-proj/V-proj adjacency case specifically (its SUMMARY.md names K/V as the example but the matcher may be more generic -- confirm which MUL_MAT-adjacency shapes it actually accepts). 2. Confirm 1205's conflict with patches/1207_rd17_moe_topk_down_fold/ does not affect target models for this item (0.5B-4B Qwen attention path vs MoE topk/down fold -- likely disjoint tensors, but verify). 3. Run patch-lint and patch-rebase-check for 1205 standalone and against the current production patchset. 4. Design the hardware qualification campaign this item actually needs: K and V outputs individually bit-identical to the two-matmul baseline (1205 already claims this in its SUMMARY -- needs fresh confirmation run, not reuse of any prior receipt per this project's evidence-freshness norms), MMVQ occupancy/launch-count/TG timing on 0.5B-4B Qwen at gfx1151/gfx1201 (this item's specific hardware target -- 1205's own validation-architectures list gfx1100/gfx1201/gfx1030, gfx1151 not yet covered, note as a gap), and a 27B+ control showing no regression / no accidental activation on large FFN-dominated models. 5. If 1205 is promoted/qualified on this evidence, close this plan item as satisfied by 1205's promotion rather than tracking separate promotion criteria.

## Detailed Solution & Technical Design

This item's own IMPLEMENTED-AS-PATCH disposition (per the brief's own decision rubric) supersedes both this plan's self-authored design (kept below, in code_samples, purely for architectural reference/comparison -- NOT to be implemented) and GPT's load-time-concatenation design (see notes) -- an existing, less invasive, already-package mechanism achieves the same functional goal and should be qualified/promoted rather than duplicated by a new patch.

## Code Samples & Guidance

[SUPERSEDED reference material, not to be implemented -- see description] This plan's original self-authored sketch (model-load weight concatenation + graph-level view slicing) and GPT req_3710dbae73fb49be's sketch (llama_model::impl-owned packed tensor, env-gated) both remain below in the notes for architectural comparison only. The ACTUAL next action is qualifying patches/1205_rd12_paired_mmvq_dual_output/, not writing new code. Read patches/1205_rd12_paired_mmvq_dual_output/patch.py directly for its real anchors/mechanism before any further design work.

## Files

patches/1205_rd12_paired_mmvq_dual_output/{patch.toml,patch.py,README.md,SUMMARY.md,validation/,validation.toml} (existing, to read/qualify, not create). patches/1207_rd17_moe_topk_down_fold/ (existing conflict, to re-check applicability).

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint` and `patch-rebase-check --focal-overlay 1205_rd12_paired_mmvq_dual_output --source bigcherry-tuning` (re-run, don't assume clean from patch.toml state alone). Hardware (Brutus, not run here): K/V-specific correctness+perf campaign as in steps step 4 above, at THIS item's specific hardware target (gfx1151/gfx1201, 0.5B-4B), which is narrower than 1205's own already-declared architecture list -- fresh evidence at gfx1151 specifically is the gap to close.

## Effort & Risk

S (downgraded from M) -- this is now a qualification/validation task against an existing untested patch, not new design+implementation.

## Standards

Prefer qualifying an existing conservative mechanism over introducing a new, more invasive one (1205's epilogue-gate approach vs load-time weight duplication) when both achieve the same functional goal -- avoids VRAM duplication and a second code path for the same optimization.

## Acceptance Criteria

Acceptance requires separately identical K/V outputs, safe two-dispatch fallback for incompatible metadata/layout/hardware, and a repeatable benefit with low load-memory complexity on 0.5B-4B while showing no regression on 27B+ controls.

## Notes

Supersedes: RD48
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd48

Supersedes RD48. Source is AMD PR Set 4/#59 (discussion #26378), with source status recheck required before implementation.

2026-09-24 relevance at b11126: no existing patch (grep for RD48 = no hits); not upstream-absorbed (mmvq.cu's ggml_cuda_op_mul_mat_vec_q dispatches per-src0, confirmed by direct read at b11126 mmvq.cu:1538-1565, no K/V-paired dispatch exists). GPT design requests req_6a45d0917b114112 (failed) and req_3710dbae73fb49be (abandoned unresolved, single-in-flight cap) both unusable; this plan was authored directly from verified source by the planning agent. Source is AMD PR Set 4/#59 (discussion #26378) per the item's original notes -- re-verify that PR's current merge/content status before implementation, as the item itself flags.

2026-09-24 GPT design retry req_3710dbae73fb49be COMPLETED (succeeded, session ses_f9fb6da0779240e3). Its design is a stronger, more concrete alternative/supplement to this plan's self-authored design: load-time packing lives in llama-model.h/.cpp (bigcherry_wkv_packed vector, ggml_new_tensor_2d packed [wk_rows+wv_rows] with ggml_view_2d slices for pk/pv, ggml_backend_tensor_copy to materialize), gated by an opt-in env var GGML_CUDA_KVPROJ_FUSION (avoids unconditional duplicate VRAM for models that don't benefit), explicitly excludes LoRA-active K/V and scaled/NVFP4 weights, and wires into graph build via a use_wkv branch in the K/V-building code (Kcur/Vcur as ggml_view_2d into one ggml_mul_mat(wkv, cur) result) -- this is materially more concrete than this plan's conceptual sketch (identifies real ownership location: llama_model::impl, and a real accessor pattern: bigcherry_get_wkv_packed(il)). CRITICAL FINDING from GPT, independently plausible and NOT yet verified by this session: 1205_rd12_paired_mmvq_dual_output (confirmed to exist: `ls patches/` shows this directory) may ALREADY achieve a similar one-MMVQ-launch-for-two-weights result via a different mechanism (fusion.gate + dst_gate, no load-time concatenation) -- if true, PRBE40 may be partially IMPLEMENTED-AS-PATCH rather than pure TODO, and GPT's patch.toml sketch already declares `conflicts = ["1205_rd12_paired_mmvq_dual_output"]` on this basis. MANDATORY next step before implementation: read patches/1205_rd12_paired_mmvq_dual_output/patch.toml and patch.py in full to determine whether it already covers K/V projection specifically (vs a different shared-activation pair like gate/up) and whether PRBE40 should be re-dispositioned to 'extend 1205' rather than a new standalone package -- this planning pass did not have budget to read 1205 in full. GPT's anchors are tentative and unverified against live source; re-grep before implementation.

2026-09-24 relevance CORRECTED: initial grep for the literal string "RD48" (this item's legacy id) in patches/*/patch.toml found nothing, which led this plan to initially (incorrectly) draft a TODO/new-design plan. A later architectural cross-check (prompted by GPT req_3710dbae73fb49be independently flagging the same overlap) surfaced patches/1205_rd12_paired_mmvq_dual_output/ (plan-ids=["RD12"], not RD48/PRBE40 -- hence missed by the initial ID-only grep) as already implementing this item's functional goal (K/V shared-activation MMVQ fusion), confirmed by direct read of its SUMMARY.md. LESSON for future batches: relevance checks must also grep by FUNCTIONAL description/keywords (e.g. 'mmvq', 'kv', 'shared activation'), not only by the item's own RD/legacy id, since patch plan-ids and plan-item legacy ids do not always cross-reference each other. This plan's earlier TODO draft (self-authored design + GPT req_3710dbae73fb49be's load-time-concatenation design) is preserved in code_samples/this notes history for architectural reference only and must NOT be implemented -- qualify 1205 instead.

## Change Log

- 2026-09-09T10:56:15.028499+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:29.738621+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.310431+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.070903+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:03:05.042679+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- chg_20260910_030318_repaired-three-patching-succes_9681
- 2026-09-10T03:03:18.998203+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:05:43.401135+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030619_removed-migration-placeholder_7703
- 2026-09-10T03:06:19.314200+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:44:35.025453+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-24T04:49:13.714366+00:00 (updated-by): Updated: section:notes
- 2026-09-24T04:50:18.038797+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
