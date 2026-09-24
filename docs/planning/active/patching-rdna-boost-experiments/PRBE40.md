---
id: PRBE40
order: 0
plan: patching-rdna-boost-experiments
state: in_progress
created-at: '2026-09-09T10:56:15.028499+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-KVPROJ-001: Fuse attention K and V projection into one MMVQ dispatch

## Description

IMPLEMENTED-AS-PATCH, qualify 1205_rd12_paired_mmvq_dual_output -- but per GPT review (req_e17e0bf5a68c48d5), the 'already implements' framing overstates correctness as-is. 1205's matcher compares only output ne[0..2]; the dual-output kernel writes dst_gate using the FIRST output's strides, so differing ne[3]/nb[] between the two outputs is unsafe and currently unchecked. No fusion-memory-range/output-overlap check exists. Target K/V identity is assumed, not proven -- matcher can fuse ANY first compatible projection pair sharing src1, not specifically K/V. Patch metadata omits gfx1151 from validation-architectures (this item's specific hardware target).

## Steps

1. Read patches/1205_rd12_paired_mmvq_dual_output/patch.py in full to confirm matcher scope. 2. BEFORE qualification, add a full shape/layout equality check: require ne[0..3] AND nb[0..3] equality for BOTH outputs (not just ne[0..2]), rejecting fusion when strides or ne[3] differ. 3. Add an explicit ggml_cuda_check_fusion_memory_ranges(cgraph,node_idx,j-node_idx+1,{node_idx,j},2) call (or equivalent) proving the two outputs are disjoint, since dst_gate currently borrows the first output's strides. 4. Add overlap/layout negative tests (differing ne[3], differing nb[], overlapping outputs) that must show NO fusion. 5. Add activation evidence identifying WHICH projection pair fused (not just that fusion occurred), to confirm it targets genuine K/V pairs and not an incidental unrelated pair. 6. Add gfx1151 to patches/1205's validation-architectures list (currently gfx1100/gfx1201/gfx1030 only) since this item's hardware target requires it. 7. Confirm 1205's conflict with patches/1207_rd17_moe_topk_down_fold/ does not affect target models. 8. Run patch-lint and patch-rebase-check.

## Detailed Solution & Technical Design

This item's own IMPLEMENTED-AS-PATCH disposition (per the brief's own decision rubric) supersedes both this plan's self-authored design (kept below, in code_samples, purely for architectural reference/comparison -- NOT to be implemented) and GPT's load-time-concatenation design (see notes) -- an existing, less invasive, already-package mechanism achieves the same functional goal and should be qualified/promoted rather than duplicated by a new patch.

## Code Samples & Guidance

[SUPERSEDED reference material, not to be implemented -- see description] This plan's original self-authored sketch (model-load weight concatenation + graph-level view slicing) and GPT req_3710dbae73fb49be's sketch (llama_model::impl-owned packed tensor, env-gated) both remain below in the notes for architectural comparison only. The ACTUAL next action is qualifying patches/1205_rd12_paired_mmvq_dual_output/, not writing new code. Read patches/1205_rd12_paired_mmvq_dual_output/patch.py directly for its real anchors/mechanism before any further design work.

## Files

patches/1205_rd12_paired_mmvq_dual_output/{patch.toml,patch.py,README.md,SUMMARY.md,validation/,validation.toml} (existing, to read/qualify, not create). patches/1207_rd17_moe_topk_down_fold/ (existing conflict, to re-check applicability).

## Validation

Offline: patch-lint, patch-rebase-check for 1205 standalone and against the production patchset. Correctness: the new ne[0..3]/nb[0..3] equality check plus fusion-memory-range disjointness proof must be added and covered by negative tests (differing ne[3]/nb[], overlapping outputs) before qualification proceeds -- these are currently MISSING gaps, not just untested claims. Hardware (Brutus, not run here): K/V-specific correctness+perf campaign at gfx1151/gfx1201, 0.5B-4B, with activation evidence identifying the specific fused pair.

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

2026-09-24 GPT review req_e17e0bf5a68c48d5 applied: added missing ne[0..3]/nb[0..3] equality check, explicit fusion-memory-range disjointness proof, overlap/layout negative tests, fused-pair activation evidence, and gfx1151 to 1205's validation-architectures as prerequisites before qualifying 1205 for PRBE40.

2026-09-25: steps 2,3,5 implemented in 1205 (cd35b01f): ggml_are_same_shape+ggml_are_same_stride+F32 on both outputs, ggml_cuda_check_fusion_memory_ranges over {i,j}, marker names the fused weights. b11126 gfx1100 run r1: apply/build/activation/bit_identical PASS (6/6 rows, marker a=rd12_k_weight b=rd12_v_weight) but no promotion lanes existed; producer extended with tg128 positive / pp512 control, 10 rounds, ci95 policy (bd6272be). r2 running. gfx1201 run r1 died on a clang bus error from host disk pressure (root 98%), not code. gfx1151 BLOCKED (no card). Step 4 negative test-backend-ops fixtures still to author.

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
- 2026-09-24T05:07:35.021954+00:00 (updated-by): Updated: section:description, section:steps, section:validation
- 2026-09-24T05:07:37.936008+00:00 (updated-by): Updated: section:notes
- 2026-09-24T14:09:49.158527+00:00 (state-transition): State: pending → in_progress
- 2026-09-24T14:09:52.081799+00:00 (updated-by): Updated: section:notes
- chg_20260924_141016_five-experimental-rdna-patches_5706
- 2026-09-24T14:10:24.783529+00:00 (updated-by): Updated: section:ledger-events
