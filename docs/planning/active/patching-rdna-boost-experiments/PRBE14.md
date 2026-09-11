---
id: PRBE14
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:26.498754+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate MoE top-k weights folded into down projection

## Description

Evaluate MoE top-k weights folded into down projection while preserving the explicit composition boundary between patch 1207 and patch 1205.

## Steps

1. Preserve patch 1207's explicit MoE top-k/down-projection recipe identity and verify destination-channel scale semantics and scale-vector shape.
2. Preserve existing NVFP4 behavior and native fallback.
3. Compare fused and unfused outputs across MUL_MAT_ID expert routing, IDs, and scale cases.
4. Validate graph capture, false-positive fallback, and expert routing correctness.
5. Treat patch 1205 and 1207 as mutually exclusive unless an explicitly declared composed experiment passes PKC02 graph identity/conflict validation.
6. Qualify performance only after correctness and composition gates pass.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patches/1207_rd17_moe_topk_down_projection; MoE MUL_MAT_ID routing/down-projection seam; destination-channel scale fixtures; NVFP4/fallback tests; graph/DAG recipe identity and fused/unfused campaign artifacts.

## Validation

Correctness: destination-channel scale semantics/shape, expert IDs/routing, fused-vs-unfused output equality, NVFP4 preservation, native fallback, graph capture, and false-positive rejection. Performance: balanced fused versus unfused qualification with explicit recipe identity. Composition: 1205/1207 conflict gate through PKC02; no implicit combination.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

All RD17 requirements are carried forward: scale semantics/shape, NVFP4 preservation, fused/unfused comparison, expert routing/fallback, graph capture, false-positive fallback, and explicit 1205/1207 composition disposition. Promotion requires correctness and performance evidence for the declared recipe.

## Notes

Supersedes: RD17
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd17

Supersedes: RD17
Inherited constraint: RV105 — retain the 1205/1207 composition conflict and mutually exclusive recipe identities; link graph/DAG composition decisions to PKC02.
Migration: capability-rebaseline-v3-2026-09

Supersedes: RD17
Inherited semantic scope: RD17 detailed requirements plus RV105 composition restriction; historical evidence remains on completed predecessor.
Migration: capability-rebaseline-v3-2026-09

Supersedes: RD17
Inherited constraint: RV105 — retain 1205/1207 composition conflict and mutually exclusive recipe identities; link graph/DAG composition decisions to PKC02.
semantic-carryforward: concrete files restored 2026-09-10.



REAL HARDWARE FINDING + FIX (2026-09-11), first-ever real execution of this patch: authored a real ppl_equality correctness producer (patches/1207_rd17_moe_topk_down_fold/validation/rd17_correctness.py, orchestrated by run_rd17_ppl_check() in tools/bigcherry/patch/validation_campaign.py -- not yet CLI-wired, require_execution_package()'s unconditional README+bound-contract+validation.toml gate blocks that until full validation.toml authoring lands; callable directly in the meantime). Control reversion: revert only RD17's ggml-cuda.cu detection block (the sole site that ever sets x_scale_channel_dst=true), leaving struct/kernel plumbing compiled but inert -- RD08's apply_vdr1_control discipline.

Ran for real on Brutus: gfx1201, Qwen3.6-35B-A3B-APEX-MTP-I-Compact.gguf (a real MoE model), real wikitext2 corpus. Both fusion-subject and no-fusion-control llama-perplexity binaries built successfully; running the SUBJECT crashed: GGML_ASSERT(ggml_nelements(fusion->x_scale) == dst->ne[1]) failed in mmvq.cu.

Got a GPT root-cause review (req_6cf169798c784380) before deciding disposition. Concrete finding: the original fork commit 5e545b7da restricted this fusion to single-column MUL_MAT_ID (GGML_ASSERT(!ids || dst->ne[2] == 1)), but this project's current upstream base broadened the generic fusion host condition since that fork commit was authored, and RD17 was ported without restoring the lost precondition. The kernel only ever indexes x_scale[channel_dst] (one scale per destination row); for a real multi-token MoE batch (routing weights shaped [1,E,T,S], T>1), ggml_nelements(x_scale) is E*T*S, not just E -- exactly the crash. A genuine semantic port/rebase bug, not a false-positive match or harness artifact -- confirmed by the subject reaching RD17's OWN assertion (real activation proven) while the control (detection reverted, nothing else changed) does not.

FIXED (commit d6e3e46e), same session: restored the fork's single-column constraint in the detection predicate (weights->ne[2]==1, ne[3]==1, mm_node->ne[2]==1, exact nelements match) -- GPT-specified, smallest change faithful to the original fork capability. Decode (single-column) fusion still works; multi-token PPL/prefill now correctly falls through to the unfused MUL path instead of crashing. Also refactored rd17_correctness.py to import _DETECT_BLOCK/_DETECT_ANCHOR directly from the real patch.py instead of hand-duplicating them (closes a real desync-risk bug class the fix itself would otherwise have triggered).

Re-verification of the fix on real hardware is in progress as of this note. Patch state remains "untested" throughout -- this is a real negative-then-fixed result being preserved as evidence, never a silent promotion. PRBE14's own acceptance criteria (destination-channel scale semantics/shape, expert routing, false-positive fallback, fused/unfused correctness) are NOT yet fully satisfied -- this closes one real correctness gap found by the first real execution, not the full PRBE14 scope (which still needs the multi-token/batched extension GPT described as a separate, not-yet-attempted kernel change, plus performance qualification, plus the 1205/1207 PKC02 composition-conflict validation).



FIX RE-VERIFIED ON REAL HARDWARE (2026-09-11/12): re-ran run_rd17_ppl_check() for real on Brutus (same gfx1201/Qwen3.6-35B-A3B/wikitext2 setup) after the single-column-constraint fix. Result: PASS. Both subject (fusion-active) and control (no-fusion) binaries built and ran to completion without crashing -- subject PPL=6.8272+/-0.01595, control PPL=6.8272+/-0.01595, delta=0.0, sigma=0.0. Full artifact: artifacts/rd17-ppl-check.json under the run dir, real build identities recorded for both binaries.

HONEST CAVEAT, recorded rather than overclaimed: this PASS proves the fix eliminates the crash and introduces no PPL regression -- it does NOT prove the (now more narrowly single-column-scoped) fusion still activates during THIS workload. llama-perplexity processes multi-token batches per forward pass; the fix's restored single-column constraint means the fusion path likely never triggers for batched perplexity computation at all, so subject and control plausibly took the identical unfused code path here (consistent with the exact PPL match). This check proves safety (no regression, no crash), not that the real decode-time benefit the fork originally claimed is still reachable/exercised. Proving activation would need a real single-token decode workload with a trace marker (RD17 currently has none, unlike RD12/RD13's BIGCHERRY_PATCH_HIT markers) -- a real, separate gap, not yet closed.

Summary of RD17's real status: first-ever real execution found and fixed a genuine crash bug (restored a lost porting constraint); the fix is now real-hardware-verified safe (no crash, no PPL regression). Still NOT validated: real activation proof for the (now-narrower) single-column fusion path, the real performance claim (kernel-count reduction / decode timing), and PRBE14's full remaining scope (batched-shape kernel extension if ever wanted, PKC02 composition-conflict validation with 1205/RD12). Patch state remains "untested" -- this real progress is preserved as evidence, not a promotion.

## Change Log

- 2026-09-09T10:54:26.498754+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:35.940393+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.192523+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.895469+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:52:25.957169+00:00 (updated-by): Updated: section:description, section:acceptance_criteria, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:48.994822+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:10:12.521404+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_021313_the-semantic-audit-is-now-trac_4827
- 2026-09-10T02:13:13.325520+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:48:59.175349+00:00 (updated-by): Updated: section:files, section:notes
- chg_20260910_024925_rdna-successors-prbe1416-now_9529
- 2026-09-10T02:49:25.497189+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T13:45:40.613857+00:00 (updated-by): Updated: section:notes
- chg_20260911_134546_found-and-fixed-a-real-crash-b_5091
- 2026-09-11T13:45:46.308829+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T14:29:22.214547+00:00 (updated-by): Updated: section:notes
- chg_20260911_142926_confirmed-on-real-hardware-tha_3426
- 2026-09-11T14:29:26.549077+00:00 (updated-by): Updated: section:ledger-events
