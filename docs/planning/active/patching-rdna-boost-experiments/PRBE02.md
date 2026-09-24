---
id: PRBE02
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:35.078709+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Establish flash-attention WMMA correctness barriers

## Description

TODO, rescoped off dead identity. PRBE02's target (RD05 WMMA flash-attn barrier/race repair) originally lived inside patch 1203, which is now `rejected` as a bundle (PA39: RD06 slice measured no gain, CI95 low -0.0745% vs required >=0.5%). PA39's own decision (GPT lifecycle review req_f34f50a25c6240fe) states RD05's passing slice may only return as a NEW, separately-identified `untested` patch with fresh hardware evidence -- 1203's receipt cannot be reused. PRBE110 (a separate plan item, already created) owns the mechanical re-extraction: splitting rd05-kbc-sync (generic WMMA kernel, <=128 heads) from rd05-k00-sync (larger-head configs) and keeping RD06-owned edits (rd0506-config-table, rd0506-softcap-read, rd06-dkq-gate, rd06-wmma-gating) out. PRBE02 does NOT duplicate that extraction; it owns the DOWNSTREAM correctness-only qualification campaign for whatever RD05-only patch PRBE110 delivers.

## Steps

1. PRBE110 must land a new patches/<id>/ package carrying ONLY the RD05 edits from 1203/patch.py: `rd05-k00-sync` and `rd05-kbc-sync` in ggml/src/ggml-cuda/fattn-mma-f16.cuh (both anchors verified present and unique in patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq/patch.py at lines 669 and 687), plus an RD05-only BIGCHERRY_PATCH_HIT marker gated on the `BEST_FATTN_KERNEL_MMA_F16` case in fattn.cu. Exclude all rd06-*/rd0506-config-table/softcap edits from that package. This item is blocked until PRBE110 delivers that exact package.
2. Bind PRBE02 to PRBE110's final RD05-only package id once it lands (record id+SHA in notes).
3. Resolve baseline composition (resolve_source_composition/materialize_composition) with ONLY the new RD05-only patch applied -- no RD06/RD07 co-application.
4. Run require_ppl_equality-based correctness harness (tools/bigcherry/experiment/perplexity.py) on gfx1201 (gfx1100/gfx1030 controls) as a first-pass signal -- not sufficient alone per acceptance criteria.
5. Build the targeted matrix: head sizes 192/256/320/512/576, graph and non-graph, repeated under load, vs reference. Author dedicated test-backend-ops fixtures per head size if none exist.
6. Run backend corpus + architecture gates (gfx1100/gfx1201/gfx1030); any failure blocks closure.
7. Record the new patch identity, environment, repetition counts, fallback evidence -- do not reuse old 1203/PA39 PPL numbers as evidence for the new identity (provenance only).

## Detailed Solution & Technical Design

Correctness-only qualification, no performance claim. Target identity is now the RD05-only package PRBE110 must produce (patches/<id>/ carrying only rd05-k00-sync + rd05-kbc-sync edits to fattn-mma-f16.cuh, RD05-only activation marker at fattn.cu's BEST_FATTN_KERNEL_MMA_F16 case). The old gfx1201 PPL-equality pass (sigma=1.35, PASS) recorded under the dead 1203 identity is provenance only per PA39/req_f34f50a25c6240fe -- it does not close this item. The targeted head-size (192/256/320/512/576) graph/non-graph/loaded matrix has never been run under any identity and remains the real work, to be executed against PRBE110's new RD05-only package once it lands.

## Code Samples & Guidance

No new anchors to propose here -- PRBE110 owns authoring the RD05-only patch.py/Edit() calls (source: mmq-vec-dot.cuh / flash-attn WMMA combine files from the historical 1203/1d525bd45 fork lineage, split per PRBE110's own step 2). PRBE02's own deliverable is qualification campaign code/fixtures (test-backend-ops cases for head sizes 192/256/320/512/576, graph/non-graph variants), not patch.py edits.

## Files

Depends on PRBE110's package name (TBD, e.g. patches/12xx_rd05_wmma_fa_correctness_barriers/); flash-attention WMMA combine kernel sources under ggml/src/ggml-cuda/; targeted graph/non-graph correctness fixtures for head sizes 192/256/320/512/576; campaign evidence directory for the new identity.

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay <new-rd05-id> --source bigcherry-tuning` once PRBE110's package exists. Hardware (Brutus, not run here): gfx1100/gfx1201/gfx1030; head sizes 192/256/320/512/576 graph and non-graph; repeated loaded runs; native/reference numerical comparison within preregistered tolerance; PPL-equality as a supplementary whole-model signal only, not a substitute for the targeted matrix.

## Effort & Risk

L effort -- the targeted head-size matrix has never been built; blocked until PRBE110 delivers the new patch identity. Risk: treating the old 1203-era PPL pass as sufficient evidence would violate PA39's own explicit non-reuse decision.

## Standards

Fail closed on correctness; no performance reliance before prerequisite passes; immutable patch/source identity and reproducible evidence.

## Acceptance Criteria

Barrier repair is independently identified and all targeted shapes pass repeated graph/non-graph and loaded correctness against reference within preregistered tolerances; architecture guards hold; no dependent performance arm is accepted before this gate.

## Notes

Supersedes: RD05
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd05

2026-09-11: started a real gfx1201 build+correctness campaign for patch 1203 (isolated scratch clone, resolve_source_composition/materialize_composition methodology, pin 28ff0958291ce3465fabd7bd679d4b0edd742bd9, HIP_VISIBLE_DEVICES=2 for the real gfx1201/R9700 device). Building llama-perplexity + llama-bench for BC-baseline (1203 excluded) and BC+1203 -- currently in progress, not yet complete. IMPORTANT CAVEAT recorded honestly before results land: this initial pass is a generic PPL-equality correctness check + llama-bench timing, NOT yet the specific targeted-head-size (192/256/320/512/576) graph/non-graph/loaded correctness matrix this item's own acceptance criteria require. Treat this run as a first real signal, not closure of PRBE02 -- the full targeted correctness matrix remains separate, not-yet-done work regardless of this pass's outcome.

REAL RESULT 2026-09-12: gfx1201 PPL-equality correctness check completed. BC-baseline (1203 excluded): PPL=10.4463 +/- 0.02753. BC+1203 (RD05/06/07): PPL=10.3938 +/- 0.02737. Combined-uncertainty sigma = |10.4463-10.3938| / sqrt(0.02753^2+0.02737^2) = 0.0525/0.0388 = 1.35 -- well under this project's established 3-sigma significance threshold (see tools/bigcherry/experiment/perplexity.py's require_ppl_equality). PASS: no statistically significant PPL divergence between baseline and RD05/06/07-patched builds on real gfx1201 hardware, tierA-qwen4b-q6k, real wikitext2 corpus. This is real evidence for PRBE02's own correctness-barrier claim at the whole-model level, but does NOT yet satisfy this item's own stated acceptance criteria (the specific targeted head-size 192/256/320/512/576 graph/non-graph/loaded matrix) -- this PPL check exercises whatever head sizes this one real model's attention layers happen to use, not the full targeted matrix. Real, positive first signal; full closure remains separate work.

2026-09-24 relevance at b11126: TODO, blocked on PRBE110's extraction deliverable; do not duplicate PRBE110's own steps. Prior real PPL-equality evidence (sigma=1.35, PASS) is provenance only, not valid closing evidence for a new identity per PA39/req_f34f50a25c6240fe. GPT design request submitted (req_990c48138f9b408e, batched with PRBE04); gateway was heavily congested at submission time (10 active gpt-auto sessions, recent composer-operation-timeout failures per agent_task_gateway_overview) -- if that request never completes, this plan was authored directly against the real patch.toml/SUMMARY.md evidence on disk instead.

2026-09-24 GPT review req_7f4dea253b7247f0 applied: pinned PRBE02's PRBE110 dependency to the RD05-only edits (rd05-k00-sync, rd05-kbc-sync in fattn-mma-f16.cuh, verified present in patches/1203.../patch.py) plus an RD05-only activation marker at fattn.cu's BEST_FATTN_KERNEL_MMA_F16 case, excluding all rd06-*/rd0506 edits; item remains pending/blocked on PRBE110.

2026-09-24 GPT review req_7f4dea253b7247f0 applied: pinned PRBE02's PRBE110 dependency to the RD05-only edits (rd05-k00-sync, rd05-kbc-sync in fattn-mma-f16.cuh, verified present in patches/1203.../patch.py) plus an RD05-only activation marker at fattn.cu's BEST_FATTN_KERNEL_MMA_F16 case, excluding all rd06-*/rd0506 edits; item remains pending/blocked on PRBE110.

## Change Log

- 2026-09-09T10:53:35.078709+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:10:12.738042+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.134485+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.804684+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:30:13.076053+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023037_three-rdna-boost-successors-no_6965
- 2026-09-10T02:30:37.067104+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T23:50:28.466918+00:00 (updated-by): Updated: section:notes
- chg_20260911_235103_caught-myself-running-a-real-h_5912
- 2026-09-11T23:51:03.367650+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T03:20:07.065538+00:00 (updated-by): Updated: section:notes
- 2026-09-24T02:31:34.390785+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:34:23.654204+00:00 (updated-by): Updated: section:notes
- 2026-09-24T04:34:50.234264+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:notes
