---
id: PRBE12
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:19.359011+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Evaluate MUL_MAT plus RESHAPE plus ADD fusion

## Description

Qualify patch 1206 MUL_MAT+RESHAPE+ADD view fusion after offline safety repair, including CUDA equivalence, capture and causal performance.

## Steps

- Use only the exact RESHAPE-mediated view/add pattern and ggml_can_fuse_subgraph safety checks.
- Require view specifically at ADD.src[0], reject null addends, wrong wiring, extra consumers, non-VIEW nodes and direct-ADD near misses.
- Compare fused/unfused outputs and graph capture/replay on CUDA/HIP where supported.
- Keep PRBE05/Q8 cache and other enhancements out of the standalone arm unless explicitly declared in identity.
- Run balanced timing only after correctness and capture gates pass.

## Detailed Solution & Technical Design

The new view-mediated matcher is non-commutative by safety contract; preserve legacy direct-ADD matcher behavior unchanged. No false-positive graph rewrite is acceptable.

## Code Samples & Guidance



## Files

patches/1206_rd13_mul_mat_add_view_fusion; graph matcher/fusion source; targeted safety tests; fused/unfused output fixtures; graph capture and timing artifacts.

## Validation

Direct ADD exclusion; view at src[0]; reversed wiring; null/extra-consumer/non-VIEW rejection; output parity; graph capture/replay; balanced causal timing.

## Effort & Risk



## Standards

Exact pattern; no false positives; causal isolation; preserve legacy direct-ADD semantics.

## Acceptance Criteria

All exact-pattern and negative fixtures pass; fused output matches unfused/reference; graph capture/replay is stable; only a statistically supported benefit without regressions is promotable.

## Notes

Supersedes: RD13
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd13



REAL HARDWARE CORRECTNESS CHECK RUN (2026-09-11/12), first-ever real execution of this patch's correctness proof: authored a real ppl_equality correctness producer (patches/1206_rd13_mul_mat_add_view_fusion/validation/rd13_correctness.py, reusing the shared tools/bigcherry/experiment/perplexity.py primitive extracted from RD17's own producer -- RD13 needs no bespoke control-variant worktree, since its patch.py is one self-contained block replacement with no separate struct/kernel-plumbing edits that would stay compiled-but-inert under a partial revert; the project's normal baseline composition with RD13 excluded IS the correct control). Not yet CLI-wired for the same reason as RD17 (require_execution_package()'s unconditional gate); callable directly via run_rd13_ppl_check().

Ran for real on Brutus: gfx1201, tierM-gptoss20b-q6k (gpt-oss-20B, the contract's own declared model), real wikitext2 corpus. Result: PASS -- subject PPL=561.6933+/-1.67373, control PPL=561.6933+/-1.67373, delta=0.0, sigma=0.0 (exact match, same pattern as RD17's own fix-verification run).

HONEST CAVEAT (same class as RD17's, recorded rather than overclaimed): delta=0.0 exactly proves NO REGRESSION from RD13's patch -- it does NOT by itself prove the RESHAPE-mediated fusion actually activated during this run. RD13's own patch.py docstring says the view pattern needs "an SSM/Mamba-family model" to fire; gpt-oss-20b is MoE (the contract's own model choice) but whether it actually contains a RESHAPE-mediated residual add in its real graph is unverified here -- RD13 has real BIGCHERRY_PATCH_HIT activation trace markers (unlike RD17) that were NOT checked in this run (this producer only proves PPL equality, not activation). A real activation check (trace-marker probe under BIGCHERRY_PATCH_TRACE=1) is separate, not-yet-done follow-up work needed to know whether this PASS reflects "fusion fired and is safe" or "fusion never fired, so trivially safe."

Also worth flagging honestly, not blocking: the absolute PPL magnitude (561.69) is unusually high for a 20B model on wikitext2 (typically low double digits) -- possibly a real quirk of this specific quantization/corpus/context-length combination, possibly an unrelated tokenization/chat-template mismatch in how llama-perplexity was invoked here. Since it's IDENTICAL between subject and control, it does not indicate an RD13-specific defect, but the absolute number itself was not independently sanity-checked against a known-good baseline for this exact model/corpus/args combination -- worth a separate look before treating 561.69 as a meaningful baseline for anything else.

Remaining real work: real activation-trace verification (has real markers already, just not exercised by this specific run), the real performance claim (this contract has none -- CORRECTNESS/DETERMINISM... actually check: RD13's own contract IS performance-bearing, moe_decode workload declared), full validation.toml authoring + contract binding (blocked the same way RD17's is). Patch state remains "untested" -- this is real evidence, not a promotion.

## Change Log

- 2026-09-09T10:54:19.359011+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:26.671901+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events



- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.183883+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.881480+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:47:41.484974+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024759_three-rdna-fusion-successors-n_3469
- 2026-09-10T02:47:59.468336+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T14:55:48.900915+00:00 (updated-by): Updated: section:notes
- chg_20260911_145554_ran-the-first-ever-real-correc_4120
- 2026-09-11T14:55:54.334136+00:00 (updated-by): Updated: section:ledger-events
- chg_20260911_212428_finished-documenting-one-more_3069
- 2026-09-11T21:24:28.100057+00:00 (updated-by): Updated: section:ledger-events
