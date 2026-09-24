---
id: PRBE12
order: 0
plan: patching-rdna-boost-experiments
state: in_progress
created-at: '2026-09-09T10:54:19.359011+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Evaluate MUL_MAT plus RESHAPE plus ADD fusion

## Description

TODO, narrowed to remaining scope -- real hardware correctness evidence already exists. Patch patches/1206_rd13_mul_mat_add_view_fusion (state=untested, experiment-contract=RD13-MUL-MAT-ADD-VIEW-FUSION, conflicts=[]) already has a real gfx1201 PPL-equality run (tierM-gptoss20b-q6k, wikitext2): subject PPL=561.6933+/-1.67373, control identical, delta=0.0, sigma=0.0 -- PASS, proving no regression, but NOT proving the RESHAPE-mediated fusion actually activated (real BIGCHERRY_PATCH_HIT trace markers exist in patch.py per this item's own notes but were not exercised in that run). Remaining work: activation-trace verification, the real performance claim (this contract IS performance-bearing per this item's own notes), the negative-fixture matrix (null addend, wrong wiring, extra consumers, non-VIEW, direct-ADD near-miss), and validation.toml/contract binding (a known, pre-existing blocker: require_execution_package()'s unconditional README+bound-contract+validation.toml gate, same blocker class already hit by RD13/RD17's own producers).

## Steps

1. Re-read patches/1206_rd13_mul_mat_add_view_fusion/patch.py's existing activation markers -- CORRECTED per source audit: both markers (patch.py lines ~222 and ~235) currently use `GGML_LOG_INFO`, which is not reliably visible under normal unattended llama-server/bench verbosity (per this project's own HI90/1231 finding, INFO is filtered below llama-server's default level); change both to `GGML_LOG_WARN`, matching the convention already corrected for sibling patches (e.g. RD12/1205).
2. Run a real activation-trace probe with BIGCHERRY_PATCH_TRACE=1 (now WARN-level) against an SSM/Mamba-family model and confirm the marker fires during a real request.
3. Author the negative-fixture matrix: null addend, ADD wired to the wrong operand, extra consumers on the VIEW node, and a non-VIEW node in the mediating position -- CORRECTED: the plan's prior wording said "VIEW" generically but the patch matches exactly `GGML_OP_RESHAPE` in the mediating position, not any VIEW-family op; fixtures must target RESHAPE specifically. A direct ADD (MUL_MAT->ADD with no RESHAPE in between) is NOT a negative-fusion case for this patch -- it must retain the pre-existing legacy (unrelated) direct fusion behavior unchanged and simply must not emit the 1206_rd13 marker; construct that fixture and assert both facts (legacy fusion still fires; RD13 marker does not).
4. Run graph capture/replay with the fusion active on a real activating model; confirm stability across repeated capture/replay cycles.
5. Only after 2-4 pass, run the real performance claim comparing baseline vs baseline+1206 on an activating model, with enough repeats for a defensible CI.
6. Author validation.toml and bind the contract once require_execution_package()'s gate requirements are otherwise satisfiable.

## Detailed Solution & Technical Design

The correctness foundation (no-regression) is real and already proven; what remains is proving causation (does it actually fire, and does firing help), plus the negative-fixture safety net around the pattern matcher (view specifically at ADD.src[0], reject null/wrong-wiring/extra-consumer/non-VIEW/direct-ADD near-misses per this item's own acceptance criteria) which has never been run for real.

## Code Samples & Guidance



## Files

patches/1206_rd13_mul_mat_add_view_fusion/{patch.toml,patch.py,validation/rd13_correctness.py}; new negative-fixture test cases; activation-trace probe script (BIGCHERRY_PATCH_TRACE=1); validation.toml (to author); campaign artifacts for the activation/performance run.

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay 1206_rd13_mul_mat_add_view_fusion --source bigcherry-tuning`. Hardware (Brutus, not run here): BIGCHERRY_PATCH_TRACE=1 activation probe on a real SSM/Mamba-family model; negative-fixture matrix (marker must NOT fire); graph capture/replay stability; balanced moe_decode-style timing once activation is confirmed real.

## Effort & Risk

M effort -- correctness/no-regression already proven; remaining work is activation proof, negative fixtures, and the performance claim, plus the pre-existing validation.toml blocker shared with sibling items.

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

2026-09-24 relevance at b11126: TODO, narrowed -- do not repeat the already-real PPL-equality/no-regression evidence; focus on activation proof, negative fixtures, performance, and contract binding. GPT design request submitted (req_a8361cdd54af4bd5, batched with PRBE11); gateway congested at submission -- authored directly against this item's own existing real-hardware notes and patches/1206.../patch.py as a fallback.

2026-09-24 GPT review req_7f4dea253b7247f0 applied: verified via grep that patch 1206's two activation markers use GGML_LOG_INFO (patch.py lines ~222, ~235) -- changed step 1 to require both be changed to GGML_LOG_WARN. Corrected the mediating-node terminology from generic "VIEW" to the patch's actual match target GGML_OP_RESHAPE, and clarified that direct MUL_MAT->ADD (no RESHAPE) should retain legacy fusion behavior without emitting the 1206_rd13 marker rather than being treated as a rejected pattern.

2026-09-25 implementation: 1206 markers moved to GGML_LOG_WARN (commit after cd35b01f); PRBE39 extension (VIEW + memory-range check) landed in the same package. b11126 gfx1100 campaign (work/runs/prbe12-1206-gfx1100 on Brutus): ELIGIBLE, 0 blocking reasons. tg128 positive (tierA-qwen4b-q6k) +0.552% CI95 [0.152, 1.065] n=10; control (tierM-gptoss20b-q6k) -0.001% [-0.105, 0.106]; backend_reference 64 steps x 248320 full-vocab logprobs max diff 0; activation marker subject-only. Caveat: ran while another campaign was building on the host (paired interleaving mitigates); quiet rerun required before promotion. gfx1201 run launched.

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
- 2026-09-24T02:34:46.365428+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:39:02.337298+00:00 (updated-by): Updated: section:steps, section:notes
- 2026-09-24T14:09:37.091460+00:00 (state-transition): State: pending → in_progress
- 2026-09-24T14:09:40.423483+00:00 (updated-by): Updated: section:notes
- chg_20260924_141016_five-experimental-rdna-patches_5706
- 2026-09-24T14:10:19.108890+00:00 (updated-by): Updated: section:ledger-events
