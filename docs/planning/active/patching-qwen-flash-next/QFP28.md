---
id: QFP28
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-05T04:10:44.361443+00:00'
breadth: ''
skill: intermediate
created-by: claude
priority: P2
work: M
---

# Triage Shali12/r9700-flash-next-notes findings into Flash-Next gates, profile and patches

## Description

Source: https://github.com/Shali12/r9700-flash-next-notes (FINDINGS.md, results/speed-stage1d/1e, options-and-memory; read 2026-10-05 via page summaries, numbers approximate). Their rig: one R9700 (gfx1201, 32 GB, ROCm 10.0.0, kernel 6.17) with most expert layers in system RAM (-ncmoe 35..47, MOE_EXPERT_CACHE_MIB=4096, MOE_EXPERT_CACHE_DEVMAP=1), stew675/llama-cpp-rdna-boosts r30 (v16, f108261) over llama.cpp 84e76d8a2 / b11173, KV q8_0. Ours is an all-GPU tensor split (2x gfx1100 + gfx1201, gfx1030 MTP drafter), so -ncmoe / expert-cache / --lazy-mode --load-mode tuning does not transfer directly. Candidate take-aways below need assessment (GPT review requested) before any becomes work.

## Steps

1. Multi-request text gate. Their r29/r30 expert gather path (GGML_SCHED_DEVGATHER, default on until r31, stew675 issue #85) answered request 1 correctly then emitted '////' at full speed on every later request, on three quants; all gather-on speeds (1300-1500 t/s prefill vs ~630 off) were void. Our greedy-md5 identity check is single-request. Proposed: lab gate = 3 known-answer prompts on one server instance (one-line completion, runnable function, ~8K summary), fail on wrong answer or 200+ identical non-whitespace chars, run before any speed number; apply first to 1295 (QSA gather, v7 not run-to-run stable) and 1332 (chunked text differs from dense).
2. Check lineage: does 1295 or any tracked stew675-rdna-boosts snapshot share code with the DEVGATHER path; is the r30 patch repo (stew675/llama-cpp-rdna-boosts) covered by config/external-sources.toml; what changed in r31+.
3. Q3 quant for context >300K: ISTA GSQ-RCO IQ3_XXS 75.8 GB vs AtomicChat Q4_K_M 94.5 GB (Unsloth Q3_K_XL 90.0 GB, IQ4_XS 93.7 GB - not worth it). ISTA is calibrated on xhigh reasoning traces; authors say the gap grows at lower effort. A fit arm needs a quality check at medium effort, not only a load test. KV stays f16/q8_0.
4. Draft depth: depth 3 beat depth 5 (acceptance 0.80-0.90 vs 0.63-0.81; 52-55 vs 48-54 t/s). ABA --spec-draft-n-max on our gfx1030 drafter; winner into profile/flashnext.ini.
5. Sampling/reasoning profile: model card min_p 0.0 (llama.cpp default 0.05, measured negligible); xhigh run looped emitting '9' after 28K thinking tokens (medium: 4,645 tokens, 2.2 min, 29/29). Consider --min-p 0 and a --reasoning-budget cap in profile.
6. Grammar bug: tools + JSON schema -> HTTP 400 'failed to parse grammar ... expecting newline or end at ::= | " " | "\n"{1,2} [ \t]{0,20}' on base 84e76d8a2 (tool-eval-bench TC-65/66/67/69). Reproduce on our pin (050439614); if present, upstream-fix patch candidate (10xx).
7. 'seq_rm: rollback crossed a batch boundary' warning when one slot prefills while another drafts; their set has GGML_CUDA_GDN_CHUNKED. Compare that chunking mechanism with 1332 and grep our multi-slot logs.
8. Power cap: 220 W vs 300 W cost ~3% prefill, -9 C. Optional arm to test whether a cap reduces Brutus run-to-run drift.
9. Observations only: draft head costs ~3 GB VRAM/slot growing with prompt; GPU working memory follows the deepest slot not the slot total; decode falls 37 -> 27.5 t/s from short to 130K context (no draft), 20.4 t/s at 262K.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Each adopted step lands as its own change with lab script under tools/lab/flash-next/, results in this item, profile changes in profile/*.ini. Text gate must demonstrably fail on a known-corrupt arm (or injected corruption) before it is trusted.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Not verified locally: DEVGATHER lineage vs 1295, external-sources coverage of the r30 patch repo, grammar bug on our pin. GPT assessment requested 2026-10-05.

## Change Log

- 2026-10-05T04:10:44.361443+00:00 (created-by): Created by claude
- 2026-10-05T04:11:07.895131+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:notes
