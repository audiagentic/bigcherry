---
id: BCOP115
order: 115
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-10T12:04:19+00:00'
created-by: agent
priority: P1
work: S
---

# RD07/1267: gate Q6_K MMQ fold on safe J padding and production 1006 baseline

## Discovery / change

Pinned b11474 retains MMQ staging-padding selection that upstream [#29953](https://github.com/ggml-org/llama.cpp/pull/29953) fixed (merged 2026-10-08); source-derived RDNA4 Q6_K ne11=17 can reserve J=16 while selecting J=32, subject to the shared-memory gate. Upstream [#30168](https://github.com/ggml-org/llama.cpp/pull/30168) (merged 2026-10-09) additionally aligns host precision config. This is a **source/host witness**, not a reproduced GPU OOB. 1267's host dispatch marker does not prove modified MMA-kernel execution. Its historical +2.4675% gfx1201 pp512 result predates production 1006; 1006 independently measured ~+18% and is now in validated-enhancements. No additive or current-pin 1267 gain is established.

## Ownership / existing work / exclusions

PRBE110/1267 owns scale-fold and the terminal decision; PRBE04 owns PEF01/HI71 safety/shape qualification. PA35/1006 owns validated Q6_K float-promotion; 0300 owns J; upstream-fix/pin owners own generic #29953/#30168 parity. Rejected 1203/1266 stay terminal. Existing 1267 producer/tests are reused. No implementation, patch, recipe, benchmark definition, queued experiment, or protected neighbouring capability was changed. Last substantive PRBE110 update 2026-10-08; last 1267 path commit 2026-10-09 was a mechanical directory refactor, not independent 1267 development. The 12-hour exclusion passed. Protected activity: Radiance gfx1100 FP8/MXFP4/KL/serving, Flash-Next 1357 router/QSA and 1330/1334/1347, QFP43/DFlash, recent Meta and MTP work. This is not a repeat of BCOP105–114.

## Unresolved action / terminal disposition

Before any GPU timing: prove current-pin J allocation/launch safety with host boundary fixtures, recompose 0300+1006+1267, run existing mechanics and PEF01/HI71 controls, prove actual Q6_K MMA route and numerical/graph/multi-request parity. If the non-overlapped end-to-end ceiling is <3%, **retire 1267 without a new campaign**. Otherwise PRBE110 may run four independent gfx1201 sessions with >=10 paired ABBA rounds, CI95-low >=3% end-to-end and <=1% controls. No duplicate scheduler, cache, dispatch table, allocator, telemetry or hardware queue.

## Evidence

First-party b11126 1267 gfx1201 pp512 +2.4675% (CI95 [2.2776%,2.6508%]), tg128 -0.0331%; validated 1006 separate ~+18% pp512. Host-only 200,000-case Q6_K scale-chain fixture: zero nonzero numerical differences and 391 signed-zero bit differences; static RDNA4 Q6_K J fixtures identify underpadding candidates. Pinned/current upstream source and PR diffs inspected; vLLM GGUF Q6_K implementation inspected as an unqualified alternative. No repository pytest, HIP build, GPU execution or fresh benchmark performed. Full technical detail and acceptance conditions are in PRBE110.
