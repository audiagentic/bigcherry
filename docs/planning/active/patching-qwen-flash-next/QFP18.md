---
id: QFP18
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-04T07:54:10.353678+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Lightweight evidence-reuse promotion to validated for the Flash-Next stack

## Description

Owner direction 2026-10-04: the path to 'validated' must be quick and reuse evidence we already produce; no heavyweight per-patch 4-session contract campaigns or bespoke producers for patches that already have strong evidence. Reserve full contracts for patches meant for broad reuse or with weak/ambiguous evidence.

## Steps

Finding (2026-10-05): the current tooling already allows a cheap path. None of these patches are RD-bound (plan ids QFN/QFP/RNX/PRBE), so check_validation_packages does not require a contract or validation.toml. Promoting to validated only needs (a) patch-lint clean, (b) for optimization-tagged patches (1292, 1297, 1307-1313, 1327) a README.md with a native llama.cpp baseline comparison (check_performance_evidence marker), and (c) a deliberate lifecycle state flip with the evidence referenced.

Plan (about 1 dev-session + about 1 GPU-hour, all patches in one pass):
1. GPU, about 45 min, one queue: native-vs-v6 ABBA at 8K and 64K on pin 0504396 (stock-none vs deploy-v5-plus-1327, same flags/ts/ctx, f16 KV). It supplies the native llama.cpp arm for every patch at once.
2. Generator script tools/lab/flash-next/promote-readme.py: writes patches/<id>/README.md from a per-patch table (mechanism, env flag, activation evidence path, incremental ABBA run and delta, profile it joined, native vs v6 numbers from step 1, greedy identity hashes) and appends evidence/validation.json entries (append-only).
3. Per-patch evidence map (existing runs, nothing new):
   - incremental ABBA with separation: 1326 (v4 to v5, +9.4%/+7.8%), 1327 (post-bump half-recovery, kernel census 1053 to 1027), 1312/1313 screens + census, 1307-1311 v2/v3 fusion ABBAs (flashnext-v2-fusion-ab-3), 1308 rollback ABBA;
   - enabler/functional, neutral rationale: 1291, 1292, 1294 (determinism), 1297 (draft vocab), 1302 (OOM evict), 1303 (KV split; needed for 240K fit);
   - activation: BIGCHERRY_* env gates + census/trace logs already captured.
4. Flip state to validated in one commit via the lifecycle skill; run patch-lint, check, and the patch tests; record one ledger event.
5. After that, any new winner (e.g. 1330) is promoted in the same pass as its adoption ABBA: README row + state flip, no extra GPU time.

Risks: one profile-level native arm attributes the stack, not each patch; per-patch attribution relies on the incremental ABBAs listed. 1307-1311 incremental runs were on the old pin; the step 1 v6 run on 0504396 re-confirms the whole stack on the current pin (owner rule: reuse existing evidence).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

patch-verify-evidence passes on the new pin; lifecycle promotion recorded with evidence.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner agreed 2026-10-04: qualify after the bump. Current states: evaluated (1302, 1303, 1307-1313, 1326, ...), rejected 1304; 1237/1265/1253 already validated but only now entering the Flash-Next profile.

Correction: 1237/1265/1253 are already in every Flash-Next build via the validated-enhancements patch-set; v5 = v4 + 1326.

Replaces the earlier per-patch contract plan (estimated ~3 dev-days + ~2 GPU-days - rejected as too heavy). Existing evidence: v3/v4/v5 adoption ABBAs (flashnext-v2-fusion-ab-3, flashnext-v4-abba, flashnext-v5c-abba), 1312/1313/1326 screens with census/trace/timing evidence, offline tests for every patch. Correction: 1237/1265/1253 are already validated and in every build via validated-enhancements.

## Change Log

- 2026-10-04T07:54:10.353678+00:00 (created-by): Created by agent
- 2026-10-04T07:55:43.580424+00:00 (updated-by): Updated: section:notes
- 2026-10-04T09:25:05.297276+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:notes
- 2026-10-04T14:26:58.996448+00:00 (updated-by): Updated: work='S', section:steps
