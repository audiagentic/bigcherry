---
id: QFP18
order: 0
plan: patching-qwen-flash-next
state: completed
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
2. Historical implementation used `tools/lab/flash-next/promote-readme.py` to write per-patch README/evidence records. QFP18 is complete and that one-off generator was retired during 2026-10-08 repo cleanup; the generated patch documentation/evidence is canonical and the implementation remains available in Git history.
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

- Each promoted patch has adoption evidence against its promoted base, activation evidence and passing offline tests on the current pin.
- Whole stack compared with native llama.cpp on the current pin (done: +29% / +45% decode at 8K / 48K).
- Cross-model no-regression with runtime flags at defaults (done: 27B dual-XTX, identical acceptance and greedy text).
- Promoted patches in validated-enhancements, removed from experiment overlays; patch-lint/check/tests pass (except pre-existing 1328 / lab-disposition warnings).
- Done 2026-10-05, commit 74178c4c.

## Notes

Owner agreed 2026-10-04: qualify after the bump. Current states: evaluated (1302, 1303, 1307-1313, 1326, ...), rejected 1304; 1237/1265/1253 already validated but only now entering the Flash-Next profile.

Correction: 1237/1265/1253 are already in every Flash-Next build via the validated-enhancements patch-set; v5 = v4 + 1326.

Replaces the earlier per-patch contract plan (estimated ~3 dev-days + ~2 GPU-days - rejected as too heavy). Existing evidence: v3/v4/v5 adoption ABBAs (flashnext-v2-fusion-ab-3, flashnext-v4-abba, flashnext-v5c-abba), 1312/1313/1326 screens with census/trace/timing evidence, offline tests for every patch. Correction: 1237/1265/1253 are already validated and in every build via validated-enhancements.

Native baseline design (owner 2026-10-05: compare against something useful). Stock does run Flash-Next (flashnext-stock-1/3, 2026-10-02, old pin): -sm tensor -ts 4,4,3, MTP draft on the 6900, 8K decode 65-68 t/s, prefill ~950 t/s at ub512, 1139-1478 t/s at ub1024/2048 (short prompt); its largest load was 192K with q8 KV only. Native run must therefore be: (1) matched-workload speed at depths stock can load with f16 KV (8K, 64K), each side at its own best flags, same model/draft/prompts, greedy identity recorded; (2) a capability fit table (max ctx at f16 and q8 for stock vs v6). Report stock's larger-ubatch prefill advantage at short context honestly. Per-patch attribution still comes from incremental ABBAs; the native run is profile-level only.

Owner 2026-10-05: keep a cumulative promoted base. Rule: the comparison point for a new patch is the current promoted base profile (ordered patch list + env + launch flags + pin), measured as base vs base+patch ABBA. Promotion adds the patch and creates the next profile (v6 -> v7). Evidence records the base id + pin; order dependence is accepted and documented, not re-measured. Earlier evidence stays valid for its base; if a new patch overlaps an earlier patch's code path, its ABBA shows the combined effect, and a leave-one-out (new base minus old patch) is run only if the result looks inconsistent. The native llama.cpp run is profile-level context once per pin bump, not a per-patch gate. Backlog order = profile entry order: v3/v4 fusions (1307-1313, 1308), v5 (1326), v6 (1327); enablers 1291/1292/1294/1297/1302/1303 were in the base from v1.

Owner 2026-10-05: one build only - promote into [patch-set.validated-enhancements] (no separate flash-next patch-set). Env-gated (no-op unless set): 1297, 1303, 1307-1313, 1326, 1327. Ungated or default-sensitive, need a cross-model no-regression check: 1291 (AR CPU-root thresholds), 1292 (no gate), 1294 (verify default), 1302 (always on). Add to the promotion GPU session: one balanced ABBA on the dual-XTX 27B production baseline (validated-enhancements before vs after adding the Flash-Next set), decode + prefill, greedy identity.

## Change Log

- 2026-10-04T07:54:10.353678+00:00 (created-by): Created by agent
- 2026-10-04T07:55:43.580424+00:00 (updated-by): Updated: section:notes
- 2026-10-04T09:25:05.297276+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:notes
- 2026-10-04T14:26:58.996448+00:00 (updated-by): Updated: work='S', section:steps
- 2026-10-04T14:42:55.669115+00:00 (updated-by): Updated: section:notes
- 2026-10-04T14:44:53.920928+00:00 (updated-by): Updated: section:notes
- 2026-10-04T14:47:07.046673+00:00 (updated-by): Updated: section:notes

## Ledger-events



- chg_20261004_161427_one-flag-bigcherry_featuresf_8633
- 2026-10-04T16:14:30.741719+00:00 (updated-by): Updated: section:ledger-events
- chg_20261004_161445_one-flag-bigcherry_featuresf_6153
- 2026-10-04T16:14:48.405399+00:00 (updated-by): Updated: section:ledger-events
- chg_20261004_183256_the-flash-next-speedups-about_6558
- 2026-10-04T18:32:59.802499+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-04T18:33:11.689790+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-10-04T18:33:15.022660+00:00 (state-transition): State: pending → completed
- chg_20261004_235327_model-and-scope-tuning-now-liv_4754
- 2026-10-04T23:53:30.414564+00:00 (updated-by): Updated: section:ledger-events
- chg_20261004_235343_model-and-scope-tuning-now-liv_4846
- 2026-10-04T23:53:46.687293+00:00 (updated-by): Updated: section:ledger-events
