---
id: MEN17
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-10-17T09:09:32.756838+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Post-bump hardware acceptance against stored baselines

## Description

Owner-approved (2026-10-17): after a bump, run each engine's declared acceptance configs (radiance: helper-ab speed, helper-kld and 27b-native-kld quality; llama.cpp: its production A/B) and compare against stored baselines with tolerances from measured noise, instead of reading numbers by eye. Includes named, stored KLD references per model/layout (no self-recorded reference in the same run) so the gate does not compare a control against its own reference.

## Steps

1. Baseline store: docs/reference/engines/<engine>/baselines/*.json (metric, value, noise band, run id, pin). 2. Engine declares acceptance configs + baselines in engine.toml. 3. `bigcherry --engine <name> accept` runs them through ab-benchmark and gates against baselines; writes an evidence record. 4. Stored KLD references (path + recorded-on pin/layout) used by quality configs. 5. pin-bump/pin-release can require a passing accept record.

## Detailed Solution & Technical Design

Reuse campaign.acceptance.prepare/run_entry/accept and existing engine adapter. Pass immutable candidate target pin/source/selection/existing build_plan_id, verified binary and runtime home, installed plugin versions from pin_bump.run_phase_accept; each engine.toml entry declares exact lane/experiment, distinct helper lane resolved/built for same target and verified composition, no inherited RAD_BIN/RAD_HOME or old default. Explicit override validated/recorded. Missing/ambiguous build or independent KLD ref refuses before lease; independent stored ref model/layout/tokenizer/corpus/provenance, no same-run reference or regeneration on failure. Add required native dense single/TP2/TP3, helper, quality and actual installed canary/stop, resource and named cache reuse, matched stock paired performance with RHP12 noise/repetitions; qualified nonstock PWR10 apply/restore only where profile intended, no automatic historical tuning. RSC11 current supported YAML roster preflight and full S18 migration matrix; routine release Gate E plus affected paths, RCS23 family gates distinct. Use existing capture/lifecycle owner, not direct proxy or second executor. Explicit build-only/unqualified vs qualified release proof; required skipped/no GPU/missing ref/build mismatch never pass. RSP01/pin-release requires candidate-bound evidence for qualified promotion; preserve deliberate release.

## Code Samples & Guidance



## Files



## Validation

radiance accept on the 1.7.0 pin passes against the 1.3.0 baselines (2,050 pp4096 / 124 tg512 / KLD 0.064); a degraded arm fails.

## Effort & Risk



## Standards



## Acceptance Criteria

**Transferred MEN23 acceptance (independent gate):** MEN17 acceptance entries must declare validated lane + experiment/build identity and fail closed with declaration error on missing, incompatible or ambiguous metadata; fresh-checkout bigcherry --engine radiance accept builds or locates the exact declared lane and sets engine binary/home per entry itself, not from inherited RAD_BIN/RAD_HOME or a stale stock build. A user-specified override must be explicit, validated and recorded, never default. pin-bump --accept reuses precisely the same acceptance-resolver/executor, including helper-ab's rad0002/moe-remote handoff build and 27B/native/KLD reference identity. Add fake-lane offline tests for selected/incorrect/missing lane, environment independence, cache provenance and helper-ab failure before GPU, update Radiance accept docs. Only close when a named real helper-ab entry on the qualified handoff lane succeeds from a clean checkout without manually exported RAD_BIN/RAD_HOME and quality/baseline verdict is attributable; old v1.8.0 helper OOM from using stock build remains historical evidence, not a pass. MEN17's baseline/noise/ABBA post-bump quality gates remain independent and unmet until measured.

Offline fake exact lane and missing/wrong/ambiguous/stale pin, runtime home, binary or provider and inherited env, independent KLD wrong/missing, degraded arm and no GPU/skip refusal. Real helper-ab on qualified handoff lane without manually exported env, independent quality and dense/mixed/helper native lifecycle against exact candidate, RHP12 matched cohorts and PWR10 profile evidence. Actual acceptance run receipts link candidate build and actual policy. pin-bump --accept and standalone accept share executor; required evidence verified by release, no closure on existing code or prior release record. Existing MEN23 acceptance remains.

## Notes

BACKLOG TRIAGE 2026-11-04 absorbed items (superseded into this one; their notes keep the full history): MEN11: Merge target MEN17/MEN26 exist and are open; radiance measurement/evidence paths are used by patch-promote (#543/#546).



CORRECTION 2026-11-04: the earlier note that MEN11 were absorbed into this item is withdrawn. An independent revalidation found each either still has open scope of its own (reopened) or was delivered outright. This item's scope is unchanged.

CONSOLIDATION 2026-11-22 PHASE-5: receive MEN23. MEN23 describes an acceptance entry's build-lane selection defect in MEN17's only engine-aware accept/pin-bump--accept command; independent plan ownership would duplicate the acceptance executor. Original acceptance still applies; the transferred gate is unfinished. Source is superseded for duplicate planning ownership, not completed on implementation, benchmark or live-hardware criteria. Where Vulkan is involved, implementation remains paused by owner directive.

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. Existing MEN23 transfer exact-lane gap remains; run_phase_accept currently passes layout to campaign.acceptance.accept without explicit candidate binding; engine.toml helper-ab/helper-kld comments manual RAD_BIN/RAD_HOME, helper KLD self-records reference, 27b-native not entry. No claim of actual wrong candidate observed in this review.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Current Gate E evidence 2020.11.5 only two install/canary/stop layouts, not exact candidate hardware comparison or quality. Full S18 all supported installed YAMLs (18 currently) once, normal releases existing two Gate E plus affected configs/required cohorts. Exact candidate real provider/row and native ABI/layout, independent KLD reference and actual binary/home binding; no optional gate omission promoting qualified release. Native provider A/B/C same-source, not cross-pin speed attribution.

2026-11-25 serving/configuration closure sweep: docs/design/radiance-serve-config-review-20261125.md. Bind acceptance and bench metadata to actual verified owner/child/endpoint, not /proc --manifest or newest record. Stock Gate E uses named stock YAML, nonstock intended lab arms use one YAML change with separate consent. Existing run evidence and independent KLD ref, no stale/missing provenance pass. S18 full supported config matrix distinct routine affected release gates.

Owner scope correction 2026-11-25: docs/design/radiance-native-yaml-launch.md section Scope and acceptance rule governs. Preserve behavior; smallest native launch and YAML policy change, no generalized machinery. Reuse existing tests/evidence, one shared dense/helper installed acceptance plus materially different affected paths only; cheap parser checks across approved YAMLs. Independent outstanding work stays with its owner, not a blanket S18 prerequisite. Existing exact candidate binding and independent reference correctness remain essential. Use same actual installed dense/helper cohort and affected workload checks as RSC11/RSP01; do not require separate duplicated runs. Keep unrelated existing acceptance tasks independent.

## Change Log

- 2026-10-17T09:09:32.756838+00:00 (created-by): Created by agent
- 2026-11-04T03:47:51.848107+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:45:43.413487+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4528
- 2026-11-22T11:08:28.715849+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- RV4617
- 2026-11-25T02:10:29.120798+00:00 (updated-by): Updated: section:notes, section:detailed_solution, section:acceptance_criteria
- 2026-11-25T02:15:19.358080+00:00 (updated-by): Updated: section:acceptance_criteria
- RV4652
- 2026-11-25T02:50:48.994436+00:00 (updated-by): Updated: section:notes
- 2026-11-25T02:53:31.547189+00:00 (updated-by): Updated: section:detailed_solution
- RV4668
- 2026-11-25T03:31:54.393313+00:00 (updated-by): Updated: section:notes
- 2026-11-25T04:30:40.599872+00:00 (updated-by): Updated: section:notes
