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



## Code Samples & Guidance



## Files



## Validation

radiance accept on the 1.7.0 pin passes against the 1.3.0 baselines (2,050 pp4096 / 124 tg512 / KLD 0.064); a degraded arm fails.

## Effort & Risk



## Standards



## Acceptance Criteria

**Transferred MEN23 acceptance (independent gate):** MEN17 acceptance entries must declare validated lane + experiment/build identity and fail closed with declaration error on missing, incompatible or ambiguous metadata; fresh-checkout bigcherry --engine radiance accept builds or locates the exact declared lane and sets engine binary/home per entry itself, not from inherited RAD_BIN/RAD_HOME or a stale stock build. A user-specified override must be explicit, validated and recorded, never default. pin-bump --accept reuses precisely the same acceptance-resolver/executor, including helper-ab's rad0002/moe-remote handoff build and 27B/native/KLD reference identity. Add fake-lane offline tests for selected/incorrect/missing lane, environment independence, cache provenance and helper-ab failure before GPU, update Radiance accept docs. Only close when a named real helper-ab entry on the qualified handoff lane succeeds from a clean checkout without manually exported RAD_BIN/RAD_HOME and quality/baseline verdict is attributable; old v1.8.0 helper OOM from using stock build remains historical evidence, not a pass. MEN17's baseline/noise/ABBA post-bump quality gates remain independent and unmet until measured.

## Notes

BACKLOG TRIAGE 2026-11-04 absorbed items (superseded into this one; their notes keep the full history): MEN11: Merge target MEN17/MEN26 exist and are open; radiance measurement/evidence paths are used by patch-promote (#543/#546).



CORRECTION 2026-11-04: the earlier note that MEN11 were absorbed into this item is withdrawn. An independent revalidation found each either still has open scope of its own (reopened) or was delivered outright. This item's scope is unchanged.

CONSOLIDATION 2026-11-22 PHASE-5: receive MEN23. MEN23 describes an acceptance entry's build-lane selection defect in MEN17's only engine-aware accept/pin-bump--accept command; independent plan ownership would duplicate the acceptance executor. Original acceptance still applies; the transferred gate is unfinished. Source is superseded for duplicate planning ownership, not completed on implementation, benchmark or live-hardware criteria. Where Vulkan is involved, implementation remains paused by owner directive.

## Change Log

- 2026-10-17T09:09:32.756838+00:00 (created-by): Created by agent
- 2026-11-04T03:47:51.848107+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:45:43.413487+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4528
- 2026-11-22T11:08:28.715849+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
