---
id: TRVP13
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:01:04.855235+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Stack/build inspection and release completeness

## Description

Close the common provider/stack framework with inspection, candidate completeness, exact replay mismatch policy, and end-to-end lineage from request through promotion.

## Steps

1. Expose persisted stack/build/provider/candidate identity through stack-info and build-info. 2. Add a table-driven completeness matrix across config, CLI/campaign, probe, BuildPlan classification, manifest, DB, inspection, and replay; an unclassified provider field must fail tests. 3. Enforce unresolved_selectable == 0 and classify omissions as stack mismatch, provider mismatch, or provider-candidate missing. 4. Verify replay projection: vendor_explicit requires exact provider kind/fingerprint/architecture/candidate; vendor_auto fingerprint changes retune by default. 5. Trace a final tune-campaign winner from campaign through BuildPlan, runtime stack, manifest, candidate, measurement, promotion, replay, and release artifact self-description. 6. Publish closure gates consumed by TRVP15 without absorbing CM1 backend qualification.

## Detailed Solution & Technical Design

TRVP13 is shared closure, not another kernel/provider implementation. Keep stack-info/build-info, mechanical completeness, replay mismatch policy, and lineage verification coherent. Persist provider/build/runtime identity at every projection, reject missing or mismatched fields, and make auto retune policy explicit. CM1-specific qualification remains TRVP15.

## Code Samples & Guidance



## Files

stack-info/build-info CLI; replay_projection.py; promotion_gate.py; replay_cache.py; completeness/report tooling; field-matrix and end-to-end lineage tests; release artifact inspection.

## Validation

A final tune-campaign trace demonstrates campaign→BuildPlan→runtime stack→manifest→candidate→measurement→promotion→replay with exact identities. Verify unresolved_selectable==0, all matrix fields are classified, stack/build/runtime/provider agreement holds, exact replay succeeds, provider substitution and ROCm 7.14→10 replay are rejected, and release artifacts self-describe.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Mechanical completeness is table-driven and fails on unclassified provider fields or unresolved selectable candidates. Exact provider/build replay succeeds only for matching identity; mismatch diagnostics are classified and fail closed. End-to-end lineage and release inspection are reproducible; TRVP15 consumes these gates without duplication.

## Notes

Supersedes: RO18
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro18

Supersedes RO18. Depends on TRVP03/TRVP12 and remains shared closure. Preserve patch 1225, exact identity, and governance.

## Change Log

- 2026-09-09T11:01:04.855235+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:17:59.833087+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.588209+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.448900+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T01:03:14.213881+00:00 (updated-by): Updated: section:validation
- chg_20260910_010342_successor-plans-now-have-expli_8662
- 2026-09-10T01:03:42.596223+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:46:49.517458+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034835_repaired-the-final-six-live-su_3009
- 2026-09-10T03:48:35.238800+00:00 (updated-by): Updated: section:ledger-events
