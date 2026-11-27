---
id: VTP60
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-27T10:22:17.436914+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P1
work: M
---

# Promote fast scheduler and model-layout qualification to canonical gates

## Description

Turn VTP59 host-burst proof into repeatable, provenance-bound fast tests. Reuse actual native scheduler, CTest, RCS26–28 affected-family resolution, existing replay/cost captures, MRC08 and MEN17. Do not own production tuning or replace integrated serving gates.

## Steps

S1 native host fixture and resolved defaults; S2 workload descriptors and structural regression oracles; S3 replay cost evidence and affected-family link; S4 portable screening receipts and bump integration; S5 mutation, candidate and hardware correlation validation.

## Detailed Solution & Technical Design

See docs/design/radiance-fast-qualification.md sections 1–9: exact native CTest, effective defaults, workload selection, measured-cost replay, evidence routing and acceptance. S1/S2 independent of GPU and MRC08; S3–5 consume existing owners. Review findings incorporated.

## Code Samples & Guidance



## Files

docs/design/radiance-fast-qualification.md; existing owners only. VTP59 retains tuning, RCS26–28 retain kernels, MRC08 retains record/export and MEN17 retains final acceptance.

## Validation

Native exact build, fake linkage, no model/GPU; fail missing/skipped tests, stale core, malformed cost and unknown geometry. Mutation controls for known overlap loss, default drift, slot/admission/cancel/count; validate cost uncertainty, actual row evidence and held-out hardware. Verify existing record/export privacy, exact candidate/reference, failed host gate prevents GPU. See design §8.

## Effort & Risk



## Standards



## Acceptance Criteria

- [ ] Candidate-native CPU tests and effective defaults gate run without full engine launch; negative controls demonstrably fail.
- [ ] Versioned sparse workload and affected-boundary selection, exact per-sequence invariants; no duplicate taxonomy.
- [ ] Cost inputs and uncertainty validated; unmatched rows/costs explicitly nonpass; independent held-out calibration.
- [ ] Existing RCS and MRC08/MEN17 integration; exact build identity, explicit coverage, failed/skipped/synthetic-only cannot pass release.
- [ ] Targeted real GPU/serving witnesses and independent quality remain required; document actual time savings, merge verified slices.

## Notes



## Change Log

- 2026-11-27T10:22:17.436914+00:00 (created-by): Created by agent

## Reviews

- 2026-11-27T10:25:47.614063+00:00 (updated-by): Updated: section:detailed_solution, section:validation, section:acceptance_criteria
- RV4753
