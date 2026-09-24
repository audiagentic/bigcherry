---
id: PRBE08
order: 0
plan: patching-rdna-boost-experiments
state: deprecated
created-at: '2026-09-09T10:54:03.880227+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# RD08-VDR2-NUMERIC: new contract testing RD08's VDR2 kernel under numerical-equivalence (not bit-identical) acceptance

## Description

OBSOLETE (deprecated). PRBE08 asked for a numerical-equivalence (not bit-identical) contract for RD08's VDR2 Q6_K decode kernel, distinct from a blanket bit-identity claim. That work has already been done, on real gfx1201 hardware, as part of patch 1204's own real qualification and terminal disposition (PA40, retry-4, pin 28ff0958/b10901): backend-reference correctness ran across 15 rows with an explicit numerical tolerance (worst error 2.598e-05 < 5e-4) -- exactly the numerical-equivalence framing this item wanted, not bit-identity. Performance was then measured and missed its own threshold (decode effect +0.204%, CI95 [+0.094%,+0.330%], required CI95 low >=0.3% -> FAIL). Patch 1204 is now `rejected`; GPT lifecycle review req_f34f50a25c6240fe records any future VDR2 attempt must return as a new identity with fresh evidence.

## Steps

- Freeze the RD08 VDR2 source identity and distinguish numerical equivalence from bit identity.
- Define tolerances, representative shapes, input distributions, accumulation/precision policy and reference implementation.
- Run deterministic correctness across graph/non-graph and supported architectures, including adversarial values and fallback cases.
- Compare performance only after tolerance gates pass; record failures and do not substitute final-logit-only evidence.
- Publish a contract decision with provenance and explicit promotion/rejection disposition.

## Detailed Solution & Technical Design

The VDR2 kernel may differ bitwise but must meet preregistered numerical tolerances against the reference. Keep RD08's existing safety/eligibility constraints and EC02/EC07 contract evidence visible; no blanket acceptance from a single output sample.

## Code Samples & Guidance



## Files

RD08 VDR2 kernel and patch identity; EC02/EC07 contract fixtures; numerical reference/tolerance harness; graph/non-graph campaign evidence.

## Validation

Numerical-equivalence matrix, adversarial inputs, graph/non-graph, architecture guards, fallback/negative cases, repeatability and performance after correctness.

## Effort & Risk



## Standards

Numerical equivalence is explicit; preserve EC02/EC07 contract boundaries; fail closed on tolerance uncertainty.

## Acceptance Criteria

A written numerical contract is satisfied across the registered matrix with reproducible evidence; bit identity is not claimed; failures retain fallback and block promotion.

## Notes

Supersedes: RD102
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd102

placeholder

2026-09-24 relevance at b11126: DEPRECATED. The exact numerical-equivalence contract PRBE08 wanted (tolerance-based, not bit-identical) was already executed for real as part of RD08/patch 1204's own PA40 demotion campaign on gfx1201: worst backend-reference error 2.598e-05 against an explicit 5e-4 tolerance (correctness PASS), then decode-effect performance measured and missing its own threshold (CI95 low +0.094% < required +0.3%). Patch 1204 is `rejected` (terminal, GPT lifecycle review req_f34f50a25c6240fe: any future VDR2 attempt returns as a new identity with fresh evidence, not a reopening of 1204). Re-running PRBE08 as a separate contract-definition exercise on an already-terminally-rejected patch would duplicate completed work with no promotion path. No GPT design request used -- disposition follows directly from 1204's own SUMMARY.md/patch.toml evidence already on disk.

## Change Log

- 2026-09-09T10:54:03.880227+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:10.226292+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.164993+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.855093+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:35:05.962478+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023529_three-more-rdna-successors-now_3176
- 2026-09-10T02:35:29.316872+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:27:10.705700+00:00 (updated-by): Updated: section:description, section:notes
- 2026-09-24T02:27:21.634133+00:00 (updated-by): Updated: section:notes
- 2026-09-24T02:27:27.039791+00:00 (state-transition): State: pending → deprecated
