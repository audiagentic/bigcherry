---
id: PRBE08
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:03.880227+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# RD08-VDR2-NUMERIC: new contract testing RD08's VDR2 kernel under numerical-equivalence (not bit-identical) acceptance

## Description

Define and qualify RD08 VDR2 under numerical-equivalence acceptance. This is an open contract item, not a bit-identical claim; preserve the existing EC02/EC07/RD08 evidence boundary.

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
