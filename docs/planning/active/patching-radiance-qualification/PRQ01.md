---
id: PRQ01
order: 0
plan: patching-radiance-qualification
state: pending
created-at: '2026-11-15T22:56:49.601626+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P0
work: M
---

# Reconcile source.radiance so all 26 patches apply at the pinned engine

## Description

patch-rebase-check --source radiance on main reports 6 FAILED_NEEDS_RECONCILIATION (rad0013 validated, rad0018, rad0019, rad0020, rad0025, rad0036) and 5 BLOCKED_BY_DEPENDENCY (rad0026-0030). Promotion gates need a fresh clean rebase report, so no promotion can proceed until the source applies. GPT task: gpt-briefs/RECONCILE-radiance-source.md (session ses_b389125d781346c8).

## Steps

1. Find why each anchor misses (earlier patch changed the text, or the probe's pristine source is not v1.17.0; rad0013 matters most since it is validated).
2. Re-anchor without behaviour change; update the patch tests.
3. patch-rebase-check --source radiance reaches 26 clean.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

patch-rebase-check --source radiance reports 26 total, 26 clean; tools/tests/patch tests, patch-lint and forbidden_terms pass.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes



## Acceptance

The production source applies cleanly and a check keeps it that way (PRQ04).

## Change Log

- 2026-11-15T22:56:49.601626+00:00 (created-by): Created by agent
