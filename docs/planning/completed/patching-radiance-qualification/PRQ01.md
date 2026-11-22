---
id: PRQ01
order: 0
plan: patching-radiance-qualification
state: completed
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

The original source-patch drift was diagnosed and resolved, and a check refuses off-pin vendor HEAD; the current 2.2 production source is governed by the maintained strict source-class policy.

## Notes

2026-11-22 reviewed closure RV4326: Historical vendor-pin rebase incident resolved; pinned source changed to 2.2.0 and guard retained in patch/rebase.py. The old 26/26 figure is historical, not a fresh 2.2 check. Evidence: docs/evidence/2026-11-22-prq01-prq05-reconciliation/README.md. Focused negative/positive source-class regression 11 passed, 5 subtests. No new GPU, throughput or code changes; preserve 2.2 RUM/LTX owners.

## Acceptance

The production source applies cleanly and a check keeps it that way (PRQ04).

## Change Log

- 2026-11-15T22:56:49.601626+00:00 (created-by): Created by agent

## Progress 2026-11-16

Cause found: patch-rebase-check takes its pristine source from vendor/radiance HEAD, which had been left on upstream 451b136 (58 commits past the v1.17.0 pin 19d4e8f) with 25 leftover patched files since Nov 8. Restored to the pin (the leftovers are backed up outside the repo); `patch-rebase-check --source radiance` now reports 26 total, 26 clean. No patch needed re-anchoring.

Remaining: a guard so the vendor checkout cannot sit off the pin unnoticed (patch-rebase-check and build should refuse or warn when vendor HEAD is not the recipes pin).

## Reviews

- RV4326
- 2026-11-22T04:58:27.568166+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- 2026-11-22T04:58:37.913003+00:00 (state-transition): State: pending → completed
