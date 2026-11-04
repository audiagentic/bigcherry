---
id: MEN19
order: 0
plan: run-multi-engine
state: superseded
created-at: '2026-10-17T09:09:41.576164+00:00'
breadth: ''
skill: basic
created-by: agent
priority: P2
work: S
---

# Bump-time native rewrite drift check, and pin-release confirm-then-run

## Description

Owner-approved (2026-10-17): (a) radiance pin-bump runs r3_rewrite.py offline against the new upstream as an early phase and reports failing native blocks per file (the 1.7.0 port found 38 only by hand); generic via an adapter check. (b) pin-release runs all stages in one go with a single explicit confirmation before tagging.

## Steps

1. Adapter rewrite-drift check before compile. 2. pin-release --confirm prompt / flag before the tag phase. 3. Tests.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A bump against a tree with a broken native block stops in the drift phase with the block list.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

BACKLOG TRIAGE 2026-11-04 (MERGE into MEN24; revalidated against main 6eb31a708, verdict CONFIRM): pin-bump has the drift phase (NATIVE_REWRITE_DRIFT) and caught the v1.16 rewrite break; remaining pin-release confirmation work is MEN24. Evidence: pin-bump has the drift phase (NATIVE_REWRITE_DRIFT) and caught the v1.16 rewrite break; remaining pin-release confirmation work is MEN24. Still referenced by: MEN24. Original content kept below for history.



CORRECTION 2026-11-04: the independent revalidation found this item delivered outright, not merged into MEN24. Evidence: tools/bigcherry/engine_adapters/radiance_bump.py:232-283 and release/pin_bump.py:466-473 run native rewrite drift before compile, with tests/release/test_native_rewrite_drift.py:64-132. release/pin_release.py:928-979 implements the one confirmation before tagging (--yes or interactive, fail closed), exercised by tests/release/test_pin_release_required_checks.py:315-330; MEN24.md:18-30 is further bump-speed/nightly upkeep, not the missing half.

## Change Log

- 2026-10-17T09:09:41.576164+00:00 (created-by): Created by agent
- 2026-11-04T03:47:20.057701+00:00 (updated-by): Updated: section:notes
- 2026-11-04T03:47:24.019192+00:00 (state-transition): State: pending → superseded
- 2026-11-04T04:45:17.878941+00:00 (updated-by): Updated: section:notes
