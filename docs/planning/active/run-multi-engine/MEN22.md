---
id: MEN22
order: 22
plan: run-multi-engine
state: pending
created-at: '2026-10-18T13:53:54.146144+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# pin-bump lane-build refuses the dirty checkout its own release record leaves

## Description

First real radiance bump (v1.7.0 -> v1.8.0, 2026-10-19): after drift, compile and the component gates passed, the lane-build phase (MEN16) stopped with LANE_BUILD_FAILED: 'BigCherry repository is dirty; use explicit development override'. The dirty files were releases/index.json and releases/v1.8.0.json, written by the orchestrator itself in earlier phases. The orchestrator must leave a clean tree for the phases that need one: commit the release record (in its own commit, like the repin) before lane-build, or have the lane build ignore exactly the orchestrator's release-record paths. Do not pass a blanket allow-dirty.

## Steps

1. Decide where the record is committed (before lane-build) and that complete/reaudit still update it.
2. Fix in pin_bump.py; offline test with a fake lane that asserts a clean tree at lane-build.
3. Check the llama.cpp path for the same ordering.
4. Document in docs/reference/build/PIN_BUMP.md.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline test; the next real bump (any engine) passes lane-build without a manual commit.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Worked around on the bump branch by committing the record by hand. Found by the radiance 1.8.0 bump.

## Change Log

- 2026-10-18T13:53:54.146144+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261018_140640_radiance-is-now-on-upstream-1_2179
- 2026-10-18T14:06:48.708564+00:00 (updated-by): Updated: section:ledger-events
