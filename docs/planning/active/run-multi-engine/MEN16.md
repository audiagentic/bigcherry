---
id: MEN16
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-10-17T09:09:28.078742+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# pin-bump builds its own lane and runs the CPU smoke on it

## Description

Owner-approved (2026-10-17) automation from the first radiance bump: today the bump is build lane -> pass the build dir to pin-bump --cpu-smoke-build -> pin-bump. pin-bump should build the bumped engine itself through the canonical lane (source from --source, build profile, platform from --platform) after rebase/apply, record the build identity in the release record, and run the cpu-smoke phase against that build. --cpu-smoke-build stays only as an explicit override recorded in the resume state.

## Steps

1. Add a 'lane-build' phase before cpu-smoke using campaign build machinery (no reimplementation). 2. Record build_plan_id and binary hashes in the release record. 3. cpu-smoke consumes it. 4. Tests with fake build runner; docs PIN_BUMP.md.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A radiance pin-bump run with only --source and --platform completes on Brutus; offline tests for phase order and resume.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-17T09:09:28.078742+00:00 (created-by): Created by agent
