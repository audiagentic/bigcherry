---
id: MEN20
order: 20
plan: run-multi-engine
state: pending
created-at: '2026-10-18T13:04:07.928445+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# bigcherry command for component-setting evidence (replaces helper-evidence.sh)

## Description

Promoting a component setting (libr3 now, moe-remote/backends after RCS04) needs one A/B per setting with an activation record, staged as run.json under docs/evidence/<item>/ and referenced from component.toml. Today helper-evidence.sh does this by looping ab-benchmark. Make it a bigcherry command: `bigcherry component-evidence --engine radiance --component <c> --setting <S>` reads the setting's declared A/B config, builds the lane, runs ab-benchmark on a idle-checked host, verifies the evidence rules (arms differ in that setting alone, executed activation, no failed cell), stages the run.json and prints the component.toml edit. Optional --promote edits the status after verification.

## Steps

1. Declare per setting in component.toml: ab-config path and activation marker (already there for marker).
2. Implement the command on top of ab-benchmark and the evidence verifier used by patch-verify-evidence.
3. Idle-host check and graceful teardown reused from ServerRunner.
4. Offline tests with fixture run.json files; one real run reproducing a RAD06 result.
5. Retire helper-evidence.sh and its TOOL_DISPOSITION row.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A real run of one setting through the command yields the same evidence an earlier script run staged; offline tests cover each failure.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Part of MEN14 consolidation. Component lint (RCS08) reads the same evidence.

## Change Log

- 2026-10-18T13:04:07.928445+00:00 (created-by): Created by agent
