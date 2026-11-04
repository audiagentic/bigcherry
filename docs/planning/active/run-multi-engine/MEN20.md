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



BACKLOG TRIAGE 2026-11-04 (MERGE into RCS08; revalidated against main 6eb31a708, verdict CONFIRM): RCS08 (component status/evidence CLI) exists and is open; no duplicate command. Evidence: RCS08 (component status/evidence CLI) exists and is open; no duplicate command. Still referenced by: RAD10, MEN14, WSL05. Original content kept below for history.



REOPENED 2026-11-04: an independent revalidation (against main 6eb31a708) rejected the closure recorded above, which had been based on a lighter check. It stays open. Why: MEN20.md:18-31 specifies a bigcherry component-evidence command that launches, verifies and stages a per-setting run and optionally promotes it. RCS08.md:18-27 only defines lint on existing setting declarations/evidence, with no runner or promotion command; the only literal component-evidence verb in the current tree appears in MEN20's plan.

## Change Log

- 2026-10-18T13:04:07.928445+00:00 (created-by): Created by agent
- 2026-11-04T03:47:27.994982+00:00 (updated-by): Updated: section:notes
- 2026-11-04T03:47:31.963094+00:00 (state-transition): State: pending → superseded
- 2026-11-04T04:45:22.125658+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:45:26.281835+00:00 (state-transition): State: superseded → pending
