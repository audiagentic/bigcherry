---
id: MEN18
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-10-17T09:09:37.501719+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# One 'run on dev' helper for remote lab runs

## Description

Owner-approved (2026-10-17): every remote run this week was a hand-written script: checkout a ref in a tree that shares the primary work root, export ROCm/env, check GPUs idle and the tree clean, nohup with a log, poll for done/stall. Mistakes repeated (clone without shared work root, stale tree checkout, wrong log path). Provide one command: `bigcherry remote run --host dev --ref <ref> -- <bigcherry args>` (or a tools/admin helper) that does preflight (tree lease, GPU idle, clean checkout at ref in the configured campaign tree), detached execution with a run id and log, and `remote status/wait <run-id>` reporting done/failed/stalled.

## Steps

1. Use host config from environment.local.toml and the tree-activity lease (HI151). 2. Detached run with run id, log, exit code file. 3. status/wait with stall detection. 4. Tests with a local fake host.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

The radiance helper-ab acceptance launched and awaited with two commands.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

BACKLOG TRIAGE 2026-11-04 (MERGE into RSC02; revalidated against main 6eb31a708, verdict CONFIRM): `bigcherry remote run|status|wait|list` exist (tools/bigcherry cli); leftover launch utilities belong to RSC02. Evidence: `bigcherry remote run|status|wait|list` exist (tools/bigcherry cli); leftover launch utilities belong to RSC02. Original content kept below for history.



REOPENED 2026-11-04: an independent revalidation (against main 6eb31a708) rejected the closure recorded above, which had been based on a lighter check. It stays open. Why: tools/bigcherry/cli/remote_run.py:1-33 implements detached remote run/status/wait and tools/tests/cli/test_remote_run.py:119-265 covers preflight, statuses, stall and wait; docs/reference/tooling/REMOTE_RUNS.md:8-10 documents the commands. RSC02.md:18-29 instead owns local bc-serve supervision, not ref-pinned remote command execution, and MEN18.md:33-40's real helper-ab launched-and-awaited acceptance is not evidenced.

## Change Log

- 2026-10-17T09:09:37.501719+00:00 (created-by): Created by agent
- 2026-11-04T03:47:12.143773+00:00 (updated-by): Updated: section:notes
- 2026-11-04T03:47:16.088686+00:00 (state-transition): State: pending → superseded
- 2026-11-04T04:45:09.081314+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:45:13.503420+00:00 (state-transition): State: superseded → pending
