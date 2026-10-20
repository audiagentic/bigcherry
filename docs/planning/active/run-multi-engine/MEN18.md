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

# One 'run on Brutus' helper for remote lab runs

## Description

Owner-approved (2026-10-17): every remote run this week was a hand-written script: checkout a ref in a tree that shares the primary work root, export ROCm/env, check GPUs idle and the tree clean, nohup with a log, poll for done/stall. Mistakes repeated (clone without shared work root, stale tree checkout, wrong log path). Provide one command: `bigcherry remote run --host brutus --ref <ref> -- <bigcherry args>` (or a tools/admin helper) that does preflight (tree lease, GPU idle, clean checkout at ref in the configured campaign tree), detached execution with a run id and log, and `remote status/wait <run-id>` reporting done/failed/stalled.

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

## Change Log

- 2026-10-17T09:09:37.501719+00:00 (created-by): Created by agent
