---
id: RCD04
order: 4
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:00.974407+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# bigcherry jobs MVP: JobSpec, series identity, failure taxonomy, JSON CLI

## Description

Thin BigCherry layer over the chosen platform (docs/design/JOBS_ORCHESTRATOR.md MVP): typed JobSpec (no extra_args), batch expansion with overrides, series freezing N/contract hash/base revision/focal+common+validated digests, run_id stable with attempts on harness retry, FailureKind taxonomy (retryable vs done-invalid vs block-series; scientific FAIL = done-fail), N+1 session rejection, disable/enable preserving slots. CLI: bigcherry jobs submit|submit-batch|list|show|retry|disable|enable|cancel|pause|resume with --json.

## Steps

1. tools/bigcherry/jobs/{model,db,failure,resolve,batch}.py + cli/jobs.py.
2. State store (SQLite WAL + append-only events, or Slurm sacct + thin identity DB per RCD02).
3. Offline tests from the design's named list.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline unit tests (the 43 named tests in the design, adapted to the platform choice).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-26T00:52:00.974407+00:00 (created-by): Created by agent
