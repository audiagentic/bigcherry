---
id: RCD05
order: 5
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:05.135167+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Job runner: commit pinning, runner worktrees, GPU visibility, disk guards, stall detection, log retention

## Description

Per-attempt detached runner worktree at the commit resolved immediately before spawn (vendor/llama.cpp symlinked, BIGCHERRY_ENVIRONMENT exported), GPU visibility from configured physical devices (Host.gpu_visibility_env / platform allocation, no raw VIS=), root + work filesystem guards before and during a job, stall = all progress channels idle (stdout, artifacts, process CPU, phase markers), bounded server-log retention with byte-count/sha records, TMPDIR/ccache under the work root with CCACHE_COMPILERCHECK=content.

## Steps

1. resolve.py/monitor.py/worker.py per design.
2. Retention policy for runner worktrees (24 h harvested, 7 d failed).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline tests for commit re-check, worktree layout, disk guard, stall channels; hardware: a pull mid-queue never changes a running attempt.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-26T00:52:05.135167+00:00 (created-by): Created by agent
