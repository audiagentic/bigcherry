---
id: RSC14
order: 0
plan: run-serving-configuration
state: pending
created-at: '2026-11-19T22:22:14.183742+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Flash-Next production start fails out of memory when builds run beside it: make host-RAM admission robust

## Description

2026-11-20: the no---set production smoke of installed bc-radiance-1170.34.0 (flashnext-4gpu, RAM n-gram table 47.7 GiB) stopped with 'E startup stopped: out of memory' while C++ builds ran on the host; the rerun minutes later (MemAvailable ~79 GiB) passed. bigcherry-2 saw the same borderline earlier (engine admission needed ~74.8 GiB, 74.13 available). The admission is correct to refuse, but a production start that fails whenever a build or page cache grows is fragile: the plan admits at owner start, and the engine's own host-room check happens later after memory moved.

## Steps

1. Measure the gap between the owner's admission figure and the engine's startup check (what is allocated between them; the RSC08 ledger margin is 1024 MiB) and what concurrent load does to MemAvailable during the ~2 min load.
2. Decide the fix from evidence: reserve/pin the host memory at owner admission so the engine cannot lose it, or retry once with a bounded wait when the shortfall is transient (with the reason logged), or a lease/priority rule that heavy builds do not start beside a production start; no hand-picked margins (derive from measured startup consumption).
3. A production start beside a build either succeeds or refuses before loading with a clear reason.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Reproduce: production start of flashnext-4gpu while a radiance lane build runs; it succeeds or refuses before load with a named reason; owner and engine admission agree.

## Effort & Risk



## Standards



## Acceptance Criteria

No late out-of-memory stop of a production start caused by memory moving between admission and the engine's check.

## Notes



## Change Log

- 2026-11-19T22:22:14.183742+00:00 (created-by): Created by agent
- 2026-11-19T22:22:32.846506+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
