---
id: RCD08
order: 8
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:15.711845+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Evidence harvest and reports (replaces summarize.py / noise.py)

## Description

Idempotent harvest stage: verify record run/series/contract/composition identity, copy explicit evidence paths into the canonical checkout, git add explicit paths only, git diff --cached --check, commit, record SHA (never stash, never add -A). Reports per session/series/patch: verdicts, 4-session aggregates, reference ladder, production lane, noise/outlier view; review-ready only when all inputs have a recorded evidence commit.

## Steps

1. harvest.py + gitops.py.
2. report.py porting summarize.py and noise.py into library modules.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline: idempotent harvest, dirty unrelated path rejection; hardware: a completed series harvested and reported without manual scp.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-26T00:52:15.711845+00:00 (created-by): Created by agent
