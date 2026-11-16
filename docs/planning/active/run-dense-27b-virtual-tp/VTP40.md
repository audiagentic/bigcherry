---
id: VTP40
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-14T06:57:56.876565+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Speculation window by load on TP3 int8

## Description

At one stream a decode step carries 4 rows (dflash2, 3 drafts) where the XTX int8 GEMV is flat to 9 rows and the R9700's to 64; at 8 streams 32 rows already sit in the XTX row's expensive band (VTP37 step 3a). The window is fixed at 3. One stream keeps 2.67 tokens a step. The scheduler has `--draft-window adaptive` and exports per-position acceptance; whether the DFlash2 container admits another depth (`draft_depth_fixed`) is not established.

## Steps

1. Per-position acceptance from `/metrics` at 1 / 4 / 8 streams.
2. Whether the drafter admits other depths.
3. Arms: depth 3 / 5 / 7 fixed and `draft_window adaptive`, 1 / 4 / 8 streams, decode-only tok/s.
4. A named config variant if it pays.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Greedy identity (verification is exact), int8 stress (rad0028 guard), 2x ABBA.

## Effort & Risk

S. Expected one stream +0% to +20%; nothing or negative at 6-8 streams unless the window follows load. Kill: conditional acceptance at position 3 under 0.5, or the drafter refuses other depths.

## Standards



## Acceptance Criteria

- [ ] One-stream decode-only tok/s +8% or more with no loss at 4 / 8 streams, or the acceptance curve that shows why not.

## Notes

Created 2026-11-14 (`docs/evidence/2026-11-14-tp3-long-context/README.md`). ENG02 is the Flash-Next equivalent; VTP37 step 3a changes the cost of rows at 8 streams.

## Change Log

- 2026-11-14T06:57:56.876565+00:00 (created-by): Created by agent
- 2026-11-14T06:58:48.037448+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
