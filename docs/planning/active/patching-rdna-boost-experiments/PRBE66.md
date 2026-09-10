---
id: PRBE66
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:09.575084+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# HIP-GRAPH-001: Selective HIP graph bypass for unstable FA shapes

## Description

Add the narrowest HIP graph bypass for unstable long-context FA shapes on gfx1201, while preferring PRBE67 recapture if root cause is proven.

## Steps

Reproduce R9700 100K+ FA graph failure and log graph shape/stream/capture state; compare short context, other ops, and graphs-off controls; implement a narrow eligibility bypass only for the failing kernel family/depth; verify no crash/output drift and measure PP/TG/graph overhead; hand off root-cause evidence to PRBE67.

## Detailed Solution & Technical Design

Disable graph capture only where stale/unstable long-context FA is proven to fail. Preserve graph benefits globally and make the predicate shape/depth/backend specific.

## Code Samples & Guidance



## Files

HIP graph eligibility/cache bypass; reproduction/instrumentation logs; long-context correctness and performance evidence.

## Validation

No crash and output parity; PP/TG, graph overhead, capture/recapture events, repeated 100K+ runs.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Use the narrowest bypass that eliminates reproducible failure without broad graph regression; prefer PRBE67 when robust recapture fixes the root cause.

## Notes

Supersedes: RD83
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd83

## Change Log

- 2026-09-09T10:58:09.575084+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:15.987575+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.426062+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.254495+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:17:45.266636+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031805_repaired-four-more-graph-and-v_2834
- 2026-09-10T03:18:05.350207+00:00 (updated-by): Updated: section:ledger-events
