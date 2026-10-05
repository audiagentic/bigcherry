---
id: BCOP04
order: 4
plan: bc-optimizations
state: pending
created-at: '2026-10-05T04:33:00+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Close or reopen QFP11 from measured AllReduce boundary cost

## Description

Follow-through for the QFP11 audit that found captured-graph/AllReduce boundary overhead much smaller than rank-arrival skew. QFP11 should remain retired unless post-balancing measurements make boundary overhead material.

## Steps

1. Preserve AR timing decomposition into arrival skew, collective and resume/boundary cost.
2. Re-run representative short/long-context lanes after current QFP07/QFP09/QFP13 work.
3. Keep QFP11 closed if boundary overhead remains <1 ms/token and <3% decode wall.
4. Reopen only with measured evidence; reuse existing graph-cache/replay ownership rather than adding a scheduler.
5. Ensure QFP09/QFP13/QFP11 ownership text remains consistent.

## Related

QFP07, QFP09, QFP11, QFP13, QFP01/1291.

## Acceptance Criteria

- Current boundary-cost evidence is recorded after latest balancing/fusion changes.
- QFP11 is explicitly closed or reopened against numeric gates.
- No duplicate graph/scheduler mechanism is introduced.
