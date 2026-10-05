---
id: BCOP05
order: 5
plan: bc-optimizations
state: pending
created-at: '2026-10-05T04:34:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Act on QFP09 critical-path rank-skew findings

## Description

Follow-through for QFP09 audits that showed the late rank changes with context and that raw GPU work is not equivalent to movable critical-path work. QFP09 owns cross-rank placement/arrival-skew diagnosis; QFP13 owns local launch-gap reduction.

## Steps

1. Use timestamp-paired AllReduce intervals to derive per-rank busy, idle and excess critical-path time between collective arrivals.
2. Measure short/80K/160K+ lanes after current QFP13 reductions.
3. Attempt placement changes only where >=0.5 ms/token of movable excess is identified.
4. Attribute attention-specific skew to QFP07 and local same-stream gaps to QFP13 rather than duplicating mechanisms.
5. Promote placement only for >=3% decode improvement with <=2% regression elsewhere and no context-headroom loss.

## Related

QFP07, QFP09, QFP13, QFP01/1291.

## Acceptance Criteria

- Critical-path excess is measured by rank/context rather than inferred from utilization.
- Each proposed placement has a quantified movable opportunity.
- Ownership is consolidated and stale generic launch-gap work is removed from QFP09.
