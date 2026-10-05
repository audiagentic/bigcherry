---
id: BCOP03
order: 3
plan: bc-optimizations
state: pending
created-at: '2026-10-05T04:32:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Qualify forced-front MTP overlap using the existing 1321 primitive

## Description

Follow-through for the FMTP03 consolidation audit. Patch 1321 already provides forced-front plus bounded-tail drafting; FMTP03 should not grow a second live-front continuation/lifetime API unless measurements prove the existing primitive cannot hide useful draft work.

## Steps

1. Confirm current 1321 behavior and remove/retire stale FMTP02/FMTP03 text proposing duplicate live-front/lease machinery where still present.
2. Implement/qualify target-verification overlap using `forced=current_front` plus bounded `n_tail` on the drafter, retaining only newly generated tail tokens.
3. Measure target verification, forced replay, continuation and exposed overhang at short/24K/80K.
4. Add slack-adaptive tail-depth selection from prior-round timing EWMAs only after fixed-depth correctness is proven.
5. Require rejection-safe recurrent/KV/state ownership and unchanged target output/acceptance semantics.

## Related

FMTP02, FMTP03, patch 1321, QFP08.

## Acceptance Criteria

- One forced-front mechanism serves fresh and promoted fronts.
- >=70% of candidate draft/replay work is hidden or the overlap path is rejected.
- No duplicate live-front lease/transition API remains planned without measured necessity.
- End-to-end promotion meets FMTP03 performance/correctness gates.
