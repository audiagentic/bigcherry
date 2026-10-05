---
id: BCOP03
order: 3
plan: patching-bc-optimizations
state: superseded
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

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- One forced-front mechanism serves fresh and promoted fronts.
- >=70% of candidate draft/replay work is hidden or the overlap path is rejected.
- No duplicate live-front lease/transition API remains planned without measured necessity.
- End-to-end promotion meets FMTP03 performance/correctness gates.

## Notes

2026-10-05 review: superseded by owner item FMTP03, which already carries the 1321 forced-front + n_tail design, the 70% hidden-work gate and EWMA tail depth. Not implemented; work stays open under FMTP03.

## Related

FMTP02, FMTP03, patch 1321, QFP08.

## Change Log

- 2026-10-05T04:36:49.467652+00:00 (updated-by): Updated: section:notes
- 2026-10-05T04:37:35.081982+00:00 (state-transition): State: pending → superseded
