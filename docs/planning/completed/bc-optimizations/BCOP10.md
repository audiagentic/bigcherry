---
id: BCOP10
order: 10
plan: bc-optimizations
state: superseded
created-at: '2026-10-05T04:39:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Qualify adaptive MTP draft vocabulary after static 64K trim

## Description

Follow-through for QFP05/1297. The static 64K LM-head trim already improved draft ms/step; determine whether frequency/context-adaptive fixed-bucket vocabularies add enough accepted-target-tokens/s to justify selection/gather complexity.

## Steps

1. Establish current 64K static control at short/24K/80K with LM-head share of speculative-step wall.
2. Test static frequency-ranked vocabularies and fixed-shape adaptive unions at 8K/16K/32K/48K/64K using prompt/recent-generated/recent-draft/top-k tokens.
3. Compare contiguous selected-row gather+GEMV with indexed/sparse row GEMV; include selection, gather and canonical-ID remap cost.
4. Stop adaptive work if static trimming drives LM head below 10% of speculative-step wall or selection/gather consumes >50% of saved LM-head time.
5. Keep generic TOP_K optimization outside QFP05 unless profiling makes it material.
6. Rank by accepted target tokens per wall-second with target correctness unchanged.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- Best static control and adaptive buckets have ABBA measurements at short and long context.
- Adaptive path promotes only for >=3% effective TG improvement at both representative depths with <=2% regression.
- Graph shape remains fixed/reserved; no runtime topology variants.
- Generic vocab-pruning/TOP_K APIs are not duplicated.

## Notes

2026-10-05 review: superseded by owner item QFP05, which already carries the frequency/adaptive bucket matrix, the 10% LM-head stop gate and the accepted-target-tokens ranking. Not implemented; work stays open under QFP05.

## Related

QFP05, patch 1297, FMTP policy; llama.cpp #25187 and #29883; FR-Spec/SpecVocab mechanism evidence.

## Change Log

- 2026-10-05T04:37:12.876277+00:00 (updated-by): Updated: section:notes
- 2026-10-05T04:37:58.375838+00:00 (state-transition): State: pending → superseded
