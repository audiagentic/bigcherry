---
id: PRBE108
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-12T22:08:29.479059+00:00'
breadth: ''
skill: ''
created-by: agent
---

# Reconcile RD39-42's formal +2.38% gain vs a quick non-interleaved -5% check

## Description

RD39-42/1215's formal contract-qualification evidence (10 rounds, deliberately interleaved baseline/subject to rule out ordering/drift bias) shows a real +2.38% mean decode gain (95% CI [1.45%, 3.32%]) on gfx1100 with GGML_CUDA_GRAPH_OPT=1. A quick, non-interleaved B-vs-C check run 2026-09-13 (always baseline first, then subject, 4 rounds, tg32/Qwen3.6-35B-A3B) instead showed a consistent ~5% NEGATIVE delta (B~114 t/s, C~108 t/s). This is a real, unresolved discrepancy -- not yet reconciled. The quick check's lack of interleaving is the most likely explanation (systematic drift/thermal/warm-up effects that interleaving specifically exists to rule out), but this has not been confirmed.

## Steps

1. Rerun the B-vs-C comparison with proper interleaving (baseline/subject/baseline/subject...) matching the formal methodology's own protocol.
2. If interleaved results match the formal +2.38% gain: the quick check's ordering bias is confirmed as the explanation, close this item.
3. If interleaved results still show a negative/different delta: something has genuinely changed since the formal evidence was gathered (build drift, pin drift, environment change) -- investigate further before trusting either result.
4. Check whether the current pin (b10901/28ff0958291c) matches exactly what the formal 10-round evidence was measured against, or whether the pin has moved since.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-12T22:08:29.479059+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260912_220855_found-a-real-unresolved-discr_5936
- 2026-09-12T22:08:55.337667+00:00 (updated-by): Updated: section:ledger-events
