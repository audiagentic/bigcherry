---
id: BCOP01
order: 1
plan: bc-optimizations
state: superseded
created-at: '2026-10-05T04:30:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Act on QFP17 QSA prefill memory and tiled-indexer findings

## Description

Follow-through item for the optimization audit that identified dense QSA/attention masks as the dominant ub1024 compute-buffer growth and staged upstream tiled lightning-indexer qualification. QFP17 owns implementation/performance evidence; BCOP01 exists to ensure the audit conclusions are either implemented, measured, or explicitly rejected.

Current branch evidence already shows QSA chunking can make ub1024 fit at 240K, but correctness/promotion remains governed by QFP22/1332. This item must not duplicate that implementation.

## Steps

1. Confirm QFP17/QFP22 current state after 1332 and patch 1007 work; close any action already fully implemented.
2. Complete dense-mask attribution at ub512/1024/2048 and record peak per-rank compute-buffer savings from QSA-only chunking.
3. Resolve 1332 output-identity/correctness before treating its throughput as promotable.
4. Qualify llama.cpp PR #29901 tiled lightning indexer independently on gfx1100 and gfx1201: kernel time, PP, TG, LDS, VGPR/SGPR, spills and occupancy.
5. Adopt/port #29901 only if it meets QFP17's AMD promotion gate; do not duplicate its kernel locally if the pinned upstream already contains it.
6. Update QFP17/QFP22 and patch ledger with final promote/reject result.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- QSA chunking has a correctness-qualified promote/reject result, not throughput-only evidence.
- gfx1100/gfx1201 #29901 qualification is recorded or documented as already upstream/pinned.
- Peak-memory and PP/TG deltas are recorded at representative long-context lanes.
- No duplicate QSA/indexer implementation is introduced.
- QFP17/QFP22 and ledger agree on final ownership/status.

## Notes

2026-10-05 review: superseded by owner items QFP17 (carries #29901 qualification, VGPR/spill and ub512/1024/2048 gates) and QFP22 (1332 identity). Not implemented; the work stays open under those owners. #29901 is not in pin 050439614.

## Related

QFP17, QFP22, patch 1332, patch 1007; upstream llama.cpp #29901.

## Change Log

- 2026-10-05T04:36:42.727836+00:00 (updated-by): Updated: section:notes
- 2026-10-05T04:37:28.360312+00:00 (state-transition): State: pending → superseded
