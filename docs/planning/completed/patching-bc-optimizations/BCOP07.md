---
id: BCOP07
order: 7
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-05T04:36:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Complete QSA chunk correctness and constant-topology promotion

## Description

Follow-through for QFP22/1332 audits. Patch 1007 has acted on the stale Meta subgraph-pointer/reallocation failure, so that portion is already implemented. Remaining work is output identity, constant graph topology and final promotion of QSA-only chunking.

## Steps

1. Treat patch 1007 as completed prerequisite; do not duplicate its Meta subgraph realloc fix.
2. Run `GGML_SCHED_DEBUG_REALLOC=1` and graph node/leaf/reserve tracing for unchunked, chunk256, tail chunk and first TG decode; require no unexpected post-reserve reallocation.
3. Resolve dense-vs-chunk deterministic output/logit divergence before performance promotion.
4. Keep graph topology fixed by configured chunk size; make short tail rows inert rather than creating runtime-dependent topology.
5. If view bookkeeping remains unsafe after 1007, A/B corrected views against fixed-shape GET_ROWS materialization and reject GET_ROWS if it costs >3% prefill wall.
6. Run chunk 128/256/512, ub512/1024, 1K/24K/80K+, MTP/no-MTP, sequential-request and tail-chunk correctness matrix.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- No unexpected scheduler reallocations after reserve.
- Dense/chunk correctness is explained and passes the chosen identity gate.
- ub1024 long-context memory saving and >=5% PP gain survive correctness qualification.
- <=2% TG regression.
- QFP22/1332 ledger is updated to promote or reject the patch.

## Notes

2026-10-05 review: superseded by owner item QFP22, which already carries the realloc-debug, #29958 fixed-topology and GET_ROWS steps. Done so far: patch 1007 validated and in patch-set.upstream-fixes; 1332 reworked to a contiguous per-chunk mask (build b-chunk9 running). Still open under QFP22: dense-vs-chunk text identity, the chunk/ubatch/depth matrix, promote or reject.

## Related

QFP17, QFP22, patch 1332, patch 1007; llama.cpp #29958.

## Change Log

- 2026-10-05T04:37:02.864950+00:00 (updated-by): Updated: section:notes
- 2026-10-05T04:37:48.421081+00:00 (state-transition): State: pending → superseded
