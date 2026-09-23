---
id: PRBE110
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-23T09:19:09.142442+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Extract RD07 (and a scoped RD05) from rejected 1203 as new untested patches

## Description

1203_rd050607_rdna4_wmma_fa_q6k_mmq was rejected 2026-09-23 (PA39) because its RD06 contract measured no gain; RD05 and RD07 passed inside that bundle. Per GPT lifecycle review req_f34f50a25c6240fe, the passing slices may only return as NEW, separately identified untested patches with fresh hardware evidence -- the 1203 receipt binds 1203's composition and is not reusable as promotion evidence.

## Steps

1. Author a new package carrying only the rd07-* edits of 1203 (mmq-vec-dot.cuh fold/hoist/sum-line, mmq.cu activation marker, J_MAX env, test-backend-ops perf cases), bound to RD07-Q6K-MMQ-PREFILL-FOLD, state=untested, with a patch-local producer derived from 1203's RD07 checks.
2. Separately assess RD05: rd05-kbc-sync affects the generic WMMA kernel (<=128 heads reach np==1); rd05-k00-sync only matters for larger-head configs. Split the combined RD05/RD06 activation marker. RD06-owned edits (rd0506-config-table, rd0506-softcap-read, rd06-dkq-gate, rd06-wmma-gating) stay out.
3. Fresh real-hardware qualification for each new identity; promote only on its own evidence.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Follow-on to PA39's terminal rejection of 1203. PA41's RD06 remediation deferral is unchanged.

## Change Log

- 2026-09-23T09:19:09.142442+00:00 (created-by): Created by agent
