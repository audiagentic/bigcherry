---
id: QFP22
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-04T21:08:00.115435+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Cleanup + promotion validation: 1321/1322 MTP-ahead and 1332 QSA token chunks

## Description

Owner 2026-10-05: 1321, 1322 and 1332 look helpful; put them on the cleanup and promotion-validation list. Fixes go inside each patch (owner rule). Promotion uses the lightweight tier: adoption ABBA vs the promoted base (v6), activation evidence, offline tests, greedy check (vs f32 reference where a near-tie is involved, see QFP17).

## Steps

1321 + 1322 (MTP-ahead, promote together; 1322 requires 1321):
1. Cleanup: drop the tail_p_min / BIGCHERRY_MTP_AHEAD_PMIN knob (only made things worse) unless a variant needs it; keep default off.
2. Rework candidate before validation: promote only full-length fronts (tail shorter than a fresh draft -> fresh draft), per FMTP03 notes.
3. Validation: multi-request ABBA (8 arms/depth) at ~8K and ~64K on v6, BIGCHERRY_MTP_AHEAD=0/1; promotion stats line as activation; greedy identity. Current single-ABA evidence: per-step -7..9%, t/s +1-2% (within noise) - need the ABBA to show a real win.

1332 (QSA token chunks):
4. Fix inside 1332: no-MTP single-token decode after a chunked prefill segfaults in ggml_backend_meta_graph_compute (stale meta bookkeeping for views of the kq_mask input). Replace input views with ggml_get_rows of the chunk's kq_mask rows from a small per-chunk I32 row-index graph input (set_input hook), or fix the meta view bookkeeping.
5. Validation: no-MTP decode + MTP serving at 24K/80K; ub1024+chunk256 vs v6 ub512 prefill/decode ABBA at 80K fill and a short-prompt prefill (where larger ub should help most); greedy checked against the f32 reference at near-ties (QFP17).
6. Decide the production setting (ub1024 + BIGCHERRY_QSA_CHUNK=256 in a feature set) only if the ABBA shows a win.

7. Add winners to the flashnext feature set (new row in 0910, e.g. flashnext-v7) and promote via validated-enhancements.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per patch: patch-lint + offline tests; adoption ABBA vs v6 with complete separation or a stated enabler rationale; activation evidence; greedy identity (or f32-reference agreement at near-ties); 1332 additionally no-MTP decode and MTP serving without crashes.

## Effort & Risk



## Standards



## Acceptance Criteria

- 1321/1322 promoted or explicitly parked with ABBA evidence.
- 1332 crash fixed inside 1332 and promoted or parked with ABBA evidence.
- Winners enabled through a new BIGCHERRY_FEATURES set.

## Notes

## Change Log

- 2026-10-04T21:08:00.115435+00:00 (created-by): Created by agent
