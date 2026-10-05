---
id: BCOP06
order: 6
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-05T04:35:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Isolate MTP draft nondeterminism before accepting small throughput wins

## Description

Follow-through for QFP15. Existing evidence has not proven pure-upstream nondeterminism and the prior per-node eval-callback technique violated Meta/tensor-split graph invariants. The remaining discriminator is draft-device placement plus a genuinely unpatched upstream control.

## Steps

1. Run the existing draft trace with target fixed and MTP draft moved from gfx1030 to gfx1100; compare hidden hashes, top-id/probability and acceptance at warm-up/24K/80K across >=3 runs.
2. Build/run a genuinely unpatched current-upstream control; verify no BigCherry base overlay is present.
3. If divergence follows gfx1030, isolate QSA/indexer/TOP_K/reduction ordering there. If it persists in pure upstream and on gfx1100, document benchmark noise/state behavior rather than inventing a deterministic kernel path.
4. Do not revive per-node graph splitting. Observe materialized outputs only at existing safe compute boundaries.
5. Until resolved, judge <=3% speculative speedups using ms/step plus ABBA/multi-request evidence, not one t/s run.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- Device-placement and pure-upstream controls are both completed.
- Root ownership is narrowed or QFP15 is explicitly converted to benchmark-hygiene guidance.
- No Meta-unsafe tracing path is reintroduced.

## Notes

2026-10-05 review: superseded by owner item QFP15, which already carries the gfx1100 draft-placement control, the unpatched-upstream control, the ban on per-node eval callbacks and the ABBA rule. Neither control has been run; work stays open under QFP15.

## Related

QFP15, QFP13, FMTP01, patch 1294.

## Change Log

- 2026-10-05T04:36:59.486647+00:00 (updated-by): Updated: section:notes
- 2026-10-05T04:37:45.134059+00:00 (state-transition): State: pending → superseded
