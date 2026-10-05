---
id: BCOP06
order: 6
plan: bc-optimizations
state: pending
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

## Related

QFP15, QFP13, FMTP01, patch 1294.

## Acceptance Criteria

- Device-placement and pure-upstream controls are both completed.
- Root ownership is narrowed or QFP15 is explicitly converted to benchmark-hygiene guidance.
- No Meta-unsafe tracing path is reintroduced.
