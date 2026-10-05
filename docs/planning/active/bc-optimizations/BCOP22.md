---
id: BCOP22
order: 22
plan: bc-optimizations
state: pending
created-at: '2026-10-05T04:51:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Qualify paired-MMVQ through its fused-core dependency chain

## Description

Backfill of the first optimization audit. RD12 paired Q6_K K/V projection had multi-architecture correctness evidence but lacked performance qualification and depends on RD25 plus an underlying fused-core prerequisite. Treat the chain as one qualification programme rather than isolated ports.

## Steps

1. Recheck RD12/RD25/current patch state and close any dependency already implemented upstream or in BigCherry.
2. Make the fused-core prerequisite -> RD25 -> RD12 dependency explicit in current planning if still valid.
3. Compare paired-MMVQ against current-upstream HIP on the same gfx1100/gfx1201 decode/prefill matrix.
4. Record pp512/pp2048, tg128/tg512, activation rate, kernel time, VRAM and correctness; retain gfx1030 as a correctness/control architecture where applicable.
5. Do not create a second paired-MMVQ plan if PRBE11 or another current owner already covers qualification.

## Related

RD12, RD25, PRBE11 and fused-core prerequisite work.

## Acceptance Criteria

- Dependency ownership is explicit and non-duplicated.
- Current-upstream comparison exists before port/promotion.
- Performance and correctness evidence determine promote/reject/retire status.
