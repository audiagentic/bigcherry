---
id: BCOP22
order: 22
plan: patching-bc-optimizations
state: superseded
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

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- Dependency ownership is explicit and non-duplicated.
- Current-upstream comparison exists before port/promotion.
- Performance and correctness evidence determine promote/reject/retire status.

## Notes

2026-10-05: superseded by PRBE11, which already owns paired-MMVQ (patch 1205) qualification and states RD25/PRBE19 is not a prerequisite, so the dependency-chain concern is resolved. The current-upstream comparison matrix was added to PRBE11's notes. Qualification itself not run.

## Related

RD12, RD25, PRBE11 and fused-core prerequisite work.

## Change Log

- 2026-10-05T04:55:33.280600+00:00 (updated-by): Updated: section:notes
- 2026-10-05T04:56:02.419170+00:00 (state-transition): State: pending → superseded
