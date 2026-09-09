---
id: NRO08
order: 8
plan: nasone-rdna-optimizations
state: superseded
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Wave32-native TOP_K reductions and item-per-thread tuning

## Description

Apply nasone follow-up commit `7f1d25f7e05cb34053d317a0f72f601d846c8662` (`ROCm: make TOP_K wave32-native`) on top of NRO07. The follow-up replaces shared-memory tree reductions in TOP-1/multi-pass helpers with wave32 shuffle reductions where safe, splits 64-bin radix accumulation into two 32-lane scans, and increases per-thread item ownership to reduce intermediate passes/launches.

This must not be folded into PNRO06's first qualification because then a win or regression could not be attributed to the hybrid algorithm versus the wave32 mapping.

## Steps

1. Require `1256_nro07_topk_hybrid` and verify source pre-image matches the PNRO06 post-image expected by the follow-up.
2. Port wave32 shuffle reduction with explicit `USE_SHUFFLE`/fallback structure; preserve a shared-memory path for shapes/builds where wave32 assumptions are not proven.
3. Make 32-lane width explicit in shuffles/ballots; do not rely on HIP `warpSize` implicitly if the algorithm is wave32-specific.
4. Port two-half 64-bin scan logic and validate bucket selection at every boundary.
5. Port `ITEMS_PER_THREAD` tuning separately from semantic changes where possible; record launch/pass-count effect.
6. Run exact correctness fixtures from PNRO06 unchanged. Any output difference versus NRO07 is a failure unless traced to a pre-registered tie-order policy.
7. Add stress tests for ncols around block_size*items_per_thread boundaries.
8. Profile LDS reduction traffic, wave occupancy, VGPR use, pass count, and kernel duration.
9. Compare PNRO06-only versus NRO07+NRO08 on identical real TOP_K signatures.
10. If the wave32 follow-up wins only on a subset, retain selector/fallback rather than replacing PNRO06 globally.

## Detailed Solution & Technical Design

For TOP-1, each wave reduces candidate `(index,value)` pairs through `__shfl_down`, writes one candidate per wave to shared memory, then wave 0 reduces those wave winners. This reduces shared-memory footprint from one candidate per thread to one per wave and removes block-wide barriers from every reduction stride.

For 64 radix bins on wave32, one lane cannot scan all 64 with width-64 shuffles. The follow-up computes high and low 32-bin prefix scans separately, offsets the low half by the high-half total, and chooses the first half that satisfies the rank limit. Tests must target limits immediately around the high/low boundary.

Increasing items/thread changes block coverage and intermediate `ncols_output`; verify integer rounding and last-pass detection for very small and non-multiple sizes.

## Code Samples & Guidance

Keep the fallback reduction implementation in code until validation proves all targeted gfx1100 configurations use wave32 safely. Treat wave size as an architecture execution property, not a universal CUDA assumption.

## Files

- `docs/planning/active/nasone-rdna-optimizations/PNRO07.md`
- `patches/1257_nro08_topk_wave32/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}`
- shared NRO static tests; reuse/extend PNRO06 TOP_K reference suite.

## Validation

Exact PNRO06 correctness suite plus targeted 31/32/33 and 63/64/65 bin/column boundaries, high/low radix-half threshold selection, block/item coverage boundaries, ties and NaNs.

Performance: PNRO06 as control, NRO08 subject, same binary/config where possible. Capture LDS instructions/occupancy if profiler supports them.

## Effort & Risk

Medium-high. Smaller than PNRO06 but architecture-sensitive; subtle shuffle-width or lane-mask errors produce rare routing errors at boundary ranks.

## Standards

Dependency-aware source composition; exact routing semantics; keep PNRO06 fallback; no unsupported wave-size extrapolation.

## Acceptance Criteria

- Exact outputs match PNRO06/reference across full fixture matrix.
- Wave32 path activates only where supported.
- Measurable reduction in kernel time/LDS traffic or launch/pass count on real gfx1100 signatures.
- No >1% non-target model regression.

## Notes

PNRO07 is the only plan item that should use source commit `7f1d25f7...`; NRO07 owns the preceding hybrid implementation commit.

Superseded by: PNRO07
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created from nasone wave32 TOP_K follow-up; P0.

## Ledger-events


- Pending: ag-ledger MCP unavailable in authoring session.
- 2026-09-09T11:24:56.791286+00:00 (updated-by): Updated: section:notes
- 2026-09-09T11:43:04.659906+00:00 (state-transition): State: pending → superseded
- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:57:59.867446+00:00 (updated-by): Updated: section:ledger-events
