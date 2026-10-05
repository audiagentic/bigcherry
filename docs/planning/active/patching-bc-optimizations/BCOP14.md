---
id: BCOP14
order: 14
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T04:43:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Qualify upstream AMD quant-kernel register and byte-permute optimizations

## Description

Backfill follow-through for earlier optimization audits performed before the active-branch rule was corrected. Preserve the useful findings on the live branch without copying stale automation-branch plan edits. Relevant upstream candidates are llama.cpp #29910 Q2_K register-pressure/unroll reduction and #29927 native AMD `amdgcn_perm` q1_0 unpack.

## Steps

1. Check whether the current pin already contains/supersedes #29910/#29927.
2. If still relevant, qualify exact upstream changes on gfx1100/gfx1201 with generated ISA, VGPR/SGPR, spills, occupancy and correctness.
3. For Q2_K sweep ub16-512 and PP/TG lanes; do not equate zero spills with a win—retain architecture/shape crossover evidence.
4. For q1_0 test both MMVQ/decode and MMQ/prefill sites and verify byte-selector semantics before performance claims.
5. Reuse HIP-autotune/PKC ownership for dispatch; do not create another architecture table.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- Both upstream candidates are classified as adopted, rejected, already-upstream or pending with AMD measurements.
- Correctness plus ISA/resource evidence accompanies performance data.
- No duplicate dispatch registry is introduced.

## Notes

2026-10-05 check (step 1): both #29910 (Q2_K VGPR spills) and #29927 (amdgcn_perm q1_0 unpack) are still OPEN upstream and not in pin 050439614. The pin already uses __builtin_amdgcn_perm for a different unpack in vecdotq.cuh and mmq-load-tiles.cuh, so the intrinsic is proven to build on our toolchain. Item stays pending: no other plan item owns this (PKC05 is the 27B Q8_0 headroom item, not a fit). Remaining work is the hardware qualification in steps 2-4.

## Related

PKC05/1273, HIP autotune; llama.cpp #29910, #29927.

## Change Log

- 2026-10-05T04:56:45.106215+00:00 (updated-by): Updated: section:notes
