# Single-band fused IQR prefill — gfx1100 qualification

## Change

Decode each 16-row weight unit inside the 128-K loop, sharing its fragment-order bytes through LDS across the waves that consume it. The native int8 WMMA accumulation, scale folds and output stores remain unchanged. No global decoded plane is written by the fused kernel. Dispatch is restricted to one token band and the eleven qualified computed-trellis forms: L16 B2–B7 row-F32, L15 B2 row-F32, and L16 B4–B7 BF16 with LRB=0 (all LV=0, LRB=0). Other forms and multiple bands retain the legacy route. The scratch allocation contract is unchanged.

## Qualification of the fused candidate

The following results were supplied by the independent build and GPU qualification session for the enabled fused candidate. They precede the final cleanup commit `20e85e3af`; a rebuild and output re-hash of that commit are still required before release.

| Gate | Result |
| --- | --- |
| Resource records | 11 fused kernels added; 0 changed among 7791 existing records |
| Fused resources | 236–239 VGPR, zero spills, 6 waves, 36864 bytes LDS |
| Byte identity | 491 output hashes, 0 differences |
| Numerical oracle | 1473 checks passed |
| Cold replay | 60 launches per point, ABBA ordering; 56 IQR points |
| Replay outcome | 24 wins, all at M=65/96/128; 0 regressions |
| Single-band candidate/shipped ratio | 0.89 mean |
| Multi-band controls (M=192/256/512/2048) | 1.00 ratio |

Single-band candidate/shipped timing ratios (lower is better):

| N × K | M=65/96/128 ratio range |
| --- | ---: |
| 13312 × 5120 | 0.78–0.80 |
| 5120 × 6656 | 0.94–0.95 |
| 6144 × 5120 | 0.87–0.89 |
| 5120 × 2304 | 0.93–0.96 |

Relative to the int8 tiled comparator, the single-band IQR excess falls from +36% to +20%. The improvement is limited to the qualified single-band route; no long-chunk improvement is claimed.

## Rejected alternative

The separate spin-free producer retained the global decoded plane and introduced a second kernel launch. Its 64-point cold replay produced 15 wins, 20 regressions and 29 neutral points. It is removed rather than retained as dead code. The one-band fused route attacks the decode-and-publish cost without multiplying decode work across token bands.

## Final-source checks

The pinned-source native rewrite check passed with no failures after the cleanup. Enumeration of the pinned form list selects exactly 11 of 44 forms. No build or GPU run was performed in this final code-only change; the final-commit rebuild and byte-identity confirmation belong to the independent qualification session.
