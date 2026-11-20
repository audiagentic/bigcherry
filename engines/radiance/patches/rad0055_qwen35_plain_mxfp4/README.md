# rad0055_qwen35_plain_mxfp4

**Status:** validated (host-only source checks; hardware qualification pending)
**Plan item:** RUM06

## Failure and comparison

In the Radiance 2.2 Qwen3.8-27B TP2 smoke, both ranks refused the fused delta-net `ssm_inz` weight:
`arch/common/rad_block_gdn_fp8.h:266` calls `LinearFP8::declare`, and
`arch/common/rad_fp8.h:1354-1355` calls `linear_retired`, whose
`linear_is_mxfp4(e)` branch at line 1179 returns `"MXFP4"`. The resulting
`linear_refuse_retired` returns `RAD_E_UNSUPPORTED` at line 1194.
It happens before the q|k|v|z 4-part row split (or any gfx1100 kernel dispatch).
The wht6 TP wire is therefore unrelated to this refusal.

v1.17.0 has no `linear_retired` check. The same weight was admitted through
`LinearFP8::declare`'s OCP-MXFP4 `mxfp4a8` branch (v1.17
`arch/common/rad_fp8.h:1506-...`). The branch is still present in 2.2,
and the `DenseFP8` drafter already exempts plain MXFP4 at its
`if (known && !mx)` check.

## Fix

Remove **only** plain OCP-MXFP4 from the 2.2 retirement classifier; all other
format retirements remain intact (M2XFP4, rotated variants, block-scaled FP8).
The 2.2 `LinearFP8` OCP-MXFP4 path, `RowParts::set(...,16,...)`,
`RAD_SHARD_ROW`, and the rad0019 `span_parts` ownership machinery remain
unchanged. The fix is format-based, not model/head-count/rank-specific, and
does not add a fallback or a second kernel path.

Both `qwen27b-tp3` and `qwen27b-tp3-capacity` use the same MXFP4
checkpoint as `qwen27b-2xtx` and therefore share this refusal.
The `qwen27b-tp3-int8` family (including -556, -long, -bf16 and
-bf16i8 variants) also passes through `GdnFP8::declare`, but uses
the distinct retained w8a8 weight branch, not the MXFP4 retirement.

## Verification scope

The RUM06 patch test applies this patch to the *already composed*, frozen
2.2 source, reapplies it byte-identically, verifies MXFP4 admission while
retaining every other retirement, and detects a drifted anchor. The libr3 native
rewrite check uses the frozen 2.2 `libr4d` source. These are host-only
compatibility tests, not claims of runtime activation, KLD, or TP3 parity.
