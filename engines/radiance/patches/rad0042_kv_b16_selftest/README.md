# rad0042_kv_b16_selftest

## Summary

VTP44 patch B2. rad0041 added `bf16_fp8` and `bf16_i8` rows at Flash-Next's geometry; this patch adds the device cases that
check them, by extending the cases that already check the fp8_v4 and k6_v4 rows there instead of writing parallel ones. Every case
compares a kernel against an independent host oracle (or, for the fused row, against the unfused pair byte for byte).

| case | shapes |
|---|---|
| numeric, gqa 12, block 4 | prefill 64 and 100 rows (ragged last tile), decode 1 to 5 rows (1 to 4 waves) |
| numeric, gqa 6, block 4 | prefill 64 and 100 rows, decode 1, 4, 8 and 10 rows |
| split-KV prefill, mixed step | `kSplitG12` and `kSplitG6Bs4` rows for both layouts (the existing production-shaped tails and verify blocks) |
| fused gate+quantiser | `attn_decode_gq_h256_gqa12_b16f8` / `_b16i8` byte-identical to the decode plus `gate_quant_fp8` pair at 3 and 24 rows |

The fused case's cache is bf16 K (finite), then V bytes (E4M3 without NaN codes, or int8 in -127..127) and, for `bf16_i8`, an f16 scale per
32 dims at 3 * head_dim; row sizes come from the ABI's `RAD_KVQ_*_ROW16`, not literals.

## Outcome

Untested. The cases compile into `r4d_selftest` (gfx1201 build, no device) and libr3's `r3_rewrite.py --check` passes. The device run is queued on
the `gpu` lease against the rad0038 baseline (0 failures, 246 skipped); the new cases have no baseline, so their first run is the evidence.

## Detail

`libr4d/r4d_selftest.cpp` only.

### Promotion record

Tier: diagnostic

Mechanics: card-free patch-package tests (apply, anchors, idempotence) and patch-lint; the selftest cases compile in a gfx1201 build and pass libr3's rewrite check.
Activation: the cases run on a gfx1100 card (one 7900 XTX, libr3's r3_selftest, each case alone) on 2026-11-18: all 21 cases of the two packed layouts (bf16_fp8 and bf16_i8, gqa 12 and gqa 6) pass, including bf16_i8_rows; the run before the inventory fix failed 14 of them, and the run after passed all. See docs/evidence/2026-11-16-vtp44-generic-kv-layouts/rad0042-selftest/ (before-summary.txt, after-summary.txt).
