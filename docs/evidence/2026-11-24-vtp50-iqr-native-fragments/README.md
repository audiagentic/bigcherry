# VTP50: native token and weight fragments for the IQR 16-row tile (gfx1100)

Builds on the accumulate-once change of the same tile. Where the form/token-tile/chain record qualifies (the same guard), the tile now loads the token rows as their 16 K bytes in natural order (one 16-byte load, the same address in lane and lane + 16), reads the decoded weight rows back from the wave's LDS tile in the same order (two 8-byte reads a step) and feeds the gfx11 matrix instruction directly: no operand exchange is left in the loop. The next-unit token ring is dropped on the native form (its entries would double) and kept on the compatibility form, which loads its eight bytes as before.

## Resource record (full libr3, 7791 kernel symbols, previous change against this)

324 symbols change, all `iqr_m16`; no other kernel changes. No new spill anywhere. Waves per SIMD: 182 gain a wave, 104 are unchanged, 38 lose one (for example the chained B7 row-f32 one-tile forms 150 -> 153/155 VGPR and the unchained two-tile B4-B7 forms 151-164 -> 169-177 VGPR). A symbol that loses a wave is kept native because the replay below is faster for it (the real trunk form, B6 under a bf16 scale, two tiles, unchained, is 6-7% faster at 24 and 32 rows).

Instruction census of the B6 one-tile unchained kernel (per lane per 128-K unit, 64 weights): 1056 -> 900 instructions; operand selects and moves 187 -> 87; `ds_bpermute` 36 -> 4; the decode (window extract, mask, multiply-add, byte-sum, packing) is unchanged at about 7.6 vector instructions a weight.

## Byte identity (new and previous plugins in one process, same inputs, full BF16 output bytes, M 1-64, chained and unchained, 7 shapes including the ragged 5376)

Seven forms: table4, zero6, grid8, trellis4 (12-bit state), computed trellis B4, B6, B7: 7 x 2 x 7 x 64 = 6272 comparisons, 0 differing. `r3_selftest` (built from main) cases `gemm_iqr_nt_m64` (1034 ok), `gemm_iqr_tiled` (129), `gemm_iqr_gated` (36) and `gemm_iqr` (1199): 0 FAIL on the 2020.8.7 plugin and on this plugin. (An older selftest binary, built before the two-slice route for the 4-bit table rows, still demanded byte identity to the int8 row for those FP64-gated rows and reported two one-bf16-step differences; that comparison is obsolete, not a defect of this change.)

## Cold replay on the real trellis encodings (one XTX, 256 MiB flush, median of 15; ratio native / previous, B6 and B7, six trunk shapes)

| rows | mean | best | worst |
|---|---|---|---|
| 4 | 0.96 | 0.88 | 1.05 |
| 8 | 0.90 | 0.76 | 0.99 |
| 16 | 0.93 | 0.85 | 1.03 |
| 24 | 0.93 | 0.87 | 1.01 |
| 32 | 0.94 | 0.85 | 1.03 |

## Serving (qwen27b-tp3, IQR container, three ranks, 3 warm-ups, seeds 1-3 x 3 repetitions, ABBA, four launches a side)

| streams | decode ms/step previous (sd) | native | difference |
|---|---|---|---|
| 1 | 22.41 (0.02) | 21.51 (0.07) | -0.90 ms (-4.0%) |
| 6 | 30.89 (0.47) | 29.96 (0.22) | -0.93 ms (-3.0%) |
| 8 | 33.41 (0.08) | 33.02 (0.19) | -0.39 ms (-1.2%) |

First token 512 tokens 243 -> 248 ms and 2048 tokens 766 -> 783 ms (inside the 236-250 and 721-787 ms spread of every other arm of the series; the prefill rows use other tiles); temp-0 canary texts identical in all eight arms.
