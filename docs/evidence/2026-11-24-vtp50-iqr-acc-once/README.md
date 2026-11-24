# VTP50: the IQR 16-row tile keeps its integer WMMA sums native per scale group (gfx1100)

The 16-row tile of the IQR weight family (`iqr_m16`, which serves the drafter's 4-bit rows and every computed-trellis trunk row from 2 to 64 tokens) ran each 16-K step of a 128-K scale group through the WMMA compatibility shim: operand exchanges, a zero-accumulator matrix instruction, a layout conversion and an add. The rewrite keeps the group's int32 sum in the gfx11 accumulator layout across the eight steps and converts once before the float fold (the pattern of the int8 K-parallel tile). The row-f32 forms seed the chain with one uniform integer constant, which is layout invariant. The rewrite applies where the resource record is no worse: computed-trellis forms at one or two token tiles, not the chained 3-bit row-f32 NT1, not the gated unchained B6 NT1; all other forms and every NT is native.

## Resource record (full libr3, 7952 kernel symbols, shipped against this change)

7643 unchanged, 304 changed (the native `iqr_m16` specialisations), 5 unrelated IQR prefill records differ (236 -> 240 VGPR, 2-3 spill registers; present in a control build without this change, waived by the lead). All 404 `iqr_m16` specialisations checked: no new spill, no lost wave. Native 304 / stock 100: other forms 40/40 native, kind 5 (table trellis) 48/48, kind 6 (computed trellis) 216 of 316 native (token tiles 1: 172 of 184, 2: 44 of 44, 3 and 4: 0 of 88, which keep the shim form; plus four chained B3 row-f32 NT1 and eight gated unchained B6 NT1).

## Byte identity (candidate and shipped plugin in one process, same inputs, full BF16 output bytes)

- Five forms (table4, zero6, grid8, trellis4, mad4) x six projection shapes and a ragged one x M 1-64 x chained/unchained: 4480 comparisons, 0 differing. Trellis-MAD row-f32 B2/B4/B6, gated and plain: 480 cases identical.
- The real IQ6R trunk encodings: computed trellis B6 and B7 x six trunk shapes and N=5376 (a ragged 256-row run) x M 1-64 x chained/unchained: 1792 comparisons, 0 differing.
- `r3_selftest` cases `gemm_iqr_nt_m64`, `gemm_iqr_tiled`, `gemm_iqr_gated`, `gemm_iqr`: all pass, 0 FAIL.

## Cold replay on the real trellis encodings (one XTX, 256 MiB flush, median of 15, us)

| shape | M | shipped | new |
|---|---|---|---|
| 13312x5120 | 1 / 4 / 8 / 16 | 80 / 162 / 150 / 164 | 81 / 117 / 126 / 123 |
| 5120x6656 | 1 / 4 / 8 / 16 | 49 / 95 / 96 / 100 | 46 / 71 / 80 / 80 |
| 6144x5120 | 1 / 4 / 8 / 16 | 44 / 83 / 82 / 87 | 43 / 61 / 63 / 64 |
| 13312x5120 | 17-32 (two token tiles) | 217-229 | 146-158 |
| 5120x6656 | 17-32 | 130-137 | 92-95 |

Rows 4-16 and 17-32 are 20-32% faster; M=1 (served by the GEMV, not this tile) is unchanged; rows 33-64 (three and four token tiles) keep the shim form and are unchanged.

## The five small-row cells flagged by the first replay

The first 616-comparison replay (three repeats) showed five cells 0.6-3.6% slower outside their spread, all at M 4-8. Rerun with nine repeats each (cold and prefetch-warm, chained and unchained, 24 cells over the three shapes): every cell is within +-1.2% and inside its spread (for example 5120x1536 M8 cold 19.6 vs 19.6 us, 6144x5120 M8 warm 39.8 vs 39.9 us, 5120x6656 M4 chained cold 37.4 vs 37.5 us). Computed-trellis rows at M=1 go to the GEMV and rows at M>=2 to this tile, so the earlier cells were measurement spread, not a regression of the edited tile.

## Serving (qwen27b-tp3, IQR container, three ranks, 3 warm-ups, seeds 1-3 x 3 repetitions, ABBA, four launches a side; head-tile engine against head-tile engine plus this change)

| streams | decode ms/step before (arm sd) | after | difference |
|---|---|---|---|
| 1 | 25.77 (0.06) | 22.36 (0.04) | -3.41 ms (-13.2%) |
| 6 | 38.39 (1.48) | 31.12 (0.40) | -7.27 ms (-18.9%) |
| 8 | 38.96 (0.19) | 33.49 (0.07) | -5.47 ms (-14.0%) |

Tokens/s 99 / 264 / 306 -> 112 / 298 / 331; first token 512 tokens 244 -> 244 ms, 2048 tokens 778 -> 745 ms; temp-0 canary texts identical in all eight arms. The int8 container does not use this tile (control not repeated). Against the int8 container on the same layout the IQR container is now +2.1 ms at 1 stream and 6 streams and +1.7 ms at 8 streams.
