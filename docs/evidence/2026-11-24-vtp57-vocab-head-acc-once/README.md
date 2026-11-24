# VTP57: the vocabulary-head tile keeps its integer sums native (gfx1100)

The head of the IQR container stores the vocabulary as width classes; its gfx1100 tile ran two integer WMMA passes per 128-K block through the compatibility shim and spilled at every token-tile count. The rewrite keeps the block's integer sums (high plane, the x128 shift, low plane) in the gfx11 accumulator layout and converts once before the float fold, and converts the weight fragment once per K step instead of once per token tile. Integer sums are exact, so the logits are bit-identical by construction; the check below confirms it.

## Resource record (full libr3, 7791 kernels, 2020.8.7 against this change)

Exactly four records changed, all `vocab_tile_kernel`; every other kernel is identical.

| token tiles (rows) | VGPR before / after | private bytes before / after |
|---|---|---|
| 1 (1-16) | 240 / 167 | 1280 / 0 |
| 2 (17-32) | 240 / 205 | 5576 / 0 |
| 3 (33-48) | 240 / 240 | 8196 / 52 |
| 4 (49-64) | 256 / 256 | 11012 / 204 |

## Bit identity

Two plugins in one process (the shipped 2020.8.7 libr3 and the new one), same inputs, full f32 logits compared bytewise at every row count 1-64, five configurations: the 27B head as the first and second rank hold it (class rows 59648 / 12800 / 16128 / 22016, three quarters real, the rest pad rows with id -1, rank-offset row map), two classes on the second rank, one padded class, a ragged three-class K of 2560. 320 of 320 pass, no unwritten logit. `r3_selftest` case `logits_gemm_vocab_iqr` on the new plugin: 67 ok, 0 fail. Serving level: KLD of the IQR container on three ranks (exact wire, speculation 0, BF16 corpus 4) gives the same per-position rows byte for byte before and after (mean 0.0019061782760480181), and the temp-0 canary texts are identical.

## Cold replay (one XTX, 256 MiB flush before each launch, median of 8, head slice of one rank)

| rows | before | after |
|---|---|---|
| 1 / 4 / 8 / 16 | 4717 / 4626 / 4689 / 4784 us | 867 / 871 / 894 / 898 us |
| 17 / 20 / 24 / 32 | 25.3 / 24.9 / 25.1 / 25.2 ms | 1.24 / 1.23 / 1.23 / 1.26 ms |
| 33 / 48 | 78.0 / 78.3 ms | 1.65 / 1.63 ms |
| 64 | 100.6 ms | 2.83 ms |

## Serving A/B (qwen27b-tp3, IQR container, three ranks, 3 warm-up requests, seeds 1-3 x 3 repetitions, ABBA, one launch per arm)

Four launches per side. Old = the previous swap-root engine, new = the same engine with the new libr3.

| streams | decode ms/step old (arm sd) | new | difference | tokens/s old / new |
|---|---|---|---|---|
| 1 | 30.22 (0.02) | 25.81 (0.12) | -4.40 ms (-14.6%) | 86 / 99 |
| 6 | 56.73 (1.21) | 37.46 (0.19) | -19.27 ms (-34.0%) | 195 / 262 |
| 8 | 58.31 (1.04) | 39.13 (0.34) | -19.18 ms (-32.9%) | 233 / 305 |

First token (one stream, max of 5): 512 tokens 254 -> 248 ms, 2048 tokens 786 -> 777 ms.

Control, the int8 container (plain head, not served by this kernel), two launches per side: 20.25 -> 20.26 ms (1 stream), 29.14 -> 28.77 (6), 31.96 -> 31.96 (8): no change beyond the spread.

## Left over

The 33-64 row tiles keep a small spill (52 and 204 bytes). Rows above 32 are not a decode shape at eight streams with depth 3; tracked under VTP57.
