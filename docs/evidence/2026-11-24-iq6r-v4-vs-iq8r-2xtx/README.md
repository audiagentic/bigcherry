# IQ6R-V4 (trellis 6/7-bit) against IQ8R on two XTX cards (qwen27b-2xtx, 2020.8.7)

Containers: IQ8R + DFlash2 (28.76 GiB) and the 27B IQR-V4 conversion with the DFlash2 drafter (22.54 GiB, 1109 weights, 301-line recipe). The three-card layout refuses the IQR container (vocabulary head classes of 33280 rows are not a multiple of 256 x 3 ranks), so everything below is the two-XTX layout; the wire is `wht6` as shipped for speed and `exact` for quality, stated per table.

## Quality (BF16 reference, corpus 4, 8240 positions, speculation depth 0)

| wire | container | mean KLD | median | p99 | top-1 | ppl ratio |
|---|---|---|---|---|---|---|
| exact | IQ8R | 0.001535 | - | 0.0127 | 97.34% | 1.0011 |
| exact | IQR-V4 | 0.001979 | - | 0.0166 | 96.99% | 1.0016 |
| wht6 | IQ8R | 0.002364 | 0.00090 | 0.0200 | 96.63% | 1.0004 |
| wht6 | IQR-V4 | 0.002630 | 0.00099 | 0.0221 | 96.40% | 1.0006 |

## Speed (12 launches, ABBA x3, 3 warm-up requests per server, seeds 1-3 x 3 repetitions, 512-token prompts, 192 new tokens, wht6)

| streams | IQ8R ms/step (arm sd) | IQR-V4 | difference | aggregate tokens/s |
|---|---|---|---|---|
| 1 | 24.39 (0.04) | 34.60 (0.02) | +10.2 ms | 117 vs 75 |
| 4 | 28.04 (0.08) | 38.50 (0.05) | +10.5 ms | 276 vs 205 |
| 6 | 30.92 (0.12) | 61.41 (1.35) | +30.5 ms | 287 vs 176 |

First token (max over 5 repetitions, one stream): 512 tokens 263 vs 290 ms; 2048 tokens 950 vs 990 ms.

## Where the time goes (step-phase trace, rank 0, one traced launch each; verify pass)

| rows | pass wall IQ8R / IQR | trunk GEMMs IQ8R / IQR | head IQ8R / IQR |
|---|---|---|---|
| 4 | 30.9 / 39.8 ms | 14.7 / 19.8 | 0.87 / 5.30 |
| 16 | 33.5 / 43.1 | 15.6 / 21.0 | 0.94 / 5.52 |
| 24 | 36.7 / 74.8 | 18.3 / 29.8 | 0.99 / 26.87 |

Drafter passes are equal for both containers. The IQR vocabulary-head tile spills at every token-tile count (1280, 5576, 8196, 11012 B private at 1-4 tiles) and 17-32 rows use two tiles; the trunk's 17-64 row path for trellis forms is the wide tile (spills 3.6 KB). Decode GEMV and 16-row tiles are clean (the GEMV tile has no shim exchanges). Tracked as VTP57 and VTP56.
