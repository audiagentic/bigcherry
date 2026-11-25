# Radiance release series: decode and first-token results, 2020.8.0 to 2020.12.0

One table of what each release did to serving speed, assembled from the paired A/B notes already in `docs/evidence` (each row names its source). Every delta is measured within one session; absolute levels drift between sessions (the same 2020.8.2 arm read 35.79 ms and 31.87 ms at six streams in two sessions), so do not read the first and last rows against each other to the millisecond. Layout: `qwen27b-tp3` (IQ8R) or `qwen27b-tp3-iq6r` (IQR-V4), three cards, DFlash2 drafter at depth 3, 512-token prompts, 192 new tokens, temperature 0, ms per decode step.

## IQ8R, three cards: decode ms/step

| Step | Change | 6 streams | 8 streams | Source |
|---|---|---|---|---|
| 2020.8.0 to 8.1 | K-parallel tile spill fix | 43.79 to 36.69 (-16.2%) | 48.21 to 37.67 (-21.9%) | 2026-11-23-27b-decode-gemm-efficiency |
| 8.1 to 8.2 | K-parallel accumulate-once | 39.37 to 35.79 (-9.1%) | 39.06 to 36.44 (-6.7%) | 2026-11-24-vtp37-kpar-acc-once-ab |
| 8.2 to 8.3 | drafter 4-bit row as 16-token launches | 31.87 to 28.74 (-9.8%) | 36.27 to 32.17 (-11.3%) | 2026-11-24-vtp37-drafter-two-m16-ab |
| 8.3 to 8.4 | logits head native sums | -0.91 ms (s.e. 0.69) | -0.03 ms | 2026-11-24-vtp37-logits-acc-once-ab |
| 8.4 to 8.5 | look-ahead rule | -0.3% | +0.2% | 2026-11-24-vtp37-kpar-lookahead-ab |
| 8.5 to 8.6 | decode attention native sums | -0.2% | +0.3% | 2026-11-24-decode-attn-pv-split-ab-8-5-vs-8-6 |
| 8.6 to 12.0 | IQR kernels, packages, fused tiles | level within the floor at every pair | level | 2026-11-24-gate-e-runs and the regression notes |

The first three steps carry nearly all of the gain. The rest are bit-identical kernel improvements that serving at these stream counts cannot resolve (the harness floor is 0.85 ms at six streams and 0.26 ms at eight).

## IQR-V4, three cards: decode ms/step

| Release | 1 stream | 6 streams | 8 streams |
|---|---|---|---|
| 2020.8.7 with the weighted head only | about 30 | 56 to 61 | 58 to 60 |
| 2020.9.2 (head tile, 16-row accumulate-once) | 21.6 | 29.9 | 33.0 |
| 2020.11.0 to 2020.11.4 | 21.5 | 29.9 | 32.9 to 33.1 |
| 2020.12.0 | 21.47 | 30.27 | 33.21 |

## Latest release (2020.12.0) against the previous (2020.11.5), and against the start of the series

Paired ABBA, four launches an arm, one lease each, fresh work root, text identical at temperature 0. No regression.

| Container | Metric | 2020.11.5 | 2020.12.0 | change |
|---|---|---|---|---|
| IQ8R | decode, 1 stream | 20.25 | 20.25 | 0.0% |
| IQ8R | decode, 6 streams (10-launch repeat) | 28.56 | 28.88 | +1.1% (slow samples in both arms) |
| IQ8R | decode, 8 streams | 31.93 | 32.03 | +0.3% |
| IQR | decode, 1 stream | 21.47 | 21.47 | 0.0% |
| IQR | decode, 6 streams (10-launch repeat) | 30.02 | 30.27 | +0.8% |
| IQR | decode, 8 streams | 33.47 | 33.21 | -0.8% |

First token (median ms, six samples): IQ8R 68 / 100 / 127 / 132 / 264 / 1034 tokens 85.6 / 94.9 / 104.4 / 131.0 / 164.3 / 445.7 to 87.5 / 97.0 / 106.9 / 130.3 / 165.2 / 432.8 (ranges overlap); IQR 96.5 / 107.3 / 117.1 / 146.6 / 194.3 / 463.0 to 94.3 / 106.0 / 116.2 / 147.1 / 191.5 / 463.7. The 20 ms step between 127 and 132 tokens (one extra 128-row prefill band) is present in both releases and in 2020.8.7.

Against the start of the series (different sessions, so indicative): IQ8R six and eight streams went from about 43.8 and 48.2 ms at 2020.8.0 to about 28.9 and 32.0 ms at 2020.12.0, roughly 34 percent less at both; one stream from 21.8 ms (8.0, prefetch on) to 20.25 ms.

## Not covered

KLD and top-1 per release (last measured at 2020.9.2 for IQR and 2020.8.7 for IQ8R); flashnext parity for 2020.12.0 (the last compared release is 2020.11.3); releases before 2020.8.0, which used a different harness and engine pin.
