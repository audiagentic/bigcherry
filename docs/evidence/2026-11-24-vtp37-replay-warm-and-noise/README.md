# VTP37: prefetch-warm replay and the serving harness's noise floor

Two measurements that change how decode changes are gated. Both within one session; compare only within a table.

## 1. Cold against prefetch-warm replay of the K-parallel int8 tile

The cold replay flushes 256 MiB of unrelated data before every call (`--cold`-style). In the engine the prefetch op already reads the leading 12 MiB / 8 MiB of the next GEMM's weights into the
cache before the GEMM runs, so engine GEMMs are partly warm. The warm mode flushes, then reads the first 12 MiB of the call's own weights (outside the timed region), then times the call.
One XTX, scratch driver (includes about 10 us of launch a call), median of 30 calls, three repeats, 128-wide and 256-wide scale-group forms, outputs checksum-identical across all arms.

Effect of warming alone, same kernel (shipped main), M 16 / M 32 (256-wide form; the 128-wide form is within 1-3 points of it):

| shape (N x K) | weight MB | cold us | warm us | warming, M 16 | warming, M 32 |
|---|---|---|---|---|---|
| 13312 x 5120 | 68.7 | 97.8 | 86.6 | -11.5% | -6.8% |
| 5120 x 6656 | 34.3 | 54.0 | 42.4 | -21.5% | -16.8% |
| 6144 x 5120 | 31.7 | 49.3 | 37.5 | -23.9% | -14.5% |
| 3072 x 5120 | 15.9 | 31.2 | 21.8 | -30.1% | -9.5% |
| 5120 x 2304 | 11.9 | 28.3 | 21.4 | -24.4% | -16.5% |
| 5120 x 1536 | 7.9 | 23.1 | 18.6 | -19.5% | -12.7% |

Stream rate at M 16 (weight bytes over time, 960 GB/s nominal, launch included): cold 702 / 636 / 643 / 508 / 420 / 343 GB/s (73 / 66 / 67 / 53 / 44 / 36%); warm 793 / 810 / 845 / 727 / 556 / 426 GB/s
(83 / 84 / 88 / 76 / 58 / 44%). The three large rows reach 83-88% of nominal warm against 66-73% cold; the small rows stay low in both because the launch term dominates them.

Look-ahead removal (#1462) in the same modes, 256-wide form, before -> after, cold / warm: 13312x5120 -1.8..-4.2% / -1.8..-2.4%; 5120x6656 -2.5..-7.4% / -2.6..-5.8%; 6144x5120 -3.4..-4.5% / -3.6..-4.3%;
5120x2304 -5.0..-8.7% / -3.4..-8.2%; 5120x1536 -4.6..-9.1% / -0.8..-5.1%; 3072x5120 -0.6..-5.8% / +1.4..-4.9%. Warming shrinks the look-ahead gain by roughly a third on average but does not remove it;
the 128-wide form shows at most -5% cold and 0..-4% warm. So warming only partly explains why the replay gain was not visible in the serving A/B of the same change.

Consequence: report cold and prefetch-warm side by side for every kernel replay from now on; rank by the warm column.

## 2. Noise floor of the serving A/B harness (2020.8.5 against itself)

`qwen27b-tp3`, speculation depth 3, untraced, same ABBA protocol as the A/B notes (two cycles, one engine launch per arm, seeds 1-3, three repetitions per seed), the same release in both positions.

| streams | arm means, ms/step | mean | sd of arm means | sd of seed means (n = 24) | same-release B-A (s.e.) | minimum detectable difference, 2 s.e., 4 vs 4 arms |
|---|---|---|---|---|---|---|
| 6 | 29.62 30.27 29.13 30.86 30.13 30.60 29.37 30.18 | 30.02 | 0.60 | 0.90 | -0.35 ms (0.43) | about 0.85 ms (2.8%) |
| 8 | 31.67 31.46 31.78 31.49 31.52 31.68 31.99 31.82 | 31.68 | 0.18 | 0.48 | +0.10 ms (0.13) | about 0.26 ms (0.8%) |

A two-cycle A/B at 6 streams cannot show a difference below about 0.85 ms; at 8 streams it can show about 0.26 ms. The A/B notes of this session read against these floors: kpar accumulate-once and the drafter
two-M16 route are well above them; the logits change (predicted 0.2-0.3 ms) and the look-ahead removal are below the 6-stream floor and at or below the 8-stream floor.
