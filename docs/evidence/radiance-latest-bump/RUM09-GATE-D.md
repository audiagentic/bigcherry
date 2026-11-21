# RUM09 Gate D: paired Flash-Next comparison, Radiance 1.17 (1170.35.0) against 2.2

Configuration flashnext-4gpu, 262144-token context, 8 sessions, stock power profile, same prompts, seed and topology on both arms.
Baseline: two launches of the 1170.35.0 control. Candidate: two launches of the 2.2 build carrying the serving-pool fit (066778340), measured after
the first candidate pair failed the gate on one metric.

## First candidate pair (build before the pool fit): FAIL on one metric

Prefill and time to first token were better on every row. `decode.long8_ms_per_step` was +16.6% (allowed 12.6%). Per-launch rounds showed the cold
pass faster and the warm pass (prefix-cache hit) 40% slower: 37.4 ms against 26.6 ms per step. Startup settings were identical (FP8 KV, 4096-token
checkpoints, 72 snapshot slots, 12 GiB host prefix KV, nearest resume). Per-op rank-0 timing on the 2.2 build showed only paged attention
(0.025 to 2.35 ms) and the following embedding lookup (0.015 to 0.64 ms) growing between the cold and warm rounds.

Cause: the GPU KV pool was 17.61 GiB per card against 20.97 GiB on 1.17. `refit_serving_pools` fell back to the calibrated minimum whenever
the independently MiB-rounded weight and KV pools missed the budget by one MiB, discarding the measured optional headroom (about 3.4 GB).
The fix (066778340) allocates the largest admissible pool up to the measured preference, subject to the calibrated minimum and the expert floor.

## Second candidate pair (with the pool fit): PASS

| metric | baseline | 2.2 | change | allowed |
|---|---:|---:|---:|---:|
| decode1 ms/step | 26.64 | 24.50 | -8.0% | 14.1% |
| decode6 ms/step | 49.83 | 48.03 | -3.6% | 5.0% |
| long8 ms/step | 27.57 | 27.06 | -1.9% | 12.6% |
| sess8 ms/step | 29.66 | 29.01 | -2.2% | 8.3% |
| doc prefill tok/s | 3254 | 3662 | +12.5% | |
| long7 prefill tok/s | 2207 | 2453 | +11.2% | |
| long8 prefill tok/s | 2099 | 2430 | +15.8% | |
| cold TTFT s | 10.64 | 10.08 | -5.3% | |
| sess8 TTFT s | 30.18 | 27.29 | -9.6% | |

long7 decode (informational) is +4.2%. Acceptance differs by build and is not gated.

## Method notes

Two launches per arm (the minimum the baseline tolerance needs); a third was not run because neither the pass nor the failure was within the old arms' spread.
The first candidate pair used the same two old launches as the baseline.
