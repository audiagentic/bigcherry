# VTP59 on 2.3: medium-prompt concurrency, qwen27b IQ8R TP2 and TP3 (bc-radiance-2030.0.0)

Scope:
1. Find a latency-safe setting from the existing knobs for the TP2 four-stream 4K/128 inversion.
2. Confirm the chosen TP3 long-response setting (interleave off, 75 ms mixed target) on 2030.0.0.

No scheduler or simulator code was changed, and no power setting either.

## Method

- **Build:** the installed bc-radiance-2030.0.0 engine and bc-serve, unchanged. Each arm is a shipped profile
  (`qwen27b-2xtx` or `qwen27b-tp3`) plus `--set` overrides. The native tuning receipt is checked against the manifest
  at start and end of each launch, and every receipt held.
- **Fixed in every arm:**
  - power profile `stock` (observe; the shipped `efficient` profile is an open owner decision);
  - prefix cache off, so repeated prompts cannot hit it;
  - temperature 0, 3 repetitions per cell, seed-1 prompts matching the historical cohorts.
- **Order:** fresh managed start per arm, one short gpu lease per arm.
  - TP2 ABAB: A C A C.
  - TP3: A T X A T X.
- **Runs:** 2026-11-28, 16:58 to 17:55 AEDT. One TP2 ABAB was interrupted by an outside change to the install root;
  it is excluded and was rerun from a fresh pair.
- **Metrics:** decode ms/step, TTFT max and p95 event gap are reported separately. Aggregate t/s is context.
- **Latency budget used for the verdicts:**
  - a candidate must not raise the p95 event gap by more than 10% over the default;
  - it must not raise TTFT max;
  - it must not lose more than 3% aggregate on any measured workload.
- **Quality:** KL score against the bf16 corpus-4 reference (8240 positions), default and candidate in the same run.
  KL mode runs one sequence with no speculation.

## (1) TP2 `qwen27b-2xtx`

Screen: one launch per arm.

| arm | 4×4K/128 t/s | TTFT max | p95 gap | 4×512/128 t/s |
|---|---|---|---|---|
| default | 44.5 | 10.43 s | 34.8 ms | 232.8 |
| pipelined prefill on, min 256 | 49.4 | 9.25 s | 35.5 ms | 249.3 |
| pipelined prefill + `max-num-batched-tokens` 4096 | 58.8 | 7.35 s | 29.7 ms | 248.8 |
| pipelined prefill + interleave 0 + mixed target 75 ms | 49.4 | 9.04 s | 133.5 ms | 201.7 |

ABAB: default (A) against pipelined prefill on, min rows 256, batch 4096 (C). Medians are over 2 fresh launches; the
4×4K/128 row also spans seeds 1 to 3 (n = 6).

| workload | arm | aggregate t/s | decode ms/step | TTFT max | p95 gap | acceptance |
|---|---|---|---|---|---|---|
| 4×4K/128 | A | 43.4 | 25.09 | 10.75 s | 34.7 ms | 0.692 |
| 4×4K/128 | C | **58.2** | 26.90 | **7.57 s** | **29.4 ms** | 0.675 |
| 2×4K/128 | A | 51.9 | 27.15 | 3.81 s | 27.6 ms | 0.712 |
| 2×4K/128 | C | 60.7 | 27.19 | 3.06 s | 27.7 ms | 0.699 |
| 6×4K/128 | A | 47.6 | 25.90 | 14.98 s | 31.0 ms | 0.661 |
| 6×4K/128 | C | 58.1 | 26.61 | 12.07 s | 29.4 ms | 0.664 |
| 4×4K/512 | A | 117.2 | 26.96 | 12.75 s | 29.3 ms | 0.621 |
| 4×4K/512 | C | 143.2 | 27.96 | 8.83 s | 30.3 ms | 0.607 |
| 4×512/128 | A | 234.6 | 27.79 | 0.97 s | 28.8 ms | 0.675 |
| 4×512/128 | C | 247.3 | 27.92 | 0.83 s | 28.9 ms | 0.674 |

- C removes the inversion. Four streams now outrun two at the default (58.2 vs 51.9 t/s).
- C is within the latency budget at every workload: the p95 gap is level or lower, and TTFT max is lower everywhere.
- The decode step is up to 1.8 ms longer at 4×4K/128, because more rows go into each step.

Quality (`qwen27b-2xtx`):

| arm | KLD mean | KLD p99 | top-1 | ppl (reference 7.160) | gate |
|---|---|---|---|---|---|
| default | 0.001985 | 0.01677 | 96.59% | 7.161 | pass |
| C | 0.001969 | 0.01680 | 96.64% | 7.163 | pass |

**Verdict (TP2): new default.** Pipelined prefill on, `pipelined-prefill-min-rows: 256`, `max-num-batched-tokens: 4096`,
with interleave and mixed target left at their defaults. Earlier screens reached a similar throughput with interleave
off and a 250 ms target, but at a 273 ms p95 gap. Keeping the default interleave is what holds latency.

## (2) TP3 `qwen27b-tp3`, 8 streams

Shipped already: arrival governor (default), pipeline floor 256, the four-group all-reduce pin. Medians over 2 fresh
launches; the 4K/512 row also spans seeds 1 and 2 (n = 4).

| workload | arm | aggregate t/s | decode ms/step | TTFT max | p95 gap | acceptance |
|---|---|---|---|---|---|---|
| 8×4K/512 | A shipped | 149.5 | 24.03 | 22.39 s | 34.5 ms | 0.600 |
| 8×4K/512 | T interleave 0 + 75 ms | 178.7 | 37.54 | 15.85 s | **163.9 ms** | 0.603 |
| 8×4K/512 | X batch 4096 | 154.5 | 23.99 | 21.87 s | 31.0 ms | 0.602 |
| 8×512/128 | A | 284.3 | 29.95 | 2.27 s | 40.2 ms | 0.713 |
| 8×512/128 | T | 281.4 | 36.29 | 2.00 s | 40.1 ms | 0.701 |
| 8×512/128 | X | **308.0** | 35.80 | **1.60 s** | 40.3 ms | 0.713 |
| 8×512/192 | A | 345.5 | 31.74 | 2.25 s | 39.9 ms | 0.707 |
| 8×512/192 | T | 350.2 | 37.45 | 1.99 s | 40.1 ms | 0.696 |
| 8×512/192 | X | **378.2** | 37.16 | **1.61 s** | 40.1 ms | 0.714 |

- **T does not confirm as latency-safe on 2030.0.0.** All four long runs landed in the slow regime: a p95 gap of
  163 to 164 ms against 34.5 ms, for +19% long throughput. Earlier screens on the bump branch had shown both regimes.
  T is not a default; it remains a throughput profile for callers that accept the gap.
- **X (batch 4096) is within the latency budget:**
  - short: +8% and +9.5% aggregate;
  - TTFT max 1.6 s against 2.25 s, because eight 512-token prompts fit one prefill step;
  - long: +3% throughput, with p95 gap and TTFT not worse.
- The cost of X: decode ms/step on short workloads rises from about 31 to about 37 ms, because more rows go into each
  step. The p95 token gap is unchanged.
- 8×512/192 at 378 t/s is above the stored 2020.8.5 record of 360.8 t/s (VTP61).

Quality (`qwen27b-tp3`):

| arm | KLD mean | KLD p99 | top-1 | ppl (reference 7.160) | gate |
|---|---|---|---|---|---|
| default | 0.001379 | 0.01121 | 97.22% | 7.173 | pass |
| X | 0.001287 | 0.01151 | 97.37% | 7.168 | pass |

The default reproduces the 2.3 bump's canonical score exactly.

**Verdict (TP3): new default `max-num-batched-tokens: 4096`.** Keep interleave and mixed target at their defaults. The
chosen long-response setting (interleave 0, 75 ms) is not confirmed as latency-safe.

## Not covered

- The long-context profiles (`qwen27b-tp3-long*`, 262k context, budget governor) are untouched and not measured here.
- No mixed short-beside-long arrival screen in this round.
- IQ6R and FlashNext layouts were not measured.
- `summary.json` holds every cell, the effective scheduler settings per arm, and the quality results.
