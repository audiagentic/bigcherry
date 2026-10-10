---
id: QFP49
order: 49
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-10T13:54:01.542976+00:00'
breadth: ''
skill: advanced
created-by: claude
priority: P1
work: L
---

# Prefill cross-card sum: overlap the transfer with compute, narrower wire, rank balance

## Description

The large (RCCL) AllReduce is the largest single cost in Flash-Next prefill and nothing is hiding it.

Measured (run dp1, main at 1042e84d + lock counters, production profile, tools/lab/flash-next/ar-prefill-stats.py over the prefill window without model load; the same at 24K and 98K), per target card: computing 44-49% of the time; in ncclDevKernel_Generic_4 32-37% (10.0 / 9.7 / 11.0 s of a 29.6 s prefill at 24K); idle with no kernel queued 17-19%. There are 96 sums per 512-token batch (7,296 at 24K), about 1.4 ms each on the card. Per sum: the cards arrive 1.3 ms apart on average (median 0.9; the R9700 is usually last), and the exchange after the last card arrives takes 0.9 ms (6.6 s of the 29.6 s). So of a card's time in the sum kernel about two thirds is the exchange through host memory (no peer-to-peer between these cards) and one third is waiting for a slower card.

Already tried and rejected: sending large sums through the chunked cpu-root path (1291, BIGCHERRY_AR_CPU_ROOT_LARGE_MAX_BYTES=64 MiB): 1,339 -> 1,048 t/s at 24K with 1 MiB chunks and 883 with 8 MiB (runs ar1, ar2); decode 89 -> 77 t/s. RCCL is the better of the two existing paths.

Owner, 2026-10-11: "can we copy from or to the card while it is busy so we parallel the work and keep them busy". Yes: a transfer on its own stream runs beside kernels on the compute stream. What is missing is independent work to run meanwhile: within one batch the next op needs the sum.

## Steps

1. Message size and PCIe rate: bytes per sum (n_tokens x n_embd x 4) against the 0.9 ms exchange, per link (x8 / x8 / x4). States how far the exchange is from the link limit.
2. Asynchronous transfer: the exchange on a transfer stream with the consuming kernel waiting on its event, not a kernel spinning on the compute stream. Shows how much of the 32-37% is pure waiting.
3. Half-batch staggering in the Meta split backend (TokenWeave, arXiv 2505.11329): split a prefill batch in two; while half A's sum is in transit the cards compute half B's part of the layer. Default-off flag; attention ordering within a layer must be kept exact.
4. Narrower wire for large sums (1250 q8 wire, 1272 host compressed wire: both unqualified): f16 first, with a CPU f32 reference check.
5. Rank balance: move the tensor split (TS=0.31,0.27,0.42) so the cards arrive together, or let an early card start its transfer before the last arrives.
6. The 17-19% idle: host-side profile of the server during prefill (perf), since the dispatch call itself is about 1 s per 100 s (1356 lock counters).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Each step is one patch or one profile setting, lightweight tier: offline mechanics + patch-lint, activation marker, ABBA on one binary at 8K / 24K / 98K with complete separation, greedy identity where the mechanism is exact (async transfer, staggering) and a CPU f32 reference comparison where it is not (narrow wire), repeated-run identity at 98K (at least twelve runs: the 1356 race was only visible there).

## Effort & Risk



## Standards



## Acceptance Criteria

- The exchange's rate against the PCIe link limit is stated.
- The asynchronous transfer and the half-batch staggering are each promoted or rejected with their measurement.
- The narrow wire is promoted with a reference comparison no worse than production, or rejected.
- A card's computing share of prefill time is reported before and after (ar-prefill-stats.py), not only tok/s.

## Notes

The same structure exists in radiance on two 7900 XTX (collectives through pinned host memory, 33% of two-card time; a lossy wire took prefill from 1,535 to 2,120 tok/s), so what works here should carry over. Threads: QFP41 owns the threading rules and QFP46 the prefill concurrency contracts; the dispatch workers (1356) are evaluated, not validated, since 2026-10-11.

## Change Log

- 2026-10-10T13:54:01.542976+00:00 (created-by): Created by claude
