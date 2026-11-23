# Three one-card blocks at once: no cross-card contention

Question: if each card runs its own independent DiT pass (guidance-pass parallelism: no per-layer reduction), do the
cards slow each other down through the shared host?

Method: LTX-2.5 IQ8R, one video block (attention + FFN) at 2048 tokens, run as three separate processes at the same time,
one per card (two RX 7900 XTX, one R9700), under one `gpu` lease. Same lab mode as the solo timings in
`../2026-11-22-ltx03-issued-block-timing/` (`LTX03_TIME_TP1`).

| card | solo block | concurrent block | solo FFN | concurrent FFN |
|---|---|---|---|---|
| R9700 | 6.68 ms | 6.68 ms | 3.65 ms | 3.67 ms |
| XTX a | 13.05 ms | 13.02 ms | 6.70 ms | 6.70 ms |
| XTX b | 13.05 ms | 13.57 ms | 6.70 ms | 6.94 ms |

Result: concurrent times match solo within 4% (one XTX is 4% slower on the block, within its run-to-run spread). Independent
passes on the three cards do not contend; no host-memory traffic is involved.

Consequence (arithmetic from these block times, not a measured step): four guidance passes with the R9700 taking two and each
XTX one finish in about 13.4 ms of block time against 26.7 ms for the R9700 alone, about 2.0x the best single card. The
distilled model runs one pass and gains nothing. Full-model residency on a 24 GB XTX at the target resolution is still
unmeasured: the engine's budget plan shows a 9.2 GiB activation arena at the maximum 33,833-row step against 18.2 GiB of DiT weights.
