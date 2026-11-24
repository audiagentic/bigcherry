# flashnext: 4gpu (RDNA2 tail) against 2xtx-combine, tp2xtx and 2xtx-rdna2-tail (2020.8.5)

Method: release 2020.8.5 installed from its archive; one engine launch per arm, each under its own lease; speculation depth 3, untraced; 512-token prompts, 192 new tokens, temperature 0; 1, 4 and 8 streams, seeds 1-3 x 2 repetitions per seed. Order of the first two cycles: A B D C C D B A (A = 4gpu, B = 2xtx-combine, D = tp2xtx, C = 2xtx-rdna2-tail); cycle 3 repeated A B B A. Arms A3, A4, B3 and B4 each got a warm-up of three requests per server before measuring, followed by a cold-expert probe (twelve prompts from distinct domains). The earlier arms did not get the warm-up request: their first arm of each config shows a cold-cache outlier (A1, B1) and is excluded from the comparison below.

## Decode ms per step (mean of seed medians), aggregate tokens/s in brackets

| config | arm | 1 stream | 4 streams | 8 streams |
|---|---|---|---|---|
| 4gpu | A1 (cold) | 31.16 (86.0) | 98.40 (99.7) | 101.11 (175.0) |
| 4gpu | A2 | 24.19 (111.0) | 39.52 (213.8) | 55.47 (284.3) |
| 4gpu | A3 (warm) | 24.18 (110.9) | 39.29 (218.3) | 54.04 (283.6) |
| 4gpu | A4 (warm) | 25.16 (107.9) | 39.91 (214.8) | 54.31 (283.1) |
| 2xtx-combine | B1 (cold) | 30.36 (100.3) | 39.72 (228.2) | 51.96 (311.9) |
| 2xtx-combine | B2 | 22.92 (128.5) | 36.44 (242.2) | 49.13 (318.5) |
| 2xtx-combine | B3 (warm) | 33.95 (88.8) | 45.08 (205.0) | 51.63 (324.8) |
| 2xtx-combine | B4 (warm) | 22.93 (128.5) | 36.56 (248.3) | 48.97 (322.0) |
| 2xtx-rdna2-tail | C1 | 24.02 (104.7) | 38.66 (207.7) | 51.92 (261.1) |
| 2xtx-rdna2-tail | C2 | 24.03 (104.7) | 39.49 (207.0) | 51.10 (259.4) |
| tp2xtx | D1 | 28.44 (90.6) | 53.32 (164.2) | 65.88 (203.2) |
| tp2xtx | D2 | 24.88 (98.1) | 55.77 (156.3) | 67.94 (193.7) |

Reading: at 8 streams 4gpu runs 54-55 ms/step against 49-52 for 2xtx-combine, about 6-12% slower, and 284 against 318-325 tokens/s. At 1 and 4 streams 4gpu is level with or slower than combine (B3's 1- and 4-stream numbers are a cold outlier after its launch). tp2xtx is clearly slowest (66-68 ms at 8 streams). The RDNA2 tail does not pay on this model: its per-request floor (about 190 us against about 105 us on an XTX) holds the collective back.

Cold-expert probe (events/s, same server after the warm-up): 4gpu 92.7 and 92.7; 2xtx-combine 78.7 and 98.1. No clear tail benefit.

## Memory placement (from the planner)

| config | experts on helpers | VRAM (helpers) | host pool | KV pool | margin |
|---|---|---|---|---|---|
| 4gpu | 25078 | 8.03 GiB | 3.37 GiB | 69%, 5.5 full sessions | 3.4 GB |
| 2xtx-combine | 19092 owned | 6.69 GiB | 17.49 GiB | 70%, 5.6 full sessions | 16.9 GB |

Prefetch copies per pass: 80 on 4gpu, 304 on 2xtx-combine. Max context in both: 16384 tokens (as configured).

## Quality (KLD against the stored reference, corpus 4, 8240 positions)

Speculation depth 0 (the scoring engine rejects depth 3), expert_tiered placement as configured.

| config | mean KLD | median | p99 | top-1 agreement | ppl ratio |
|---|---|---|---|---|---|
| 4gpu | 0.0591 | 0.0158 | 0.625 | 0.8837 | 0.9957 |
| 2xtx-combine | 0.0663 | 0.0162 | 0.762 | 0.8837 | 1.0001 |

Caveat, stated plainly: the stored flashnext reference was recorded on an older engine build (release 1170.34.0, tp2xtx layout). Both candidates sit about 0.06 mean KLD away from it, which is the mismatch between that build and this one, not a measure of the layouts. Only the difference between the two rows can be read, and it is small (equal top-1 agreement, equal ppl ratio within 0.5%). The rows are not comparable to later KLD rows made against a different reference.

## To do

Record a fresh flashnext reference on the current build: self-reference with the production config, speculation depth 0, so that later KLD rows are comparable. Cheap; do it when the card is free.
