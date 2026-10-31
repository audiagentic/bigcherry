---
id: RHP10
order: 10
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-31T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Long-context thrash experiment

## Description

Test whether expert residency changes explain the 290K-1M decode slowdown (decode 92-122 tok/s to ~290K, 74 at 580K, 50 at 1M). A falling decode rate does not by itself show thrashing: growing KV attention work and memory pressure compete as explanations. Prior evidence: a 6 GiB tail cut expert promotions 8,084 -> 5,767 and host-to-device traffic 51.3 -> 32.2 GiB over a short-context run with no speed change.

## Steps

1. Write the protocol, experiment or collector against the code as it is (radiance 1.14.0, remote-MoE protocol v19).
2. Land card-free tests first where the item has any logic; measure only with the gpu lease.

## Detailed Solution & Technical Design

## Code Samples & Guidance

## Files

## Validation

At 1M context with tails of 0 / 3 / 6 / 14 GiB, with the context filled to ~290K and ~580K: resident expert bytes, misses, mover activity, transfer volume, KV occupancy and decode/prefill throughput; the result decides RHP05-07.

## Effort & Risk

Depends on: none (needs the gpu lease and all four cards). Added by the 2026-10-31 GPT review of this plan.

## First long-context runs, 2026-10-31 (`rad23-longctx`, `rad23-longctx2`)

Flash-Next, `flashnext-2xtx-rdna2-tail` (4 GiB tail), `ctxsweep.py` over the documentation text; used context 330K and 688K (the 295K and 590K targets tokenise longer). Static heat is `perf51/heat.rank0.hot-first`; live heat is the snapshot of RHP08 inverted the same way. The 688K row of `rad23-longctx` follows a 330K request that stays cached; `rad23-longctx2` is the 688K request alone on a fresh server, three arms back to back.

| Run | Arm | Prefill tok/s | Decode tok/s | Promotions | Host-to-GPU GiB |
|---|---|---|---|---|---|
| after 330K | XTX 22,528, static | 2413 / 1392 | 113 / 69.8 | 11365 | 2480 |
| after 330K | XTX 23,500, live | 2507 / 1492 | 110 / 87.7 | 10001 | 2126 |
| 688K alone | XTX 22,528, static | 1833 | 77.9 | 8929 | 1128 |
| 688K alone | XTX 23,500, static | 1893 | 74.8 | 8002 | 993 |
| 688K alone | XTX 23,500, live | 1944 | 98.5 | 8067 | 970 |

(Prefill/decode pairs in the first two rows are 330K / 688K.) The extra XTX budget alone cuts host-to-GPU traffic by 12% and does not move decode; adding the live heat on top gives +26% decode at 688K in both settings (87.7 against 69.8, 98.5 against 77.9) and +4 to 7% prefill. Prefill at 688K is 1392 to 1944 tok/s depending on what is already cached, so only runs set up the same way compare. Decode pairs spread widely inside a run (69 to 86), so the second fact to keep is one run per arm in the second table.

Caveat that blocks a conclusion: the live heat was recorded on the same documentation text the sweep reads, so it is fitted to the test. The held-out test is heat from one corpus against a sweep over another.
