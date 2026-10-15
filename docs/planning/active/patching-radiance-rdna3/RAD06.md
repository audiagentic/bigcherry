---
id: RAD06
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-14T20:15:06.353093+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
---

# Prefill: multi-buffered helper hand-off pipeline and gfx11-native grouped MoE GEMM

## Description

Prefill profile 2026-10-15 (R9700 engine + two XTX helpers, heat ownership, 22 GiB, GEMV on): helper large-lane request ~5.7 ms (routes 0.62-0.88, gate_up ~3.0, quant 0.085, down ~1.6, reduce 0.46 ms); engine moe_mailbox_add (wait+merge) 47.7% avg 3.56 ms, own moe_gemm_q 12.4%, dense 12.8%, publish 4.9%; prefill ~2,500 tok/s vs two-R9700 ~6,500. Luna review (req_e0aeb86a624d415c): shm protocol near minimal latency; serialised payload/compute dominates. Ranked: (1) multi-buffered chunk pipeline (2-4 payload slots per rank, per-chunk generations, independent merge) so helpers compute chunk N while the engine does chunk N+1; (2) gfx11-native tiled grouped MoE GEMM (bf16 WMMA, nibble dequant into LDS) for the ~4.6 ms of helper GEMM; (3) ownership balanced by measured prefill route load; (4) per-helper early merge; (5) hardware wait primitive.

## Steps

1. Measure remote-token density per prefill layer (compact result rows may cut the 21 MB/helper write). 2. Protocol hardening first: exact-generation completion, error completion path so a dead helper cannot hang the engine, fail-closed offset validation, per-slot activation staging. 3. Payload slot ring + chunked publish/merge in the engine MoE block (rad0002) and helper. 4. gfx11 grouped GEMM prototype against moe_gemm_w4a8 on the CPU compare. 5. Rebalance ownership by prefill histogram if helper skew is measured.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

--compare per layer; KL vs the XTX-pair reference within the noise floor; balanced prefill A/B on the 7306-token prompt and a 32K prompt; helper per-stage timing; no stalls under the queue-stall watcher.

## Effort & Risk

Pipelining changes the engine MoE block's step structure (patch) and the protocol (v11); the GEMM is a new native kernel. Both gate the prefill target.

## Standards



## Acceptance Criteria



## Notes

Follows RAD02/RAD05.

2026-10-15 07:45 progress: protocol hardening done (e26c1b94, v11: exact-generation completion, helper_fault word so a stopped helper traps the engine instead of hanging it). R3_MOE_NATIVE=1 in the helpers (gfx11 native bf16 WMMA inner loop) now default (2bf231e3): profiled helper prefill gate/up 3.0 -> 2.25 ms, down 1.59 -> 1.23 ms, engine wait+merge 3.56 -> 2.91 ms; balanced A/B prefill 2518/2519 -> 2324(warm-up)/2665, decode 91.4/95.6 -> 108.2/108.8 (different text); KLD 0.0642 / top-1 88.73%. Best Flash-Next on Brutus now ~2,665 prefill / ~108.8 decode (R9700 engine + two XTX helpers, heat ownership, 22 GiB, GEMV, native). Still open: chunk pipeline (multi-buffered payload), deeper gfx11 grouped GEMM, ownership rebalancing.

2026-10-15 08:10 helper budget re-tuned with GEMV + native (balanced 20/22/22/20/20/22 GiB): 22 GiB prefill 2737/2762/2763, decode 108.0/108.8/~108.5 (69036e14); 20 GiB prefill 2744/2716/3011(outlier), decode 89.2/89.1/89.2 (fe17e0de). With decode no longer helper-bound, more helper experts no longer cost prefill: 22 GiB is the configuration. Best Flash-Next on Brutus: ~2,755 prefill / ~108.5 decode (R9700 engine + two XTX helpers). Models/quantisation recorded in engines/radiance/README.md (d6ae81bc).

2026-10-15 08:20 analysis: MoeFP8::step already loops prefill in row passes (c.rows = max_tok -> one 4096-row pass). Per layer the R9700's own experts take ~1 ms (already overlapped by early publish) and the helpers ~4.5 ms, so a within-layer chunk pipeline cannot go below the helper time. Heat ownership gives the R9700 most routes (hot experts, ~1 ms) and the XTX helpers the cold experts with few rows each, where the WMMA grouped GEMM's 16-row tiles waste work: helper cost per route is ~an order above the R9700's. Revised priority: (1) helper prefill kernel for few-rows-per-expert -- read each owned expert's weights once a layer (~0.4 GB a helper, ~0.4 ms of bandwidth) and compute all its rows from them (extends RAD05's GEMV); (2) cross-layer row pipelining (earlier tokens start the next layer's attention while later tokens' experts are still remote; RAD04-style engine change; upstream's simpler overlap lost 6%). Expected ceiling after (1): ~3,500-4,000 tok/s prefill; two-R9700 ~6,500 (P2P, 32 GB a card) is unlikely on this hardware.

2026-10-15 few-rows-a-expert large-lane kernel (R3_MOE_HELPER_ROWS=1, commits aceef746 + 8cae20dc): correct (--compare layer 20, 1024 rows: rel_l2 0.0011/0.0012, same as the grouped GEMM) but much slower. Balanced A/B, 9700 engine + two XTX helpers, 22 GiB, heat, 7,306-token prompt, order 0 1 1 0 0 1: ROWS=0 prefill 2757.9/2764.0/2761.3, decode 108.1/108.8/108.7; ROWS=1 prefill 1719.8/1682.1/1726.8, decode 96.1/96.2/96.3 (complete separation, -38% prefill, -11% decode: MTP verify batches use the large lane too). Completion hashes differ (69036e14 vs cb7ecba3): bf16 rounding of a different accumulation order, within the compare tolerance. Why slow: scalar FMA against bf16 WMMA, and every N tile of 32 re-dequantises the same activation rows (40 tiles for gate/up), so the routed-row reuse it was meant to win is lost. Kept default off as evaluated/shelved. Remaining prefill levers: (1) multi-buffered chunk pipeline, (2) WMMA-based few-row tiles (16-row WMMA with activations shared across N in LDS), (3) prefill-load ownership rebalance.

## Change Log

- 2026-10-14T20:15:06.353093+00:00 (created-by): Created by agent
- 2026-10-14T20:41:34.505375+00:00 (updated-by): Updated: section:notes

## Ledger-events


- chg_20261014_204134_flash-next-decode-on-brutus-re_3947
- 2026-10-14T20:41:42.123589+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-14T21:08:32.689755+00:00 (updated-by): Updated: section:notes
- 2026-10-14T21:09:38.493597+00:00 (updated-by): Updated: section:notes
- 2026-10-14T22:19:13.458071+00:00 (updated-by): Updated: section:notes
- chg_20261014_221932_added-an-experimental-prefill_2342
- 2026-10-14T22:19:36.581927+00:00 (updated-by): Updated: section:ledger-events
