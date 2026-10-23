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

2026-10-15 07:45 progress: protocol hardening done (e26c1b94, v11: exact-generation completion, helper_fault word so a stopped helper traps the engine instead of hanging it). R3_MOE_NATIVE=1 in the helpers (gfx11 native bf16 WMMA inner loop) now default (2bf231e3): profiled helper prefill gate/up 3.0 -> 2.25 ms, down 1.59 -> 1.23 ms, engine wait+merge 3.56 -> 2.91 ms; balanced A/B prefill 2518/2519 -> 2324(warm-up)/2665, decode 91.4/95.6 -> 108.2/108.8 (different text); KLD 0.0642 / top-1 88.73%. Best Flash-Next on dev now ~2,665 prefill / ~108.8 decode (R9700 engine + two XTX helpers, heat ownership, 22 GiB, GEMV, native). Still open: chunk pipeline (multi-buffered payload), deeper gfx11 grouped GEMM, ownership rebalancing.

2026-10-15 08:10 helper budget re-tuned with GEMV + native (balanced 20/22/22/20/20/22 GiB): 22 GiB prefill 2737/2762/2763, decode 108.0/108.8/~108.5 (69036e14); 20 GiB prefill 2744/2716/3011(outlier), decode 89.2/89.1/89.2 (fe17e0de). With decode no longer helper-bound, more helper experts no longer cost prefill: 22 GiB is the configuration. Best Flash-Next on dev: ~2,755 prefill / ~108.5 decode (R9700 engine + two XTX helpers). Models/quantisation recorded in engines/radiance/README.md (d6ae81bc).

2026-10-15 08:20 analysis: MoeFP8::step already loops prefill in row passes (c.rows = max_tok -> one 4096-row pass). Per layer the R9700's own experts take ~1 ms (already overlapped by early publish) and the helpers ~4.5 ms, so a within-layer chunk pipeline cannot go below the helper time. Heat ownership gives the R9700 most routes (hot experts, ~1 ms) and the XTX helpers the cold experts with few rows each, where the WMMA grouped GEMM's 16-row tiles waste work: helper cost per route is ~an order above the R9700's. Revised priority: (1) helper prefill kernel for few-rows-per-expert -- read each owned expert's weights once a layer (~0.4 GB a helper, ~0.4 ms of bandwidth) and compute all its rows from them (extends RAD05's GEMV); (2) cross-layer row pipelining (earlier tokens start the next layer's attention while later tokens' experts are still remote; RAD04-style engine change; upstream's simpler overlap lost 6%). Expected ceiling after (1): ~3,500-4,000 tok/s prefill; two-R9700 ~6,500 (P2P, 32 GB a card) is unlikely on this hardware.

2026-10-15 few-rows-a-expert large-lane kernel (R3_MOE_WORKER_ROWS=1, commits aceef746 + 8cae20dc): correct (--compare layer 20, 1024 rows: rel_l2 0.0011/0.0012, same as the grouped GEMM) but much slower. Balanced A/B, 9700 engine + two XTX helpers, 22 GiB, heat, 7,306-token prompt, order 0 1 1 0 0 1: ROWS=0 prefill 2757.9/2764.0/2761.3, decode 108.1/108.8/108.7; ROWS=1 prefill 1719.8/1682.1/1726.8, decode 96.1/96.2/96.3 (complete separation, -38% prefill, -11% decode: MTP verify batches use the large lane too). Completion hashes differ (69036e14 vs cb7ecba3): bf16 rounding of a different accumulation order, within the compare tolerance. Why slow: scalar FMA against bf16 WMMA, and every N tile of 32 re-dequantises the same activation rows (40 tiles for gate/up), so the routed-row reuse it was meant to win is lost. Kept default off as evaluated/shelved. Remaining prefill levers: (1) multi-buffered chunk pipeline, (2) WMMA-based few-row tiles (16-row WMMA with activations shared across N in LDS), (3) prefill-load ownership rebalance.

2026-10-17 idea source - borrow hipBLASLt/Tensile tuning (radiance links no BLAS; all GEMMs are its own kernels): (1) per-shape solution table: a small set of tile variants per radiance GEMM (macro tile, wave tile, depth-U, split-K) benchmarked per (M,N,K) bucket on gfx1100 and selected at runtime from a table - the pattern of our llama.cpp hip-autotune record/tune/replay, applied to libr3/worker kernels; (2) split-K / stream-K for skinny GEMMs - Tensile's answer to few rows per expert, the case the FMA few-rows kernel lost on; (3) grouped GEMM with per-group tile choice (cold vs hot experts); (4) workgroup-to-tile mapping for Infinity Cache reuse; (5) prefetch/double-buffered LDS and direct-to-LDS loads. Starting points: the gfx1100 Tensile logic YAMLs shipped with hipBLASLt (/opt/rocm/lib/hipblaslt/library) record winning tile configs per shape - mine them for tile sizes near our shapes before writing a tuner.

2026-10-17 GPT dev review of the shelved native GDN chunk scan (req_e235aa9b0ebf4445, main 3cef62f): (1) likely defect - WMMA accumulator layout ownership: r3_gdn_wmma_native_compat adds into acc assuming gfx11 layout, while u/O/p are converted native<->gfx12 around one loop only; O[i] = r3_acc_gfx12_to_native(O[i]) before the final contraction is right only if every earlier producer left gfx12 layout - a silent permutation (no NaN). (2) R3_GDN_COMPARE prints only on mismatch/error - no 'active' line, no call count, so silence proves nothing; it also launches independently of R3_GDN_NATIVE, and copies initial ht only when h0 == ht (unsafe when distinct). Fix diagnostics first (one-time active line, call count/shape/dtype, matched/mismatched report, fail on zero invocations, exact state aliasing, confirm the loaded libr3.so). (3) Then convert layouts explicitly at each producer/consumer boundary and enable native WMMA one contraction at a time. (4) Offline served-replay test from captured 27B tp2 arguments: 64-token chunk zero state; 65/127/129 tokens; multi-chunk nonzero state; two calls carrying state; multiple sequences; non-identity ssm_state_indices; both state dtypes incl. in-place; tp2 head counts/strides - compatibility kernel as oracle, compare output+state after every call, then served KLD. (5) Must be re-diffed against radiance 1.7.0's GDN before any port. Low priority (scan ~4% of 27B prefill; worth 2-3%).

2026-10-17 radiance 1.7.0, helper ownership by heat (ab-benchmark tools/lab/radiance/helper-heat-ab.json, run rad06-heat-*, 2 balanced rounds, natives on, ROWS=0): heat-cold (production, helpers own coldest) pp4096 2,050/2,052, pp1024 2,105/2,102, tg512 126/126, tg2048 141/141; heat-hot (helpers own hottest: perf51/heat.rank0.hot-first = inverted, never-routed last) pp4096 2,134/2,140 (+4.2%), pp1024 2,144/2,143 (+1.9%), tg512 136/130, tg2048 143/140 - complete separation on prefill, decode not worse. Helper large-lane gate_up 2.52 -> 2.10 ms, down 1.41 -> 1.17 ms a request (fuller 16-row tiles). Next: confirm quality (helper-kld with hot-first), then make hot-first the production ownership; then the bigger levers - chunk pipeline (hide the helper wait) and a WMMA few-row/split-K helper GEMM.

2026-10-17 critical path on radiance 1.7.0 (hot-first ownership, helper-ab bench via ab-benchmark with server --profile-ops, run rad06-trace-*; rocprofv3 preload crashes radiance in mailbox_init, so radiance's own per-op device timing was used - it adds syncs, shares are approximate): R9700 op time over the whole bench (prefill + decode): moe_mailbox_add (helper wait + merge) 29.9%, gemm_nt_q 25.4%, hc_read 10.8%, local experts moe_gemm_q 8.7%, moe_mailbox_publish 4.2%, logits 3.1%, rest ~18%. ~780 helper requests a layer over the bench (one a layer every engine step), so the exposed wait is ~30%, not the ~10% the GPT review (req_3db4e687d2b7413d) estimated from one request a layer. Decision: P1 helper grouped/few-row GEMM first (3.3 of 3.9 ms a request; cuts the wait directly, no protocol change), then P2 overlap the R9700's local experts with the remote wait, then a two-slot pipeline if the wait stays large.

2026-10-17 grouped MoE GEMM form on the helper (R3_MOE_HELPER_MOE_FORM, helper-form-ab.json, run rad06-form-10172301, hot-first ownership): auto (libr4d's 32-row prefill form at ~790-row requests) pp4096 2,137/2,139, pp1024 2,149/2,149, tg512 130/130; forced 16-row decode form pp4096 1,708/1,709 (-20%), pp1024 1,722/1,713, decode flat. The '~4.5 rows an expert' estimate was from coldest-first ownership; with hot-first the helpers' runs are long and the 32-row form fits - per-layer form choice is not a lever. Setting kept experimental/off. Next: measure the real rows-per-expert distribution of helper requests (histogram in the helper's large-lane stats) before writing a custom grouped kernel; otherwise move to overlapping the R9700's local experts with the remote wait.

2026-10-18 helper request shape (run rad06-hist-*, hot-first): ~40 active experts a request, experts by rows 1: 9.0, 2: 4.0, 3-4: 4.5, 5-8: 5.0, 9-16: 4.0, 17-32: 3.6, 33-64: 3.3, 65+: 7.0 (coldest-first similar, 8.0 at 65+). gate_up 2.11 / down 1.18 ms a request = ~2.4 TFLOP/s on a ~120 TFLOP/s card, ~96 MiB weights (~0.1 ms at 960 GB/s): the grouped GEMM is neither compute- nor bandwidth-bound - kernel efficiency, not tile padding. Sent to GPT dev for diagnosis.

2026-10-18 down tile on gfx1100 (R3_MOE_HELPER_MOE_DOWN_TILE=1, helper-downtile-ab.json, run rad06-down-*): --compare (synthetic 790-row routing) down 1,443 -> ~954 us a call, but real serving only 1.18 -> 1.14 ms a request; end to end pp4096 2,137/2,134 -> 2,148/2,149 (+0.6%, complete separation), pp1024 +0.7%, decode flat. Lesson: --compare's synthetic routing is not representative - kernel work needs replayed real helper requests (capture a real request's activations + routes to a file, replay offline). Tile 2 pin refused for this shape. Next: adopt DOWN_TILE=1 in configs (small win), build request capture/replay for kernel tuning, then P2 overlap of the R9700's local experts with the remote wait.

## Change Log

- 2026-10-14T20:15:06.353093+00:00 (created-by): Created by agent
- 2026-10-14T20:41:34.505375+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261014_204134_flash-next-decode-on-dev-re_3947
- 2026-10-14T20:41:42.123589+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-14T21:08:32.689755+00:00 (updated-by): Updated: section:notes
- 2026-10-14T21:09:38.493597+00:00 (updated-by): Updated: section:notes
- 2026-10-14T22:19:13.458071+00:00 (updated-by): Updated: section:notes
- chg_20261014_221932_added-an-experimental-prefill_2342
- 2026-10-14T22:19:36.581927+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-16T23:24:29.586576+00:00 (updated-by): Updated: section:notes
- 2026-10-17T06:23:55.280354+00:00 (updated-by): Updated: section:notes
- 2026-10-17T11:07:24.668876+00:00 (updated-by): Updated: section:notes
- 2026-10-17T11:43:23.117713+00:00 (updated-by): Updated: section:notes
- 2026-10-17T12:24:32.312803+00:00 (updated-by): Updated: section:notes
- 2026-10-17T13:02:05.756354+00:00 (updated-by): Updated: section:notes
- 2026-10-17T14:54:03.505854+00:00 (updated-by): Updated: section:notes
- chg_20261018_015401_flash-next-prefill-on-the-r970_7463
- 2026-10-18T01:54:05.325170+00:00 (updated-by): Updated: section:ledger-events
- chg_20261018_025249_flash-next-helper-kernels-are_1169
- 2026-10-18T02:52:54.052954+00:00 (updated-by): Updated: section:ledger-events
- chg_20261018_040732_flash-next-prefill-on-the-r970_1864
- 2026-10-18T04:07:50.542590+00:00 (updated-by): Updated: section:ledger-events
- chg_20261018_071801_internal-experiment-balancing_7304
- 2026-10-18T07:18:08.819412+00:00 (updated-by): Updated: section:ledger-events
- chg_20261018_095640_flash-next-decode-is-about-5-8_1680
- 2026-10-18T09:56:47.405126+00:00 (updated-by): Updated: section:ledger-events
- chg_20261018_112016_flash-next-decode-is-about-1_8026
- 2026-10-18T11:20:24.648316+00:00 (updated-by): Updated: section:ledger-events
- chg_20261018_140648_the-faster-prefill-and-decode_5408
- 2026-10-18T14:06:52.749723+00:00 (updated-by): Updated: section:ledger-events
