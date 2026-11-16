---
id: VTP36
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-12T22:44:13.590288+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# TP3 performance programme: prefill and decode on XTX + XTX + R9700 (kernels, all-reduce, split, layer formats)

## Description

The single owning item for performance of the native Radiance TP3 world: dense Qwen3.8-27B (MXFP4 and w8a8) on XTX + XTX (gfx1100, Gen4 x8) + R9700 (gfx1201, Gen4 x4), no GPU peer access, every all-reduce host-staged. Layouts `qwen27b-tp3`, `qwen27b-tp3-capacity` and `qwen27b-tp3-int8`.

It takes over VTP35 (MXFP4 prefill parity, superseded by this item) and the untried steps of VTP31 (completed; its acceptance was met). It links rather than duplicates the per-kernel and split items that keep their own acceptance: VTP32 (split solver), RAD47 (gfx1100 MXFP4 GEMM row selection at M 9-128), RAD48 (native gfx11 int8 prefill tile) and RAD36 (MXFP4 GEMV at 2-8 rows, drafter verify).

**Baseline (2026-11-12, VTP14 N7/N6 refresh, build 8a6f2776, DFlash2 depth 3, exact wire):**

| | 1 / 4 / 6 / 8 streams tok/s | prefill tok/s at 512 / 2048 / 8192 (one prompt) |
|---|---|---|
| MXFP4 TP2 | 83.1 / 199.9 / 194.3 / 224.2 | 1595 / 2384 / 2846 |
| MXFP4 TP3 | 115.6 / 222.7 / 214.2 / 239.3 | 1848 / 1919 / 2709 |
| w8a8 TP2 | 65.3 / 175.9 / 167.9 / 194.4 | 1438 / 1487 / 1903 |
| w8a8 TP3 | 83.6 / 188.0 / 179.2 / 213.1 | 1547 / 1615 / 2181 |

Where the time goes (VTP31): collectives are about two thirds of TP3 prefill device time (experiment 8) and half of 8-stream decode device time (experiment 9); a decode all-reduce costs 0.55-0.67 ms against TP2's 0.32 ms and is latency (flag round trips, slowest-rank wait), not bandwidth; the fused all-reduce + RMSNorm + fp8 quant exists only for world 2, so TP3 runs an extra pass; the R9700's extra collective share is mostly waiting for the XTX cards.

## Steps

Each lever is a managed patch or validated layout setting with measured ABBA and explicit quality baseline. **VTP36 owns the TP3 programme, not a duplicate host transport (VTP38).**

1. Finish the existing rad0031/rad0032/rad0033 GPU two-shot fused-pieces/write-once/duplex candidate, record build/split/tuning and phase timelines, complete 12-run int8/speculation stress and repeat 5,5,6 split 3x ABBA. Two already completed int8/MXFP4 ABBA comparisons with bit-identical logprobs are evidence, not final production approval.
2. Profile 512/2048/8192 one-prompt prefill and 1/4/6/8-stream decode with rad0023/rad0031 push/barrier/reduce/assemble phases, per rank and per layer kind; keep summed `--profile-ops` device events distinct from wall time. Revisit pipeline row floor where relevant.
3. Kernel owners: RAD47 gfx1100 MXFP4 crossover, RAD48 native gfx11 int8 (already on main), RAD36 low-row MXFP4 GEMV. Validate incremental complete-TP3 gain rather than double-count merged work.
4. VTP32 split solver: per-rank/per-op calibration and prefill versus decode regimes; 5,5,6 split validation remains independent of transport.
5. XDP03 mixed per-card/layer-kind formats through explicit KLD/quality; re-check P2P after justified host/BIOS changes only.
6. Give VTP38 the best validated GPU baseline, arithmetic and phase evidence. HL02 owns host-link measurement; **VTP38 alone** implements generic per-rank CPU-push/host-reduce collective and fallback. Do not implement the same production path twice.
7. Refresh accepted programme baseline with 3 final matched ABBA repeats versus proper TP2/TP3 exact-wire controls, full stress and VTP37 reordering gate.

8. **Long context (2026-11-14).** Add 30k / 60k / 100k one-prompt TTFT and the long-prompt-beside-4-streams arm at 100k to the workload for the `-long` configs, with decode chunks/s and gap p50/max during the prefill. Report TTFT as a linear and a quadratic term (two prompt lengths per arm).
9. **Wide steps for a lone prefill (settings only).** Arms: 2048; 4096; `checkpoint_interval 2048` + `max_num_batched_tokens 4096` + `prefill_span 2`; the same with `prefill_interleave 2`; 8192 with `prefill_span 2`. 3x ABBA for the chosen pair, the VTP37 reordering gate at the chosen chunk, int8 stress. The winner becomes a named variant of `qwen27b-tp3-int8-556-long` (RSC12).
10. **Report decode and prompt phase separately.** The multi-stream rows mix a 512-token prompt phase (~40% of the 8-stream round) with decode; record decode-only tok/s beside the round figure.
11. **R9700 after-boundary GEMM time.** Explain the ~0.65 ms the R9700's first GEMM after each all-reduce costs in the pipelined prefill (VTP37 `prof-base` against `prof-on`) before any split or transport change that relies on the R9700's prefill compute.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Matched ABBA (at least 2x, 3x for the final refresh) per lever: 1/4/6/8-stream tok/s with speculation on, one-prompt TTFT at 512/2048/8192, four 4096-token prompts, an 8192-token prompt beside 4 generating streams.
- Greedy identity against the exact wire for every exact method; KLD against BF16 for any lossy option or format change.
- `--profile-ops` / rad0023 phase times before and after.
- int8 stress (12 runs) for any change to the speculative or collective path (rad0028 regression guard).

## Effort & Risk



## Standards



## Acceptance Criteria

- MXFP4 TP3 one-stream TTFT within 5% of TP2 at 2048 tokens (carried from VTP35), or a recorded NO-GO stating the transport limit with the profile that shows it. *2026-11-14: +5.7% (775 against 733 ms, matched ABBA); not met.*
- TP3 aggregate tok/s at 4/6/8 streams stays at or above the baseline table for both formats.
- Every shipped lever is selectable per layout or run (no model or card constants), with its gain attributed by ABBA.
- Every reordering change passes the reordering quality gate defined in VTP37 (noise floor against serial; no quality loss against BF16).

## Notes

**2026-11-13 VTP36 candidate receipt (not final production):** rad0032 fused pieces + rad0033 write-once + duplex weights, explicit `all_reduce:nblocks=4,nthreads=256,ts_pieces=4,ts_fuse=1,ts_once=1,ts_duplex=1`. Two completed ABBA comparisons report bit-identical logprobs: int8 2048 TTFT 1054.3→797.5 ms (-24.4%), 8 streams 233.2→265.6 tok/s (+13.9%); MXFP4 2048 TTFT 1064.4→774.7 ms (-27.2%), 8 streams 237.8→277.4 tok/s (+16.7%). Stress and 5,5,6 split validation were still outstanding. Evidence with sizes and sample counts in `docs/evidence/2026-11-13-hl02-vtp38-consolidation/README.md`; implementation lives in the separate unmerged VTP36 performance branch at this checkpoint. VTP38 must compare CPU host reduce against this improved reference, not the older ~4 ms rad0031 baseline.

**History — earlier programme review and GPU timeline:**

Consolidated 2026-11-13 from a read-only GPT sweep (tp3-perf-gather) and a review of the plan tree. Superseded into this item: VTP35. Completed: VTP31 (acceptance met by experiment 10 and the VTP14 N7 refresh). Superseded elsewhere: RAD04 (TP2 prefill all-reduce overlap; upstream pipelines prefill at two ranks and rad0020/rad0025-0027 extend it). Linked, kept separate: VTP32, RAD47, RAD48, RAD36, HL05, XDP03. Not applicable to dense TP3: the RHP heat-placement items (MoE expert placement), Flash-Next helper and tail items, llama.cpp/ggml collective items (PGC/PHC).

Evidence: docs/planning/completed/run-dense-27b-virtual-tp/VTP14.md (N6/N7 refresh), VTP31.md (experiments 1-13), VTP33.md (capacity, rad0029); patches rad0020-rad0029; layouts engines/radiance/layouts/qwen27b.toml.


### GPT detailed review (2026-11-13, gateway req_main_47cb7e059896) and changes to this item

**Priority:** int8 TP3 decode and prefill now come first, in VTP37 (P0). This item keeps MXFP4 and the shared all-reduce work.

**Leading explanation of the 2048-token MXFP4 gap (to be confirmed by experiment 1):**
- The pipeline floors differ. TP2 uses the plugin default `kPipeMinRows` 1024 (512 tokens serial, 2048 pipelined); the TP3 layout sets 256 (both pipelined). TP2 gains the pipeline between 512 and 2048 and TP3 does not, which matches TP2 +49% vs TP3 +4% tok/s over that range.
- The x4 link floor: with shard weights 4,4,1 the R9700 moves 10S/9 = 22.2 MiB each way per 2048-row all-reduce, a ~2.96 ms floor at Gen4 x4 against ~1.33 ms for TP2 over x8; x ~129 all-reduces = ~210 ms, the size of the measured 208 ms gap (an order-of-magnitude agreement, not proof that it is all exposed).
- Not the cause: gfx1100 rows in M 9-128 (pieces are >= 256 rows, tiled regime) or a one-shot/two-shot switch (every piece is above the 524,288-element two-shot floor).

**Corrections:**
- Decode all-reduce cost is not mainly flag round trips: rad0023 showed ~2 us minimum flag wait, ~120 us R9700 fold and ~99 us waiting for XTX arrival (one-shot path, before rad0024). Arrival skew and data movement first.
- The R9700 owns a 1/9 shard (4,4,1), so moves 10S/9, not S.
- RAD47 (gfx1100 MXFP4 rows) is implemented and measured on main; it is not a new gain here. RAD48's native int8 tile is on main (#601).
- `--profile-ops` totals are summed device events, not wall time.
- The checked-in TP2 layout uses `tp-wire = wht6` and `--max-num-batched-tokens 2048`; ABBA receipts must record the exact-wire override.
- Exactness invariant: a transport change is byte-identical against the current TP3 exact method with the same split, not against TP2.
- rad0016-rad0029 SUMMARY status labels are stale relative to the hardware evidence (docs cleanup).

**Reordered steps:** (1) matched profile at 512/2048/8192 with the pipeline floor at 256, 1024 and off, per-step chunk, cut, kernel rows and collective phases; (2) phase timestamps in the weighted three-rank two-shot (`r4d_ar_twoshot_Nrank_exact.hip`, extending rad0023) and a narrow TP3 overlap lane for it; (3) one transport intervention chosen by (2). Downgraded: LL handshake (~0.5% decode), three-rank fused AR + norm for fp8 (~0.85% prefill device time), chained launches. CPU reduce, epilogue push and per-card formats only if the profile justifies them. If the transfer floor leaves under 165 ms removable at 2048 tokens, record the NO-GO.

Full review and brief: `docs/evidence/2026-11-13-vtp36-gpt-review/`.

2026-11-13 all-reduce timeline review (rad0031, baseline, int8 TP3; medians per all-reduce; columns push / wait1 / reduce / wait2 / assemble / total, us):

| Workload | XTX 0 | XTX 1 | R9700 |
|---|---|---|---|
| prefill 2048, ~1000 rows (~10 MB) | 386 / 1790 / 889 / 406 / 431 / 3995 | 399 / 1701 / 1301 / 2 / 453 / 3990 | 1334 / 20 / 188 / 1108 / 1254 / 3956 |
| decode 8 streams, 32 rows (320 KiB) | 14 / 5 / 23 / 17 / 12 / 81 | 14 / 4 / 39 / 2 / 14 / 82 | 2 / 161 / 5 / 37 / 42 / 251 |

Readings:
- Prefill is bound by data over the R9700's x4 link. With share w the R9700 always sends S and receives (1+w)S in any exact scheme, so 4,4,1 is already near its floor (shrinking w saves at most S/9). Measured R9700 push ~6.7 GB/s vs XTX ~14 GB/s. The phases run serially (~4 ms) against a duplex floor of ~1.4-2 ms: overlap (rad0032 pieces/duplex) is the lever, not the shares.
- A one-shot for prefill is worse (R9700 receives 2S); it stays right for decode (handshake-bound). Hierarchical exchange (XTX pair sums first, R9700 sends S and receives S) reaches the R9700 minimum; recorded as VTP38's fallback.
- XTX 1 is the last reducer in every prefill all-reduce (1301 vs 889 us for identical work) at the HL01 contended rate; ~400 us of barrier wait (~10%). HL01 found no PCIe configuration cause (2026-11-13 lspci, IOMMU passthrough).
- Decode: the R9700's 161 us at wait1 is idle time (it finishes its GEMMs first), not link time; give it more compute (split 5,5,6 candidate, +3-4% at 6/8 streams, needs 3x ABBA).

New cheap experiments for this item (in order, after rad0032/rad0033):
1. Uneven XTX reduce shares for prefill, e.g. two-shot weights 4.5,3.5,1 (layout configuration), and the mirror 3.5,4.5,1 to confirm the skew follows XTX 0's pulls. Pass: wait2 on XTX 0 and the R9700 < 100 us, prefill 2048 TTFT better beyond noise.
2. Split 5,5,6 3x ABBA for decode (held from VTP37).
3. Hierarchical exchange only if VTP38's host reduce fails its CPU-bandwidth gate.
The host-reduced all-reduce (CPU sums and pushes, no card pulls) is VTP38, including GPU-live DMA-BUF correctness, device-specific support, collective failure/abort protocol, and HL02's separately measured contention arms. Do not implement a second host reducer under this item.

**2026-11-14 long-context decomposition (bf16 cache, single runs; `docs/evidence/2026-11-14-tp3-long-context/README.md`).** One 100,366-token prompt: 53.8 s. TTFT fits a·N + b·N² with a = 3.49e-4 s/token (the all-reduce chain: 12,544 calls x 2.83 ms = 35.5 s) and b = 1.86e-9 (18.7 s at 100k). The quadratic term is attention on the XTX ranks: 13.5 ms a 1024-row call at ~23 TFLOP/s against the R9700's 9.0 ms for twice the heads; the XTX all-reduce lane idles 19.0 s behind it. `--max-num-batched-tokens 4096`: 44.7 s (-17%), a unchanged, b halved; 8k unchanged; beside 4 streams the long prompt 61.7 -> 51.3 s but decode gap max 1.55 -> 2.44 s. `--prefill-interleave 2`: decode during the prefill 5.0 -> 13.5 chunks/s for +8% long TTFT. `--busy-prefill-chunk 512`: gap max 0.68 s, long TTFT +46%. 4096 with interleave 2: 44.7 s alone; beside 4 streams 54.4 s, 11.1 chunks/s, gap max 2.41 s. `--tp-attn-split rows` is refused by the w8a8 plugin. Attention fill is RAD52; scheduling is VTP39. None of this is a default: no repeats yet.

**2026-11-14 XTX compute against the chain.** Outside attention an XTX rank computes ~8.9 ms a layer-step (GEMM ~82 TOPS at M~1024, 57% of the iu8 roof; delta-net scan 1.2-1.3 ms a call, ~10x the R9700 per head) against the chain's 11.3 ms. A transport gain past ~20% needs XTX compute to fall with it.

## Results (2026-11-13): the all-reduce

Worktree `perf/vtp36-tp3-performance`. Builds (all `radiance:stock:linux-radiance`, rad0016-rad0030 plus the patches
below): measurement build `5c6a3d0cb4f43b0fab7ae24c2861fef0` (rad0031 and a first rad0032), final build
`de5fae29d749178138275ba302404269` (rad0031, rad0032, rad0033). Logs: `${BC_TASK_TMP}/vtp36/logs/` (directory names below). World: the
layouts' settings (`qwen27b-tp3-int8` and `qwen27b-tp3`: exact wire, DFlash2 depth 3, fp8 KV, max-model-len 8192, 8
sequences, 2048-token chunks, 256-row pipeline floor, all-reduce grid 4 x 256), engines started directly with
`bin/radiance`, every arm a fresh start under the gpu lease.

### Claims checked before use

| claim | result | evidence |
|---|---|---|
| every TP3 prefill and multi-stream decode all-reduce is the weighted two-shot, weights 4,4,1 | **held** | start-up line "3-rank two-shot shard weights ... w4, w4, w1"; the rad0031 rings hold every message from 8 rows (80 KiB) up; only one-stream decode (4 rows, 40 KiB) stays on the one-shot (rad0023 rings) |
| the R9700 moves 10S/9 each way | **held** | its shard is 11% in every ring entry; its first push is 8/9 of the message |
| decode all-reduce costs 0.55-0.67 ms a call | **failed as a cost** | that is summed device time including the wait for peers. On the ranks that set the pace (the XTX) a 32-row call is 81 us; the R9700's 251 us is 161 us of waiting for them |
| ~2 us flag wait, ~120 us R9700 fold, ~99 us waiting for the XTX (one-shot) | **superseded** | the two-shot (rad0024) has no long R9700 fold: flag round trip 2.7 us, R9700 reduce 5 us, assemble 42 us, wait for the XTX 161 us (int8) or 82 us (MXFP4) |
| prefill: the Gen4 x4 floor is ~2.96 ms per 2048-row all-reduce | **failed** | 2.96 ms is one direction. The lockstep two-shot and the card's shader path carry one direction at a time, so the R9700 needs 2.6 ms of link time for a 1024-row message and 5.3 ms for 2048 rows, plus the waits between (table below) |
| ~129 x 2.96 ms = ~210 ms explains the 2048-token gap | **failed** | a pipelined 2048-token prefill is 258 all-reduces of ~1024 rows on the second lane, 3.7-4.1 ms each with 58-75 us between them: the all-reduce chain is the whole time to first token (258 x 4.05 ms = 1045 ms against 1044-1083 ms measured), not 210 ms of it |
| all-reduce is the largest share of device time | **held** (not re-profiled) | VTP37 `prof-on`; the timelines show why: on the second lane the ranks are inside an all-reduce 98% of a prefill step |
| lossy wire, grid sweeps, fused norm, chained launches are negatives | **not re-run** | nothing here contradicts them; a wider grid does not help the new forms either (`t4-*`) |

### Where a two-shot's time goes (rad0031, baseline)

Median microseconds per call, by rank (r0, r1 the XTX at Gen4 x8; r2 the R9700 at x4). `push` is the first hop,
`wait 1` until every peer's first flag, `reduce` the fold with the second hop, `wait 2` until every peer's second flag,
`assemble` the copy of the peers' reduced shards; `between` is the rank's time from the end of one call to the start of
the next (its compute); `last` is how often the rank was the last to arrive. Logs `tl-base`, `tl-base-serial`.

| workload (message) | rank | push | wait 1 | reduce | wait 2 | assemble | total | between | last |
|---|---|---|---|---|---|---|---|---|---|
| int8, 8 streams (32 rows, 320 KiB) | r0 | 14 | 5 | 23 | 17 | 12 | 81 | 304 | 41% |
| | r1 | 14 | 4 | 39 | 2 | 14 | 82 | 306 | 59% |
| | r2 | 2 | 161 | 5 | 37 | 42 | 251 | 125 | 0% |
| int8, 4 streams (16 rows) | r0 / r1 / r2 | 7 / 7 / 1 | 2 / 2 / 65 | 15 / 21 / 3 | 8 / 1 / 20 | 8 / 8 / 22 | 44 / 43 / 112 | 191 / 192 / 122 | 49 / 51 / 0% |
| int8, 64-token prompt, serial (68 rows) | r0 / r1 / r2 | 31 / 32 / 46 | 6 / 4 / 261 | 44 / 76 / 8 | 36 / 2 / 77 | 24 / 32 / 94 | 167 / 171 / 488 | 473 / 468 / 147 | 47 / 52 / 1% |
| int8, 512-token prompt, pipelined (202 rows) | r0 / r1 / r2 | 78 / 79 / 260 | 444 / 321 / 20 | 191 / 274 / 46 | 82 / 2 / 228 | 91 / 92 / 269 | 918 / 784 / 829 | 33 / 136 / 35 | 0 / 0 / 100% |
| int8, 2048-token prompt, pipelined (1024 rows, 10 MiB) | r0 | 396 | 1789 | 907 | 419 | 445 | 4037 | 58 | 0% |
| | r1 | 411 | 1711 | 1335 | 2 | 471 | 4060 | 58 | 0% |
| | r2 | 1375 | 21 | 187 | 1133 | 1261 | 4055 | 74 | 100% |
| int8, 4096-token prompt, serial (2048 rows, 20 MiB) | r0 | 783 | 829 | 1933 | 816 | 920 | 5319 | 4724 | 1% |
| | r1 | 797 | 649 | 2752 | 2 | 927 | 5228 | 4781 | 32% |
| | r2 | 2589 | 107 | 350 | 2415 | 2756 | 8286 | 1764 | 67% |
| MXFP4, 8 streams (32 rows) | r0 / r1 / r2 | 14 / 13 / 2 | 4 / 10 / 82 | 26 / 40 / 5 | 16 / 2 / 38 | 13 / 14 / 42 | 81 / 86 / 179 | 238 / 234 / 138 | 55 / 37 / 8% |
| MXFP4, 64-token prompt, serial (68 rows) | r0 / r1 / r2 | 30 / 31 / 46 | 5 / 3 / 124 | 45 / 47 / 8 | 8 / 5 / 45 | 27 / 27 / 105 | 146 / 143 / 322 | 342 / 339 / 167 | 45 / 53 / 2% |
| MXFP4, 512-token prompt, pipelined (202 rows) | r0 / r1 / r2 | 76 / 82 / 255 | 437 / 426 / 18 | 195 / 275 / 40 | 82 / 2 / 232 | 91 / 92 / 270 | 908 / 876 / 824 | 29 / 30 / 31 | 0 / 0 / 100% |
| MXFP4, 2048-token prompt, pipelined (1024 rows) | r0 / r1 / r2 | 399 / 419 / 1354 | 1736 / 1705 / 12 | 986 / 1390 / 206 | 405 / 2 / 1159 | 445 / 450 / 1374 | 4130 / 4128 / 4142 | 59 / 59 / 71 | 0 / 0 / 100% |
| MXFP4, 4096-token prompt, serial (2048 rows) | r0 / r1 / r2 | 779 / 793 / 2588 | 2685 / 2841 / 103 | 1999 / 2817 / 357 | 811 / 2 / 2462 | 924 / 926 / 2756 | 7469 / 7421 / 8264 | 3214 / 3047 / 2532 | 0 / 0 / 100% |

The flag round trip (the smallest `wait 1` of a call) is 1.5-2.7 us at decode sizes and 12-25 us at prefill sizes.
One-stream decode (4 rows, the one-shot): 15-17 us a call on the XTX, 85 us on the R9700, 70 us of it waiting.
Between server starts a 1024-row call measured 3.5, 3.7, 4.0 and 4.1 ms on the same build and settings; r1's reduce
moves most (830-1340 us), so single arms are only good to about 7%.

### Ranked causes

Milliseconds per call and rank, from the table above; the critical rank is the one that arrives last.

| cause | pipelined prefill, 1024 rows (int8; MXFP4 the same) | decode, 32 rows (int8) |
|---|---|---|
| 1. link bytes, one direction at a time | **R9700 2.64 of 4.06** (1.38 out, 1.26 in, at 6.9-7.6 GB/s); XTX 0.84-0.88 of 4.04 | XTX 0.026 of 0.081 (push, assemble); R9700 0.044 |
| 2. lockstep waits (a rank idle while a peer does its phase) | R9700 1.13 (the XTX's reduce); XTX 2.1-2.2 (the R9700's push, r1's reduce) | XTX 0.006-0.022; R9700 0.037 |
| 3. fold and host read | XTX 0.91-1.34 (r1 reads its slots 1.4-1.9x slower than r0 at every size); R9700 0.19 | XTX 0.023-0.039; R9700 0.005 |
| 4. arrival skew (compute imbalance) | none: every rank is inside the all-reduce chain | **R9700 0.161 of 0.251** (MXFP4 0.082); the XTX, which arrive last, pay nothing |
| 5. flag and fence latency | 0.02 a handshake, two a call: 1% | 0.003 a handshake: 6% of the XTX's 0.081, 1.4% of the 0.385 ms between two calls |
| 6. contention with compute | not visible: push and reduce run at the same rates beside the other piece's GEMMs as in a serial step | not visible |

- **Prefill is the all-reduce chain.** In a pipelined step the second lane is 98% all-reduce, and the R9700 is the last
  to finish its first push in 99-100% of calls. Each call costs the R9700 its link time twice over: its bytes go out,
  then it idles while the XTX reduce, then the result comes in.
- **Decode is not all-reduce bound.** A step is 129 calls of 0.385-0.409 ms (int8, 8 streams); the all-reduce is 81 us
  of each on the pace-setting ranks. The R9700 arrives 161 us early on int8 (82 us on MXFP4): its compute share is too
  small for decode. That is a split matter (VTP32), not a transport one.

### Can a card send and receive at once? (`docs/evidence/2026-11-13-vtp36-allreduce/`)

A standalone HIP program moves 32 MiB between a card and pinned mapped host memory (the mailbox's kind), by kernel
loads and stores at the all-reduce's grid (4 x 256) and by `hipMemcpyAsync`. GB/s, both directions summed:

| | R9700 (x4) | XTX (x8) |
|---|---|---|
| kernel out alone / kernel in alone | 7.0 / 6.7 | 14.2 / 12.9 |
| copy call out alone / in alone | 7.0 / 6.7 | 14.3 / 13.3 |
| kernel out + kernel in, two launches at once | 7.6 | 23.6 |
| one kernel, each pass a load and a store | 11.9 | 25.3 |
| kernel out + copy call in | 12.6 | 25.3 |
| copy call out + kernel in | 8.6 | 21.6 |

On the R9700 two loops in opposite directions do not overlap, whether they are two kernels or two phases of one; a
loop that alternates a load and a store does (1.7x). The numbers were the same before and after the two XTX traded
slots. This is why staggering blocks failed and why rad0032 fuses the first hop with the assemble.

### Levers built

| patch | axis (default) | what it does |
|---|---|---|
| rad0031 (diagnostic) | `R4D_AR_TIMELINE=DIR` | the two-shot's phase stamps, per launch and rank; reader `tools/lab/radiance/ar-ts-timeline.py` |
| rad0032 | `ts_pieces` (1), `ts_pieces_min` (0) | upstream's pieced two-shot at three ranks: a rank sends and assembles other pieces while its peers reduce |
| rad0032 | `ts_fuse` (0) | a step's first hop and assemble as one loop, so the link carries both directions |
| rad0032 | `ts_duplex` (0) | upstream's duplex weights: 64,64,0 here, the R9700 owns no shard and does no fold |
| rad0033 | `ts_once` (0) | through the host mailbox a rank writes its reduced shard once; peers assemble from the owner's scratch |

All are `--tune-pin` axes of the all-reduce rows (`tuning.tune_pin` in a layout), read only by the three-rank exact
two-shot; no model, card or world constant. Card-free test: `tools/tests/patch/test_vtp36_ts3_allreduce.py`.

Timeline of a 1024-row call (int8, 2048-token pipelined prefill), single arms, the R9700's total and the arm's time
to first token (logs `tl-p*`, `t3-*` to `t7-*`, `r1-*`):

| setting on top of 4 x 256 | R9700 per call | time to first token |
|---|---|---|
| layout pin | 3.76-4.06 ms | 994-1198 ms |
| `ts_pieces=4` | 3.31 ms | 897 ms |
| `ts_pieces=4`, odd blocks assemble first (removed) | 3.28 ms; at 8 x 256 3.66 ms | - |
| `ts_pieces=4,ts_once=1,ts_duplex=1` | 3.23 ms | 857 ms |
| `ts_pieces=4,ts_fuse=1` | 3.47 ms | 920 ms |
| `ts_pieces=4,ts_fuse=1,ts_once=1` | 3.11 ms | 835 ms |
| `ts_pieces=4,ts_fuse=1,ts_duplex=1` | 3.13 ms | 815 ms |
| **`ts_pieces=4,ts_fuse=1,ts_once=1,ts_duplex=1`** | **2.91-2.96 ms** | **770-797 ms** |
| the same with 8 / 16 pieces | 3.04 / 3.04 ms | 811 / 827 ms |
| MXFP4: layout pin -> the bold setting | 4.11-4.14 -> 2.85 ms | 1044-1064 -> 766-781 ms |

With the bold setting the XTX ranks have 256 us between two prefill calls on int8 (58 before): the second lane is no
longer saturated. A 202-row call (512-token prompt) goes 829 -> 550-618 us on the R9700; a serial 2048-row call
8.3 -> 7.2 ms (int8) and 8.3 -> 5.9 ms (MXFP4). Decode takes `ts_once` and `ts_duplex` only (a message under 51 rows
keeps one piece): the XTX call 81 -> 65 us, the mean cycle at 8 streams 402-409 -> 386-393 us (int8) and 378-380 ->
369-373 us (MXFP4).

### Validation of the bold setting (build `de5fae29d749178138275ba302404269`)

`all_reduce:nblocks=4,nthreads=256,ts_pieces=4,ts_fuse=1,ts_once=1,ts_duplex=1` (B) against the layouts' pin
`all_reduce:nblocks=4,nthreads=256` (A), one build, run after the two XTX cards traded slots (the timelines and the
microbenchmark were repeated then and agree with the earlier ones: logs `r1-*`, `duplex-4.txt`).

**Exactness** (logs `ident-int8-c1`, `ident-mxfp4-c1`), same split, same build:

- First-token top-20 log-probabilities at eight prompt lengths (306 to 6036 tokens): B bit-identical to A on w8a8 and
  on MXFP4; A against a second A run identical too.
- Greedy text, one stream, 96 tokens, at 300, 508, 1092, 2023, 2955, 4016 and 6027 tokens, and a 6090-token prompt
  beside four streams: identical in two B arms and two A arms, both formats.
- Four-stream greedy texts differ between any two runs, A against A included (w8a8: 3 of 4 and 0 of 4 the same between
  the two A arms; MXFP4: 1 of 4 and 0 of 4), as VTP37 recorded for the base build, so they say nothing about B.

**2x ABBA** (order A B B A A B B A, logs `abba-int8-c1`, `abba-mxfp4-c1`; mean ± standard deviation; every server
alive at the end of its arm):

| w8a8 (`qwen27b-tp3-int8`) | A | B | change |
|---|---|---|---|
| 1 stream, tok/s (8 rounds) | 88.4 ± 0.5 | 89.2 ± 0.1 | +1.0% |
| 4 streams | 203.4 ± 5.0 | 218.9 ± 5.7 | +7.6% |
| 6 streams | 194.7 ± 3.0 | 216.0 ± 3.2 | +10.9% |
| 8 streams | 233.2 ± 1.4 | 265.6 ± 1.8 | +13.9% |
| one prompt, 522 tokens, time to first token (8) | 283 ± 5 ms | 246 ± 16 ms | -13.0% |
| one prompt, 2021 tokens (8) | 1054 ± 3 ms | 798 ± 5 ms | -24.4% |
| one prompt, 6059 tokens (8) | 2963 ± 18 ms | 2323 ± 18 ms | -21.6% |
| four 4096-token prompts, slowest time to first token (4) | 8610 ± 38 ms | 7404 ± 7 ms | -14.0% |
| 6144-token prompt beside 4 generating streams (4) | 3415 ± 1 ms | 2685 ± 7 ms | -21.4% |

| MXFP4 (`qwen27b-tp3`) | A | B | change |
|---|---|---|---|
| 1 stream, tok/s (8 rounds) | 116.8 ± 0.3 | 122.4 ± 0.3 | +4.8% |
| 4 streams | 222.4 ± 4.1 | 252.7 ± 6.4 | +13.6% |
| 6 streams | 216.6 ± 2.8 | 245.9 ± 1.5 | +13.5% |
| 8 streams | 237.8 ± 1.7 | 277.4 ± 2.4 | +16.7% |
| one prompt, 522 tokens, time to first token (8) | 286 ± 3 ms | 225 ± 6 ms | -21.2% |
| one prompt, 2021 tokens (8) | 1064 ± 4 ms | 775 ± 4 ms | -27.2% |
| one prompt, 6059 tokens (8) | 2959 ± 14 ms | 2217 ± 5 ms | -25.1% |
| four 4096-token prompts, slowest time to first token (4) | 8800 ± 16 ms | 6781 ± 49 ms | -22.9% |
| 6144-token prompt beside 4 generating streams (4) | 3433 ± 7 ms | 2572 ± 8 ms | -25.1% |

One-prompt prefill is now 2320 / 2609 / 2734 tok/s (MXFP4) and 2122 / 2534 / 2609 tok/s (w8a8) at 522 / 2021 / 6059
tokens. The recorded MXFP4 TP2 time to first token at 2048 tokens is 859 ms (VTP14 N7, not re-run here): TP3 at
775 ms is under it. The stream rows gain because each round prefills its 512-token prompts and runs mixed steps; the
decode step itself gains about 4% (timeline cycle above).

**Stress** (logs `stress-int8-c1`): rad0028's guard pattern on w8a8 with B, twelve fresh starts: a 4/8-stream bench of
512-token prompts, then 4 streams generating 1500 tokens with a 6144-token prompt beside them.

| build | pin | runs | faulted | benches failed | mean 4 / 8 streams | long prompt's text |
|---|---|---|---|---|---|---|
| de5fae29 | B | 12 | 0 | 0 | 224.4 / 266.0 tok/s | the same in all 12 |

**Launcher smoke** (logs `layout-smoke-*`): `launch-flashnext-best.sh start --layout` on the final build with the pin in
the layout starts both layouts, passes the canary and serves a 2021-token prompt (772 ms w8a8, 722 ms MXFP4 to the
first token); stopped with the launcher's `stop`, power settings not applied.

### The int8 split on top of B (logs `abba-int8-split`)

Skew absorption by compute share: B on both arms, the layout's split (delta-net 6,6,4, feed-forward 6656/6656/4096)
against `kv=1,1,2;ffn=5632,5632,6144;lk=5,5,6;lv=15,15,18`, 2x ABBA. In the timeline the second split takes the R9700's
wait for the XTX from 146 to 69 us a call and the 8-stream cycle from 409 to 370 us.

| w8a8 | 6,6,4 | 5,5,6 | change |
|---|---|---|---|
| 1 / 4 / 6 / 8 streams, tok/s | 89.7 / 224.6 / 217.4 / 266.8 | 92.0 / 229.8 / 243.6 / 281.2 | +2.6 / +2.3 / +12.0 / +5.4% |
| one prompt, 522 / 2021 / 6059 tokens, time to first token | 243 / 755 / 2309 ms | 237 / 754 / 2209 ms | -2.8 / -0.1 / -4.3% |
| four 4096-token prompts | 7293 ms | 6905 ms | -5.3% |
| 6144-token prompt beside 4 streams | 2618 ms | 2502 ms | -4.4% |

The second split also lets the KV cache grow to ~228,000 tokens a card against ~56,000 (startup receipt).

**Quality gate (2026-11-14, same build, 42,761 positions).** Scored against the 6,6,4 serial prefill, every split change
lands at the floor's scale (feed-forward 5,5,6 alone 0.00214 / p99.9 0.078 / 97.65%; delta-net 5,5,6 alone 0.00218 /
0.092 / 97.68%; feed-forward 6144/6144/5120 0.00260 / 0.070 / 97.66%; 5,5,6 0.00248 / 0.076 / 97.72%, the same in a repeat;
floor variants: serial chunk 1024 0.00197 / 0.059 / 97.81%, chunk 512 0.00197 / 0.068 / 97.85%, pipelined chunk 1024
0.00225 / 0.061 / 98.13%). A different partition is a different, equally exact rounding, not a nearer or further one,
and no partial split does better. Scored as the gate is written, against a serial reference of the same layout:

| 5,5,6, against its own serial prefill | KLD mean | p99.9 | top-1 | PPL ratio |
|---|---|---|---|---|
| serial again | 0 | 0 | 100% | 1.0000 |
| serial, 1024-token chunks (the floor) | 0.00324 | 0.051 | 97.86% | 1.0004 |
| pipelined (the layout) | **0.00054** | **0.022** | **99.34%** | 1.0005 |

Against a BF16 reference recorded on this build (TP3, serial), 5,5,6 is the nearer of the two:

| against BF16 | KLD mean | p99.9 | top-1 | PPL ratio |
|---|---|---|---|---|
| 6,6,4 pipelined (`qwen27b-tp3-int8`) | 0.00334 | 0.086 | 96.86% | 0.9996 |
| 5,5,6 pipelined | 0.00251 | 0.054 | 97.00% | 1.0003 |

Neither meets VTP14 N6's recorded values (0.00199 / 0.043 / 97.32%): that reference was not kept, and on this one most
of both arms' mean is one document (`kld-docs-22`, positions ~1270-1600, KLD up to 19), which moves under every change,
the serial chunk change included (without it: 0.00136 / 0.049 / 96.90% and 0.00143 / 0.042 / 97.03%). The BF16
values for gate 2 should be re-recorded on a kept reference.

Kept as an opt-in setting on `qwen27b-tp3-int8`, not a layout of its own: `--set tuning.tp_ownership='"kv=1,1,2;ffn=5632,5632,6144;lk=5,5,6;lv=15,15,18"'`
(launcher `--set`, or bc-serve). The layout's own split stays the default until the owner chooses.
The first launcher start with it was refused (BC_CFG_PLAN_MISMATCH): bc-serve's measured-plan cache key left out
every `tuning.*` value, so the plan measured for the default split was reused. The key now includes the tuning section
(serving `plan_key_inputs`); with it the start measures its own plan and the engine runs the 5,5,6 split (smoke 2026-11-14).

### Copy-engine assemble for the narrow-link rank, in isolation (2026-11-14, `dmalane.hip`, `dmalane.txt`)

The R9700's step under the duplex weights (its whole partial out, the whole result in, in pieces), 40 iterations, every
word checked: A the fused kernel (rad0032), B a kernel that sends the pieces and raises a flag per piece while a second
stream waits on each flag (`hipStreamWaitValue32`) and copies that piece in with the copy engine (the engine change the
lead was offered), C a kernel out beside one copy in with no dependency (the link's ceiling).

| R9700, 4 pieces | 1 MiB | 4 MiB | 10 MiB |
|---|---|---|---|
| A fused (now) | 330 us | 946 us | 1948 us |
| B kernel out + flag-gated copy engine in | 335 us | 1129 us | 2579 us (+32%) |
| C free duplex ceiling | 200 us | 757 us | 1773 us (-9%) |

Gated per piece, the copy engine is slower than the fused loop (8 pieces: the same); with no gating at all it would win
9% at 10 MiB. On an XTX B only ties A at 10 MiB. **NO-GO**: the engine change is not worth building.

### TP3 against TP2 (2026-11-14, `tp2-vs-tp3-abba.md`)

MXFP4, `qwen27b-2xtx` (A) against `qwen27b-tp3` (B), both started by the launcher on build
`de5fae29d749178138275ba302404269` without the cards' power settings, 2x ABBA, n=8:

| | TP2 | TP3 | change |
|---|---|---|---|
| 1 / 4 / 6 / 8 streams, tok/s | 86.5 / 220.2 / 218.2 / 184.9 | 122.4 / 256.4 / 246.1 / 277.4 | +41.5 / +16.4 / +12.8 / +50.1% |
| one prompt, 522 / 2021 / 6059 tokens, time to first token | 251 / 733 / 2413 ms | 225 / 775 / 2217 ms | -10.5 / +5.7 / -8.1% |
| four 4096-token prompts | 6275 ms | 6810 ms | +8.5% |
| 6144-token prompt beside 4 streams | 2716 ms | 2551 ms | -6.1% |

TP2's 8-stream row queues two streams (`--max-num-seqs 6`). The first acceptance criterion misses by 0.7 points: TTFT at
2048 is 5.7% over TP2, whose own time on this build is 733 ms (859 ms recorded earlier). At 512 and 6144 tokens and
beside streams TP3 is ahead; with four long prompts at once it is 8.5% behind.

### NO-GOs recorded

- **Staggered blocks** (odd blocks assemble before they push): 3.28 ms against 3.31 ms unstaggered at 4 x 256, worse at
  8 x 256 (3.66 against 3.30 ms). Separate loops do not overlap on the R9700's link. Removed from rad0032.
- **A wider grid for the pieced launch** (8 x 256, 4 x 512, 8 x 512, 16 x 256 with four pieces): 3.25-3.67 ms, no better
  than 4 x 256.
- **More pieces**: 8 and 16 pieces measure 3.04 ms against 2.91-2.96 ms at four.
- **`ts_duplex` alone, lockstep**: 4.31 ms against 4.06 ms; it pays only with pieces (the R9700 then has no fold to
  hold a step up).
- **A piece floor above the instance's**: with pieces off under 2,097,152 elements a 202-row call is 835 us against
  650 us in pieces.
- Techniques 1, 4, 6 and 7 above.

### Decisions

- Taken: `qwen27b-tp3` and `qwen27b-tp3-int8` set B in `tuning.tune_pin` (a commit of its own, to drop if the layouts
  should wait). `qwen27b-tp3-capacity` keeps the grid alone: not measured.
- For the lead:
  1. the int8 split 5,5,6 (table above): passes the quality gate against its own serial prefill and is nearer BF16;
     opt-in by `--set tuning.tp_ownership=...` on `qwen27b-tp3-int8`, default pending the owner;
  2. the copy-engine assemble for the rank behind the narrow link (technique 3): NO-GO in isolation (2026-11-14, above);
  3. r1's slow host read stays with the slot at root port 00:01.0 across the card swap (HL01): r1's reduce is the
     longest XTX phase in every pieced call, so whatever lifts it shortens the calls further;
  4. acceptance: the matched TP2 arm (2026-11-14, above) puts TP3 5.7% behind TP2 at 2048 tokens;
  5. rad0021's README still says the x4 card owns no shard under 4,4,1 (it owns 1/9); `ts_duplex` is what makes that true.

### Not done

- KLD: not needed for B (bit-identical to the layout pin at the same split).
- An 8192-token one-prompt arm (max-model-len 8192; 6144 used), a matched TP2 ABBA, write-combined host buffers,
  the capacity layout.

### External techniques against the measurements

| technique | verdict | why |
|---|---|---|
| 1. Lamport / flagless handshake | NO-GO | the flag round trip is 2.7 us of an 81 us decode call (two a call: 1.4% of the 385 us between calls) and 20 us of a 4 ms prefill call |
| 2. reduce-scatter + all-gather split around the norm | not built | the two-shot already is a reduce-scatter and an all-gather. What was missing is their overlap, which pieces and the fused step give inside the call; moving the norm between them would save 1-2% of device time (VTP37) and needs the plugin to carry shards across blocks |
| 3. copy engines for the mailbox | **next candidate** | on the R9700 a copy-engine receive beside a kernel send reaches 12.6 GB/s; the fused loop reaches 11.9 in isolation but about 8.4 in a 2.5 MiB piece. The engine's DMA lane (`core/device/aql.cpp`: own SDMA queue, POLL_REGMEM) could assemble the R9700's result while its kernel sends. Write-combined against cached host buffers was not tested |
| 4. CPU-assisted reduce | NO-GO | fold and host read are not the bound on the critical rank (R9700 fold 0.19 of 4.06 ms, none with the duplex weights), and the R9700 would still move the whole message each way |
| 5. skew absorption | partly built, partly a layout matter | `ts_duplex` (4,4,0 for 4,4,1) removes the R9700's fold. The decode skew is compute: with lk 5,5,6 and ffn 5632,5632,6144 the int8 8-stream cycle falls 409 -> 370 us in one arm (the MXFP4 split: 392 us). VTP32 / VTP37 |
| 6. quantized exact-wire alternatives | NO-GO for now | bytes on the x4 link bind only in prefill, and there overlap was the larger lever; upstream's lossless coded two-shot (12.2 bits a value, four ranks only) is the exact option if more is needed |
| 7. token-split overlap in decode | NO-GO | the all-reduce is 81 us of a 0.39-0.41 ms decode cycle and two half-batches double the calls |

### Stop rule

Not reached before building: the calls cost 2.7 times the R9700's duplex link time. After the patches a 1024-row call
is 2.9 ms. The R9700 moves the message once each way (10 MiB each, `ts_duplex` and `ts_once`), which is 2.9-3.0 ms at
7 GB/s one direction at a time and about 1.7 ms at the fused loop's isolated rate. What remains is therefore the x4
link, to the extent the card overlaps directions: the fused loop gets about a third of the possible overlap in situ
(half the bytes are in fused steps, at 8.4 GB/s summed).

## Change Log

- 2026-11-12T22:44:13.590288+00:00 (created-by): Created by agent
- 2026-11-12T22:44:40.981867+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
- 2026-11-12T23:31:05.048072+00:00 (updated-by): Updated: section:notes
- 2026-11-13T05:15:40.560777+00:00 (updated-by): Updated: section:notes
- 2026-11-14T06:59:22.719204+00:00 (updated-by): Updated: section:steps, section:notes
