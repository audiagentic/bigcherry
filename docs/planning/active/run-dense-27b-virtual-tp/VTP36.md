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

Each lever is one managed patch or one layout setting, selectable from the manifest, measured by matched ABBA against the current layout. Order by expected gain over cost; profile first.

1. **Profile the current build** (rad0023 timeline + `--profile-ops`): MXFP4 2048- and 8192-token one-prompt prefill and 8-stream decode, per rank, per layer kind, per collective phase (push, wait, fold). Decide which of 2-6 attacks the critical chain.
2. **All-reduce (from VTP31 steps 1, 4-7):**
   a. typed per-run method selector (`tuning.allreduce_method`; contract amendment), so each method below is an A/B;
   b. three-rank fused all_reduce + RMSNorm + fp8 quant (TP2 has it; removes TP3's extra pass);
   c. low-latency handshake for decode-size messages (flag-in-data, LL-style 8-byte units, fewer blocks);
   d. chained launches for the three-rank rows (the four-rank rows have r4d_chain.h) and a hierarchical exchange around the R9700's x4 link;
   e. CPU-assisted exact reduce (partials to pinned host, CPU sums in fp32 rank order, SDMA copies; HL05 / RAD50 relay pattern);
   f. push from the producing GEMM epilogue into host-mapped scratch (FLUX-style).
3. **Kernels:** RAD47 crossover on gfx1100 (tiled MXFP4 row 4-17x faster than nt_m64/decode rows at M 9-128: prefill tail chunks, multi-stream verify); RAD48 native int8 tile end to end (3.1x on one XTX at M=2048); RAD36 MXFP4 GEMV at M=2-8 (drafter verify). Measure each on the TP3 layouts, not only TP2.
4. **Split:** VTP32 solver from per-card, per-op-kind calibration; per-regime (prefill-weighted vs decode-weighted) layouts.
5. **Layer formats (dynamic per-card layer management):** per-card weight format by layer kind (XDP03 mixed quantisation from BF16), e.g. int8 on the gfx1100 ranks (native int8, emulated fp8) beside MXFP4/fp8 on the R9700, gated by KLD.
6. **Peer access:** re-check after any host/BIOS change (VTP35 step 1); if available, measure the exact one-shot over P2P.
7. Fold the winners into the layouts and refresh the baseline table (ABBA vs TP2, 3 repeats).

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

- MXFP4 TP3 one-stream TTFT within 5% of TP2 at 2048 tokens (carried from VTP35), or a recorded NO-GO stating the transport limit with the profile that shows it.
- TP3 aggregate tok/s at 4/6/8 streams stays at or above the baseline table for both formats.
- Every shipped lever is selectable per layout or run (no model or card constants), with its gain attributed by ABBA.

## Notes

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

## Change Log

- 2026-11-12T22:44:13.590288+00:00 (created-by): Created by agent
- 2026-11-12T22:44:40.981867+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
- 2026-11-12T23:31:05.048072+00:00 (updated-by): Updated: section:notes
