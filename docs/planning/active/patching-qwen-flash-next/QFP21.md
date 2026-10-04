---
id: QFP21
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-04T21:02:03.440345+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Re-evaluate neutral/parked Flash-Next patches (1293 1295 1301 1305 1306 1314) per GPT review

## Description

GPT reviewer req_2359b5a495574b94 (2026-10-05) on the six evaluated-but-not-promoted patches: 1295 re-test first (parking reason stale: R9700 holds no KV under v6 ATTN_TS=1,1,0); 1301 re-test on the right model (27B Q8_0 dual-XTX MTP4 - Flash-Next has little Q8_0); 1306 re-test as a VRAM enabler (never measured for headroom); 1305 rework (routed-expert-only split from measured per-GPU expert throughput, skew-instrumented, 48K first); 1314 rework (one-wave 64-256 thread fused AR, current 1024-thread spin is the cost); 1293 close (real fix is meta-backend events / multi-buffered inputs).

## Steps

1. 1295 on v6 240K f16 ub512 MTP3: QSA_GATHER 0/1 ABBA at ~48K, ~120K, ~200-230K + no-MTP long ctx; sweep QSA_GATHER_MIN 16K/32K/64K; record ms/round, t/s, peak VRAM. If positive: rework to fused F16 gather + compact mask.
2. 1301 on Qwen3.8-27B Q8_0 dual-XTX, -sm tensor, MTP4: BIGCHERRY_Q8_F32_MAXCOLS 1 vs 5, ABBA 8K + 48K, >=256 tokens; per-ncols hits, quantize_q8_1 count/time; optional MTP1/2/4 width sweep. RDNA2 (6900 draft) needs a new gate.
3. 1306 enabler matrix: SHEXP_SPLIT 0/1 per-device VRAM + skew + decode; then +1295; then ub1024 headroom.
4. 1305 rework: separate routed-expert TS from dense/shared FFN TS; derive split from measured throughput; 48K skew diagnostic then 240K.
5. 1314 rework to low-occupancy fused AR; sweep 64/128/256/1024 threads.
6. Close 1293 (record as superseded by a future meta-event design).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Bug fixes/reworks go inside each patch's own package (owner rule 2026-10-05).

Owner 2026-10-05: untested patches with potential are worth pursuing. 1286 (DFlash/DSpark draft local copies of tensor-split target tensors) WAS hardware-tested (dflash-27b-screen-1286, Qwen3.8-27B dual-XTX, 1.6K prompt): without it every DFlash/DSpark run with an -sm tensor target fails to start; with it DFlash2-Q8 n7 drafting on the R9700 76.6 t/s (45.5% acc) vs built-in MTP5 77.2 (57.8%), on the 6900 72.7; DSpark n6 6900 64.6; prefill 1084-1103 vs MTP5 1294. Add: (a) 1286 offline mechanics test + evidence record -> promote as an enabler; (b) DFlash tuning on the 27B: n_max 4-10 x placement (R9700 / 6900) x Q4_K_M/Q8, recover acceptance (n4 R9700 gave 59.3% / 75.4 t/s), find the prefill loss; (c) 1300 Q8_0/D=256 native vector FA decode: one screen at a q8_0 KV fallback tier (contexts beyond the f16 fit).

2026-10-05 results (queue-qfp21, promoted base + patch). 1295 gather on v6 240K f16 MTP3 (A = GATHER=0, B = 1): ~48K tokens neutral (88.2 vs 88.6/89.5 t/s, 41.5 vs 41.9/41.5 ms/step, greedy identical); ~120K 71.2 vs 63.6/64.5 t/s, 46.1 vs 49.7/49.0 ms/step (-6%); ~215K 60.2 vs 47.7/48.1 t/s, 57.5 vs 61.0/60.5 ms/step (-5%); peak VRAM identical in all arms (22.8/23.2/31.2/4.1 GiB) - the R9700 OOM parking reason is gone; greedy differs at 120K/215K (gather was closer to f32 in the original test - confirm vs f32 before promotion); t/s gains include higher MTP acceptance on the different text, so quote ms/step. -> promotion candidate (needs default/threshold decision + f32 check). 1301 on Qwen3.8-27B Q8_0 dual-XTX MTP4 (A = MAXCOLS 1, B = 5): 10K 73.4/73.0 -> 64.8/64.7 t/s (ms/step 44.1/44.4 -> 49.4/49.5), 32K 72.1/72.0 -> 67.9/67.8 (47.4 -> 52.4/52.5): widening to 5 regresses 6-12%; per-ncols activation confirmed (trace logs once per width, not overhead). Width 2/3 sweep queued (queue-1301-widths).

## Change Log

- 2026-10-04T21:02:03.440345+00:00 (created-by): Created by agent
- 2026-10-04T21:05:37.061800+00:00 (updated-by): Updated: section:notes
- 2026-10-04T21:52:02.541657+00:00 (updated-by): Updated: section:notes
