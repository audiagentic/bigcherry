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

## Change Log

- 2026-10-04T21:02:03.440345+00:00 (created-by): Created by agent
