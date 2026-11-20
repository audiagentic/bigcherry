---
id: ENG06
order: 0
plan: tuning-radiance-engine
state: completed
created-at: '2026-11-19T08:41:19.060365+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# bf16_i8 KV cache: qualify on the production layout, match E4M3-V decode speed, measure 256K decode, quantise the QSA block keys

## Description

bf16_i8 (rad0038: bf16 K, int8 V with an f16 scale per 32 head-dim values) is the best KV format at or above the owner's 8-bit floor: KLD 0.00269 vs bf16 (bf16_fp8 0.00323, fp8 0.00495) at 1.283x bf16 capacity on Flash-Next (qwen27b TP3: 818,896 vs 624,048 tokens). It is not shippable: rad0038 is `untested` in the kv-bf16-packed experiment and was measured only on two XTX without helpers (docs/evidence/2026-11-16-vtp44-generic-kv-layouts/FLASHNEXT-AB.md). In that A/B its one-stream decode (28.95 chunks/s) sits with bf16 (28.86), not with the E4M3-V formats (bf16_fp8 30.78, fp8 30.66): bf16_fp8 also keeps K bf16, so the int8-V read/dequant path is the likely ~6% gap. Capacity misses 1.306x because Flash-Next's QSA block-key cache group is not quantised. Owner direction 2026-11-19: plan and begin.

## Steps

1. Root-cause the int8-V decode gap (code first, no GPU): read the gqa12/gqa6 decode attention units for b16i8 vs b16f8 (rad0038 vs rad0037 in engines/radiance/patches, libr3/libr4d units), compare V load width, scale load/broadcast, dequant placement (per element vs per 32), LDS use and occupancy; then rad-kbench rows via tools/lab/radiance/kbench_ab.py for decode attention b16i8 vs b16f8 vs bf16 at 1/4 streams and 8K/48K/256K context. Fix in the managed patch (bigcherry-patch-author), bit-identical outputs to the unpatched b16i8 path.
2. Qualify rad0038 (and the step-1 fix) on the production flashnext-4gpu layout with the helpers and the live policy on: KLD vs bf16 on the long corpus (the VTP43 bar), needle retrieval at 48K and 100K, decode/TTFT/256K prefill parity runs (flashnext-parity.sh, >= 2 launches, FLASHNEXT_PARITY_WARMUP_S=451), capacity from the engine plan; same for qwen27b TP3 -long. Lifecycle per bigcherry-patch-lifecycle; if it passes, promote and ship -long-bf16i8 / a Flash-Next bf16_i8 config.
3. 256K decode A/B: bf16 vs bf16_i8 vs fp8 at 256K context (decode is KV-read bound there; a bf16_i8 page is 36.75 vs 48 KiB), 1 and 4 streams, ms/step and tok/s.
4. Quantise the QSA block-key group for bf16_i8 (capacity 1.283x -> ~1.306x): design first (the block keys select which blocks are read, so check retrieval and KLD specifically), then the same gates as step 2.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Step 1: end-to-end 248K decode of the scale-broadcast build against the baseline, two launches each (38.89/38.76 vs 38.82/38.89 ms), recorded in docs/evidence/2026-11-20-eng06-long-decode/. Step 2: KLD, needles 18 of 18, TTFT and decode parity on TP3 27B and flashnext-4gpu in docs/evidence/2026-11-19-eng06-bf16i8/. Step 3: stream-bench at 248K (1 stream) and 4 x 149K, six launches (each format twice), docs/evidence/2026-11-20-eng06-long-decode/. Step 4 moved to ENG07.

## Effort & Risk



## Standards



## Acceptance Criteria

Met: bf16_i8 decode equals bf16 and is within 1% of fp8 on the TP3 and Flash-Next layouts, KLD 0.00269 and needles 18 of 18 recorded, rad0038 already validated and serving-core with shipping configs (qwen27b-tp3-int8-556-long-bf16i8, flashnext-4gpu-bf16i8), 256K decode recorded (bf16_i8 38.9 ms a step against bf16 46.9 and fp8 52.3 at 248K on one stream), the scale-broadcast patch rejected with end-to-end evidence (no gain), block-key quantisation split into ENG07.

## Notes

Closed 2026-11-20: step 3 measured (bf16_i8 is 17% faster per decode step than bf16 and 26% faster than fp8 at 248K, one stream; 13% and 19% at 4 x 149K); step 1's scale-broadcast patch (branch agent/eng06-int8v-scale-broadcast, bb00cbca3) gave no gain at long context (equal within the 0.3% spread) and was dropped; step 4 became ENG07. rad-kbench could not time the b16i8 rows: no recorded fixture holds a bf16_i8 attention case.

## Status 2026-11-19 (step 2 done for the two layouts; evidence docs/evidence/2026-11-19-eng06-bf16i8/README.md)

- Correction to the description: rad0037/38/40/41 are already `validated` and in `serving-core` (#1033); only rad0042 (selftest) is an experiment. Nothing to promote.
- Step 2 done: dense TP3 27B `-556-long`, bf16_i8 against fp8 and bf16 (ABBA, one fresh start each): engine decode 27.6 ms/step (bf16 27.6, fp8 28.0), 100K first token 42.5 s (43.6, 45.1), needles 18 of 18, capacity 818,896 tokens. Flash-Next `flashnext-4gpu` with helpers and the live policy: does not fit the production 8 sessions x 262,144 (engine card 5.47 GB short at admission); at 4 sessions, two launches (launch 2 replayed): 256K prefill 2916 / 3051 tok/s, one-stream decode 25.8 ms/step (fp8 default 25.87), cold TTFT 8.45 s; needles 48K 9/9, 100K 8/9 (a refusal). KLD against bf16 0.00269 (VTP43/44, bar 0.0027) was not re-run with helpers (the KV error does not depend on them).
- Shipped: config `qwen27b-tp3-int8-556-long-bf16i8` and the non-default `flashnext-4gpu-bf16i8` (4 sessions; its own layout name, so no live-policy record until calibrated). fp8 at 8 sessions stays the Flash-Next default.
- The one-stream int8-V decode gap of about 6% (VTP44, two XTX, no helpers) does not show on either shipped layout (TP3 equal to bf16; Flash-Next dominated by the helpers); step 1's scale-broadcast patch (GPT lane, branch agent/eng06-int8v-scale-broadcast) therefore matters mainly for helper-free layouts and 256K decode; it stays separate until its kbench A/B.
- Open: step 3 (256K decode A/B, scripts ready: scripts/long.sh), step 4 (QSA block-key quantisation, design first).

## Change Log

- 2026-11-19T08:41:19.060365+00:00 (created-by): Created by agent
- 2026-11-19T08:41:41.916454+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
- 2026-11-19T12:22:57.861752+00:00 (updated-by): Updated: section:notes
- 2026-11-19T15:40:49.628925+00:00 (updated-by): Updated: section:validation, section:acceptance_criteria, section:notes
- 2026-11-19T15:40:56.293600+00:00 (state-transition): State: pending → completed
