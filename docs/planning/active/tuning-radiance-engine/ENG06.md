---
id: ENG06
order: 0
plan: tuning-radiance-engine
state: pending
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

Step 1: kbench rows per format and context, before/after the fix, outputs bit-identical. Steps 2-4: KLD (long corpus, VTP43 harness), needle probes, flashnext-parity.sh compare vs docs/reference/engines/radiance/baselines/flashnext-4gpu-parity.json and the current-main baseline (>= 2 launches), qwen27b TP3 TTFT/decode vs its bf16 config; capacity read from the engine plan.

## Effort & Risk



## Standards



## Acceptance Criteria

bf16_i8 one-stream decode within 1% of bf16_fp8 on the same layout (or the remaining gap explained), KLD and needle at or better than the VTP43/VTP44 numbers, no prefill/TTFT regression vs bf16, rad0038 promoted with a shipping config; 256K decode result recorded; block-key quantisation shipped or rejected with evidence.

## Notes



## Change Log

- 2026-11-19T08:41:19.060365+00:00 (created-by): Created by agent
- 2026-11-19T08:41:41.916454+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
