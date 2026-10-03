---
id: QFP07
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:20:52.461503+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# 1303 separate attention/KV tensor split (BIGCHERRY_ATTN_TS / _ROTATE)

## Description

Patch 1303_attn_kv_tensor_split (in production profile v2): under -sm tensor, BIGCHERRY_ATTN_TS gives the full-attention family (q/k/v/qkv weights+biases, q/k norms, sinks, gate, attn_output, KV cache) its own split vector; BIGCHERRY_ATTN_ROTATE=0 disables per-layer rotation so whole KV heads pin to chosen GPUs. Recurrent (GDN) layers reusing attn_qkv/attn_gate keep -ts (anchored to ssm_out); indexer tensors/caches stay mirrored. Design GPT req_0f390645741c4413.

## Steps

1. Fold review. 2. Investigate 248K collapse (QFN04). 3. Try attention split variants with fractional R9700 share for load balance. 4. Promotion evidence (greedy/KLD vs default split).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Greedy identical vs default split at matched KV type; ABBA; deep fill to max ctx.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Background: with 2 KV heads and 3 GPUs the default split rotates heads so each GPU holds one head for 8 of 12 QSA layers, coupled to expert-weight placement. Results (8K-deep searches): KV on both XTX (1,1,0): f16/q8_0 and f16/f16 load to 248K, usable 240K (248K prefill collapses ~1100 -> 85 t/s; 256K OOM); 1,0,1 to 208K+; 0,1,1 fails <144K. Before: f16/f16 96K, f16/q8_0 144K. Profile ABBA vs production (flashnext-profile-ab-1): +2.5% pp/+6.5% tg at 10K, +7.4% pp/+7% tg at 80K, complete separation. Deep fills to 164K and 228K tokens pass. Open: does attention running only on the 2 XTX idle the R9700 (load balance); cause of the 248K collapse (paging vs a context-length threshold). GPT review: req_ee409a9b21e74e86.

2026-10-04 load-balance finding (QFP09 census): with KV/attention only on the two XTX (1,1,0), XTX flash-attn grows 0.23 -> 1.5 -> 3.6 ms/token at 10K/80K/160K and the R9700 idles in the allreduce (5.7 vs ~2.6 ms/tok at 160K). Experiment: same profile but rotated attention over all three GPUs (BIGCHERRY_ATTN_ROTATE=1, or ATTN_TS with an R9700 share) at a context that still fits (e.g. 192K), ABBA at 80K/160K; trade max context for depth speed, or switch placement by tier (240K memory-optimal vs <=192K speed-optimal).

## Change Log

- 2026-10-03T15:20:52.461503+00:00 (created-by): Created by agent
- 2026-10-03T16:05:30.647552+00:00 (updated-by): Updated: section:notes
