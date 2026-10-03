---
id: QFP04
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:20:34.367684+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# 1295/1296 QSA gather-based sparse decode attention (gather in stored KV type, dequant in FA)

## Description

Patch 1295_qsa_gather_decode (evaluated v2): for n_tokens<=8 and n_kv>=BIGCHERRY_QSA_GATHER_MIN (32768) build_attn_qsa gathers only the QSA-selected cells (bc_qsa_idx) via ggml_reshape_2d(k_all, hd*nh_kv, n_kv) + get_rows (f32) + cast f16, mask gathered with a -inf sentinel column, padded to 256, row indices clamped. Upstream sparse FA (n_kv_max, mask compaction) is NVIDIA-only; HIP QSA attention is dense. 1296 = next step: (1) gather rows in the cache's stored type (f16 or q8_0 byte gather) with no f32 round trip, one rows_idx/mask gather per layer; (2) tile FA dequant-on-load for q8_0 so no full-cache conversion.

## Steps

1. Read v2 screens (80K/160K). 2. Implement 1296 step 1 (typed gather) from GPT code; numerics bit-identical to 1295 for f16 caches. 3. Step 2: FA tile q8_0 dequant-on-load (needed only if q8 KV returns). 4. ABBA on profile v2 at 80K/160K/220K.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

f32 CPU reference max|dp| <= masked path; ms/step ABBA at 80K and 160K on profile v2; deep fill to ~227K without OOM.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Evidence 1295 (old q8 profile): -3.7% ms/step @80K, -12% @160K, unchanged <32K; more accurate than masked path vs f32 CPU reference (max|dp| 0.056 vs 0.073). Its 192K OOM was the hipGraphInstantiate OOM fixed by 1302 (QFP06), not its workspace. Online (GPT RV4214): upstream sparse-FA work gathers compact q8_0 blocks and dequantizes during FA, +143% at 88K on CUDA. Supersedes PNRO12 (same intent, never implemented). RNX02 keeps the dense head-dim-256 decode side. Screening on production profile v2: flashnext-v2-1295-d65536/-d131072 (queue-v2-1295.sh). GPT: 1296 step-1 code req_8d8c8c85fecf4332; review of 1295 req_af96db92b8e34cfe.

2026-10-04 1295 on production profile v2 (f16/f16 240K, KV on both XTX; flashnext-v2-1295-d65536/-d131072, quick ABA, 256 tokens): ~80K 53.3 vs 53.5/52.8 ms/step = neutral (with f16 KV there is no full-cache q8->f16 conversion for the gather to avoid); ~160K 64.0 vs 65.9/66.3 ms/step = -3.2%, and t/s 47.6 vs 41.8/41.1 (+15%) because acceptance rose 162-163/276-279 -> 172/246 (gather changes numerics; earlier accuracy test found the gather closer to the f32 reference). Single new arm; needs a full ABBA at ~160K and ~220K, with acceptance tracked, before adoption. 1296's main target (q8 conversion) matters less on f16 KV; its typed gather is still worth it to drop the f32 round trip.

## Change Log

- 2026-10-03T15:20:34.367684+00:00 (created-by): Created by agent
- 2026-10-03T15:45:02.801299+00:00 (updated-by): Updated: section:notes
