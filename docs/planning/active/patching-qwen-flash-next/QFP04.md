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

Upstream llama.cpp PR #29825 is now the memory baseline for this item. It merged 2026-10-03 and changes Qwen4Exp/QSA indexer scoring from all-heads-at-once plus a separate ReLU copy to sequential-head scoring with in-place ReLU and accumulation. Published CUDA compute-buffer usage roughly halves without pp/tg loss (131K/ub2048 3.0 -> 1.5 GiB; 131K/ub4096 6.1 -> 3.0 GiB; 262K/ub4096 11.7 -> 5.6 GiB). BigCherry must rebase this upstream change before attributing any remaining long-context OOM or workspace cost to QSA gather/FA.

## Steps

1. Rebase/confirm upstream #29825 at the current pin first; capture AMD gfx1100/gfx1201 compute-buffer high-water marks at 131K/262K and ub2048/4096. Do not carry a local equivalent if the upstream implementation applies cleanly.
2. Read v2 screens (80K/160K).
3. Implement 1296 step 1 (typed gather) from GPT code; numerics bit-identical to 1295 for f16 caches.
4. Step 2: FA tile q8_0 dequant-on-load (needed only if q8 KV returns).
5. ABBA on profile v2 at 80K/160K/220K.
6. Treat the remaining dense `[n_kv,n_tokens]` QSA selection mask as a separate future seam only after memory attribution. Upstream #29825 already reuses that buffer; reducing it further likely requires passing selected cell indices into `flash_attn_ext` instead of materializing a dense mask. Do not create a second score-buffer optimization.

## Detailed Solution & Technical Design

Memory ownership is now layered: upstream #29825 owns indexer score-buffer compaction; QFP04 owns selected-cell attention/gather cost. The two should compose. Any future selected-index FA interface should reuse `bc_qsa_idx`/the upstream selection representation and remove the dense mask, not add another parallel selection buffer.

## Code Samples & Guidance

For memory qualification, record peak allocator/compute-buffer bytes beside throughput and context. A speed-neutral change that recovers `-b 4096` at deep context is still a valid prerequisite because it increases the feasible operating point.

## Files

Current Qwen4Exp/QSA model builder at the rebased llama.cpp pin; existing 1295/1296 patch sources; Flash Attention extension only if a selected-index interface is promoted after memory attribution.

## Validation

f32 CPU reference max|dp| <= masked path; ms/step ABBA at 80K and 160K on profile v2; deep fill to ~227K without OOM. For #29825 baseline qualification: compare peak compute-buffer bytes before/after on one XTX and R9700, then the production tensor split; pp/tg must remain within noise and greedy/logit contract must match upstream expectations.

## Effort & Risk

Medium-high. #29825 is merged upstream and should reduce local code by replacing any equivalent memory workaround. QFP04's sparse-attention changes remain numerically sensitive and require explicit acceptance tracking.

## Standards

Upstream-first; one owner per buffer class; memory-attribution before new allocation work; no duplicate QSA score path.

## Acceptance Criteria

- Current BigCherry pin contains or cleanly rebases #29825 and its AMD memory effect is measured.
- No local duplicate indexer-score compaction remains.
- 1296 reduces selected-cell gather/attention cost without reintroducing full-cache conversion or extra large intermediates.
- Any future dense-mask removal proves peak-memory reduction and reuses existing selection indices.

## Notes

Evidence 1295 (old q8 profile): -3.7% ms/step @80K, -12% @160K, unchanged <32K; more accurate than masked path vs f32 CPU reference (max|dp| 0.056 vs 0.073). Its 192K OOM was the hipGraphInstantiate OOM fixed by 1302 (QFP06), not its workspace. Online (GPT RV4214): upstream sparse-FA work gathers compact q8_0 blocks and dequantizes during FA, +143% at 88K on CUDA. Supersedes PNRO12 (same intent, never implemented). RNX02 keeps the dense head-dim-256 decode side. Screening on production profile v2: flashnext-v2-1295-d65536/-d131072 (queue-v2-1295.sh). GPT: 1296 step-1 code req_8d8c8c85fecf4332; review of 1295 req_af96db92b8e34cfe.

2026-10-04 1295 on production profile v2 (f16/f16 240K, KV on both XTX; flashnext-v2-1295-d65536/-d131072, quick ABA, 256 tokens): ~80K 53.3 vs 53.5/52.8 ms/step = neutral (with f16 KV there is no full-cache q8->f16 conversion for the gather to avoid); ~160K 64.0 vs 65.9/66.3 ms/step = -3.2%, and t/s 47.6 vs 41.8/41.1 (+15%) because acceptance rose 162-163/276-279 -> 172/246 (gather changes numerics; earlier accuracy test found the gather closer to the f32 reference). Single new arm; needs a full ABBA at ~160K and ~220K, with acceptance tracked, before adoption. 1296's main target (q8 conversion) matters less on f16 KV; its typed gather is still worth it to drop the f32 round trip.

Upstream reference verified 2026-10-04: https://github.com/ggml-org/llama.cpp/pull/29825 . The PR notes the residual dense selection mask is the next major reserve after score-buffer compaction.

## Change Log

- 2026-10-03T15:20:34.367684+00:00 (created-by): Created by agent
- 2026-10-03T15:45:02.801299+00:00 (updated-by): Updated: section:notes
- 2026-10-04 (agent): Folded merged upstream #29825 into QFP04 as the canonical QSA indexer-memory baseline; separated score-buffer compaction from selected-cell sparse-attention ownership.
