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
2. Qualify upstream PR #29901's 4-head tiled lightning indexer independently on HIP before touching selected-cell attention. Measure indexer kernel time and whole PP at 64K/128K/240K with ub512/1024/2048 where memory permits. Keep decode as a negative control because #29901 deliberately retains the vector kernel below a token tile.
3. Read v2 screens (80K/160K) and profile the post-#29825/#29901 decode critical path. QFP04 work proceeds only if selected-cell attention/gather remains >=5% of decode GPU time or >=0.5 ms/token at >=160K.
4. Implement 1296 step 1 (typed gather) from GPT code; numerics bit-identical to 1295 for f16 caches.
5. Before step 2, prototype a direct selected-index sparse-FA seam that consumes `bc_qsa_idx` without materializing compact K/V tensors. Compare it against typed gather. Prefer the direct seam if it removes >=1 large intermediate or >=3% decode wall at 160K/220K.
6. Only if q8 KV is again a production target and direct selected-index FA is not viable, implement FA tile q8_0 dequant-on-load. Do not maintain both a q8 typed-gather path and a direct selected-index path without independent wins.
7. Full ABBA on profile v2 at 80K/160K/220K, tracking acceptance separately from ms/step.
8. Treat the remaining dense `[n_kv,n_tokens]` QSA selection mask as part of the direct selected-index FA seam. Do not create a second score/mask-buffer optimization.

## Detailed Solution & Technical Design

Memory ownership is layered: upstream #29825 owns indexer score-buffer compaction; #29901 owns indexer compute tiling if it proves beneficial on HIP; QFP04 owns selected-cell attention/gather cost. The three should compose and must be benchmarked separately.

The preferred end-state is one canonical selection representation feeding attention directly:

```cpp
// selection/indexer owns canonical compact ids
bc_qsa_idx = qsa_select(...);

// preferred sparse decode seam: FA consumes selected KV ids directly
flash_attn_ext_selected(q, k_cache, v_cache,
                        bc_qsa_idx, n_selected,
                        kv_type, scale, ...);
```

The kernel should stage only selected K/V tiles, dequantize the stored KV type while loading when needed, and perform softmax over the selected set. It must not allocate `K_selected_f32 -> K_selected_f16` or a second dense mask. The existing 1295 GET_ROWS path remains the correctness/performance control.

This direction is supported by the independent SYCL sparse-FA Qwen4Exp experiment in llama.cpp discussion #28695: at 131K/q8_0 KV on Intel Arc B70 it reports roughly 3x long-context decode improvement. That result is not transferable performance evidence for AMD, but it is useful mechanism evidence that backend-native selected-cell FA can outperform graph-level gather plus dense attention. The HIP decision must be made from gfx1100/gfx1201 measurements.

Upstream PR #29901 provides a second useful separation-of-concerns example. For the 4-head Qwen4Exp indexer it stages 64 keys once and scores them against 8 tokens, retaining the vector kernel for smaller batches. On RTX PRO 6000 the PR reports indexer kernel time 16.4 -> 6.4 ms at kv=65536, nb=2048 (2.5x), PP 3867 -> 4038 t/s at 128K, indexer GPU share 9.6% -> 4.0%, with TG unchanged. This is prefill/indexer work, not evidence for changing QFP04 decode attention.

## Code Samples & Guidance

For memory qualification, record peak allocator/compute-buffer bytes beside throughput and context. A speed-neutral change that recovers `-b 4096` at deep context is still a valid prerequisite because it increases the feasible operating point.

For direct sparse FA, instrument bytes read from K/V, selected-cell count, gather/materialization bytes, kernel duration, and total attention segment time. A kernel microbenchmark win is insufficient if graph setup/materialization remains on the critical path.

## Files

Current Qwen4Exp/QSA model builder at the rebased llama.cpp pin; existing 1295/1296 patch sources; HIP Flash Attention extension only if the selected-index interface passes the gate. #29901 qualification should remain an upstream patch/control rather than be copied into QFP04.

## Validation

f32 CPU reference max|dp| <= masked path; ms/step ABBA at 80K and 160K; deep fill to ~227K without OOM. For #29825 baseline qualification: compare peak compute-buffer bytes before/after on one XTX and R9700, then the production tensor split; pp/tg must remain within noise and greedy/logit contract must match upstream expectations.

For #29901: test gfx1100 and gfx1201, PP at 64K/128K/240K, ub512/1024/2048 where feasible, plus TG128/TG512 negative controls. Record indexer kernel ms and GPU-time share. Adopt/track upstream only if whole-PP improves >=3% or indexer time falls >=20% without >1% TG regression.

For direct selected-index FA versus typed gather: f16 and q8_0 KV where supported; 80K/160K/220K; selected counts around operational min/median/max; zero OOB under sentinel/tail cases; logits/reference accuracy no worse than 1295; promotion requires >=3% decode wall or >=0.5 ms/token improvement at >=160K and no >2% regression below 80K.

## Effort & Risk

Medium-high. #29825 is merged upstream and should reduce local code by replacing any equivalent memory workaround. #29901 should be qualified as an upstream prefill optimisation, not folded into QFP04 code. QFP04's sparse-attention changes remain numerically sensitive and require explicit acceptance tracking.

## Standards

Upstream-first; one owner per buffer class; memory-attribution before new allocation work; no duplicate QSA score path; one selected-cell representation; separate prefill indexer and decode-attention claims.

## Acceptance Criteria

- Current BigCherry pin contains or cleanly rebases #29825 and its AMD memory effect is measured.
- #29901 is qualified separately on HIP before local indexer work; no local duplicate tiling implementation is created.
- No local duplicate indexer-score compaction remains.
- 1296 typed gather is retained only if it beats the direct selected-index seam or provides a necessary correctness fallback.
- Selected-cell attention reduces cost without full-cache conversion or extra large intermediates.
- Dense-mask removal reuses `bc_qsa_idx`; no parallel selection buffer is introduced.

## Notes

Evidence 1295 (old q8 profile): -3.7% ms/step @80K, -12% @160K, unchanged <32K; more accurate than masked path vs f32 CPU reference (max|dp| 0.056 vs 0.073). Its 192K OOM was the hipGraphInstantiate OOM fixed by 1302 (QFP06), not its workspace. Upstream sparse-FA work gathers compact q8_0 blocks and dequantizes during FA, +143% at 88K on CUDA. Supersedes PNRO12 (same intent, never implemented). RNX02 keeps the dense head-dim-256 decode side. Screening on production profile v2: flashnext-v2-1295-d65536/-d131072 (queue-v2-1295.sh). GPT: 1296 step-1 code req_8d8c8c85fecf4332; review of 1295 req_af96db92b8e34cfe.

2026-10-04 1295 on production profile v2 (f16/f16 240K, KV on both XTX; flashnext-v2-1295-d65536/-d131072, quick ABA, 256 tokens): ~80K 53.3 vs 53.5/52.8 ms/step = neutral (with f16 KV there is no full-cache q8->f16 conversion for the gather to avoid); ~160K 64.0 vs 65.9/66.3 ms/step = -3.2%, and t/s 47.6 vs 41.8/41.1 (+15%) because acceptance rose 162-163/276-279 -> 172/246 (gather changes numerics; earlier accuracy test found the gather closer to the f32 reference). Single new arm; needs a full ABBA at ~160K and ~220K, with acceptance tracked, before adoption. 1296's main target (q8 conversion) matters less on f16 KV; its typed gather is still worth it to drop the f32 round trip.

Upstream references verified 2026-10-05:
- #29825 merged indexer memory compaction: https://github.com/ggml-org/llama.cpp/pull/29825
- #29901 open tiled 4-head lightning indexer: https://github.com/ggml-org/llama.cpp/pull/29901
- SYCL sparse FA Qwen4Exp mechanism: https://github.com/ggml-org/llama.cpp/discussions/28695
- Latest published release: b11401 (`a7fb71fab83b474a0892b9a05aaa3a8ddca2729b`), 2026-10-05 00:36 UTC: https://github.com/ggml-org/llama.cpp/releases/tag/b11401

## Change Log

- 2026-10-03T15:20:34.367684+00:00 (created-by): Created by agent
- 2026-10-03T15:45:02.801299+00:00 (updated-by): Updated: section:notes
- 2026-10-04 (agent): Folded merged upstream #29825 into QFP04 as the canonical QSA indexer-memory baseline; separated score-buffer compaction from selected-cell sparse-attention ownership.
- 2026-10-05 (agent): Deep scan: staged #29901 HIP indexer qualification before local work; added direct selected-index sparse-FA as the preferred consolidation target over maintaining typed gather plus dense-mask paths; added explicit performance/ownership gates and SYCL mechanism reference.
