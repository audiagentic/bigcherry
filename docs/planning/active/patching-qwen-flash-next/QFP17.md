---
id: QFP17
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-04T07:22:34.987044+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Flash-Next prefill program (attribution, ubatch via upstream #29825, QSA sparse prefill, MoE MMQ, GDN, rank balance)

## Description

Prefill today (v4, 240K f16 KV, -ub 512 -b 512): ~1,100 t/s at ~10K, ~910 t/s at ~80K. GPT survey + methods (req_6010ee79a7df4529, 2026-10-04) ranked levers; key challenge: attribute the ~17% 10K->80K decay first (context-growing QSA/attention/indexer vs constant MoE/GDN), optimize the critical-rank wall path (not summed kernel time), and gate every item by Amdahl (reject optimistic bound < ~3%). Immediate finding: validated prefill patches 1237 + 1265 (MoE MMQ compact grid, ~+7% MoE prefill on gfx1100, gfx1201 via 1265) and 1253 (BF16/WMMA chunked GDN prefill for gfx11/gfx12) were never in the Flash-Next profile (built from the stock lane + an explicit list) - screened first (deploy-v4-pp, queue-v4-pp.sh).

## Steps

0. deploy-v4-pp ABBA (1237 + 1265 + 1253 on v4): prefill and decode, greedy identity.
1. PREFILL-ATTR: rocprof at 10K/80K/240K on v4(-pp), bucket ubatches by context position (0-10K, 10-20K, ...); per bucket per GPU: MoE MUL_MAT_ID/MMQ by quant, GDN chunked vs fallback, QSA indexer/top-k/mask/FATTN, AllReduce bytes + per-rank start/end (arrival_skew, collective_tail, exposed_barrier), copies by direction, CPU get_rows/PLE, draft-context prefill wall. 1319/1320/1325 for host enqueue.
2. QSA-UBATCH: backport upstream llama.cpp #29825 (halves Qwen4Exp indexer compute buffer; speed-neutral itself) then sweep -b/-ub 512/768/1024/1536/2048 at 240K f16 (prior: ub512 -> 1024 -> 2048 = 933 -> 1139 -> 1328 t/s in an older config). Expected +5-25%. Gate: larger ubatch fits and ABBA > 5%.
3. QSA-SPARSE-PP: HIP flash-attn variant consuming selected indices directly (avoid the dense [n_kv, n_tokens] selection mask and per-query gathered KV); 0-8% at 10K, 5-30% at 80K+. Gate: QSA/FATTN/indexer >= 10% wall at 80K and growing with context.
4. MOE-MMQ-PP2: after 1237/1265, tokens-per-expert tile sizing (AMD fork #39 style) / ExLlamaV3 multi-row expert tiles (ub512 = ~10 rows/expert, ub2048 = ~40). +3-15%. Gate: routed MMQ >= 25% critical path with low rows/expert.
5. GDN-PP2: verify 1253 is taken (no fallback), sweep chunk 16/32/64, occupancy/LDS/VGPR. +2-10%. Gate: GDN >= 15% wall. (1221 is RDNA3.5-only - not the baseline here.)
6. TP-PREFILL-BALANCE: -ts tuned against AllReduce arrival times (R9700 late?), keep the independent attention split. +3-10%. Gate: p50/p95 arrival skew >= 5% of layer wall.
7. QFP08 draft-prefill overlap (TTFT, 2-10%). Gate: serialized draft fill >= 5% TTFT.
8. Conditional: two-microbatch compute/communication overlap (SGLang TBO) only if collective_tail >= 5% after balancing; CPU PLE/get_rows pinning only if >= 3-5% wall.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Wall TTFT / prefill t/s ABBA (fixed prompt, warmup, clocks) at 10K/80K; greedy identity; VRAM headroom at 240K f16.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Rejected from the survey for this lane: large-message CPU-root AllReduce (measured worse), KTransformers CPU-expert offload (experts fit VRAM; PCIe x4/no P2P), MLC-LLM/mistral.rs (no portable RDNA/IQ kernels). Sources and links in the GPT response req_6010ee79a7df4529 (llama.cpp #29825, AMD fork #39/#63, ExLlamaV3 MoE tiling, FLA chunked GDN, SGLang ROCm GDN + TBO, TRT-LLM selected-index sparse attention, AITER experimental gfx11/12).

CORRECTION 2026-10-04: 1237 + 1265 (MoE MMQ compact grid) and 1253 (chunked GDN prefill) are NOT missing from the Flash-Next profile - [source.bigcherry] composes patch-sets serving-core + upstream-fixes + validated-enhancements into every build (validated-enhancements = 0860, 1225, 0840, 1237, 1241, 1253, 1274, 1265); the lane name 'stock' only refers to build options. deploy-v4-pp failed with 'overlay repeats a base patch module' and was removed. Step 0 is void; current prefill numbers already include them. GDN-PP2 should still verify 1253 is actually taken (no fallback) on this model.

## Change Log

- 2026-10-04T07:22:34.987044+00:00 (created-by): Created by agent
- 2026-10-04T07:55:32.606718+00:00 (updated-by): Updated: section:notes
