---
id: PGC05
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-29T09:34:02.512413+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# adaptive provider: threshold arg and size-bucket routing table

## Description
Make the adaptive crossover an argument (`--allreduce-switch-bytes`), later a size-bucket table; move the Q8 threshold to an argument. Depends on PGC04.

## Steps
1. Finish candidate-path qualification first on dual gfx1100/Qwen3.8-27B-Q8_0: isolate 1254, 1261, 1241, 1206, 1245, 1205 and the located 1263 candidate; no combo throughput attribution until correctness/work equivalence passes.
2. For 1254, preserve required 1253, `GGML_CUDA_GDN_CHUNKED=0` opt-out, and marker `BIGCHERRY_PATCH_HIT patch=1254_nro05 path=gdn_mtp_prefix_bf16`.
3. For patches without activation proof, add/review trace markers before performance qualification; 1245 specifically needs `BIGCHERRY_PATCH_HIT patch=1245_gp11 path=mmvq_fusion_q8_0_ncols6`.
4. Only then sweep provider thresholds/buckets.

## Validation
Use full-vocabulary MTP logprobs (<=5e-4 where required), greedy parity, drafted/accepted counts and MTP acceptance parity, fixed-work `llama-bench`, then order-balanced `tools/lab/native-vs-patched/server-ab-*.json`. Known fact: 1254 `nro05-ab2` measured pp4096 +2.12%, decode flat, identical acceptance; this is not sign-off.

## Acceptance Criteria
No lifecycle-state changes from planning. Any throughput claim is causal, single-patch or otherwise explicitly isolated, correctness-gated, and work-equivalent.

## Change Log
- 2026-09-29: added concrete candidate qualification prerequisites and evidence rules.
- 2026-09-29T09:34:02.512413+00:00 (created-by): Created by agent

## Ledger-events


- chg_20260929_135722_allreduce-methods-are-now-sele_7144
- 2026-09-29T13:57:29.109931+00:00 (updated-by): Updated: section:ledger-events
- chg_20260929_215656_dual-xtx-27b-q8_0-plain-decode_1707
- 2026-09-29T21:57:05.647193+00:00 (updated-by): Updated: section:ledger-events
