---
id: BCOP62
order: 62
plan: patching-bc-optimizations
state: completed
created-at: '2026-10-08T18:12:33+11:00'
created-by: agent
priority: P2
---

# PNRO06/07: close unqualified hybrid TOP_K and wave32 follow-up

## Discovery / change

Pinned b11474 and upstream master share the same native HIP radix/bitonic TOP_K source blob. The nasone 1256/1257 ports are genuine but 2026-09-27 BigCherry hardware series did not establish positive CI95-low E2E gains; MoE used fused `topk_moe` rather than generic TOP_K. Validated 1294 owns QSA deterministic ties/ordered output and conflicts with 1256. Closed PNRO06 and subordinate PNRO07 without promotion; historical patch states remain untested for provenance.

## Ownership / active work

PNRO06 owns the generic TOP_K negative disposition; PNRO07 owns its wave32 increment. RNX02/1294 own QSA tie correctness; fused MoE router and QFP17's active QSA-mask work remain with their independent owners. Recent QFP17/1330, QFP37, MTP/PRBE52, MET01, PA44/PA47 and 1344 HC work are excluded and untouched. Last independent 1256 change 2026-10-04, 1257 change 2026-09-28, 1294 change 2026-10-07 11:12 UTC (outside the 12-hour window); no queued PNRO06/07 lane found.

## Terminal disposition / reopen gate

No new patch, experiment queue or scheduler. Reopen only with a production generic TOP_K callsite accounting for >=5% E2E wall time, explicit 1294-compatible deterministic semantics, host tie/shape fixture, graph/multi-ubatch correctness and >=4-session >=10-round paired CI95-low >=3% E2E benefit with <=1% control regression. Otherwise remain closed. No action required from active neighbouring owners.

## Evidence and validation boundary

- https://github.com/ggml-org/llama.cpp/blob/b11474/ggml/src/ggml-cuda/top-k.cu
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py
- https://github.com/sgl-project/sglang/issues/26771
- BigCherry PNRO06/PNRO07 2026-09-27 series-2 measurements and 1294 patch summary.
- 48 host route cases and tie-order illustration ran; source/blob and patch metadata checks passed. No build, HIP test or new hardware measurement.
