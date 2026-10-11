---
id: BCOP130
order: 130
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T04:05:44+00:00'
created-by: agent
priority: P2
work: S
---

# PNRO16: retire duplicate offline placement; MET11 owns identity gate

## Discovery / disposition

PNRO16 proposed a separate GGUF placement compiler, but `expert-place.py` already supplies deterministic STRP-based placement and a `--plan-only` permutation oracle. MET08 owns placement evidence; MET11 owns load-time permutation; 1283/1281 own expert split and range-local execution. Move PNRO16 to completed/superseded; no second solver, CLI, scheduler, cache or telemetry.

## Eligibility / exclusions

Selected `feat/qfp49-stagger-pair` from independent 1361 harness commit `2a6be00af00e6951a1ff9b96c90f3c847649dc4a` (2026-10-11 04:04:53 UTC). No independent PNRO16/MET11/expert-place.py changes or PRs within 12 hours; last substantive MET11 work October 7. Recent BCOP124–129 audited other capabilities. Protected QFP49/1360/1361, QFP50/1359, QFP41/1356, QFP17/1330, Radiance and recently BCOP-updated MET08 remain untouched.

## Bounded action / terminal

MET11 must validate positive integer caps, finite positive traffic, STRP bounds, model/shard/quant layout and verified profile/model identity; require exact B2 inverse maps, per-device VRAM budgets and negative controls. Compare one real three-shard UD-IQ4_XS map before GPU work. If identity cannot be proven, loader remains disabled; PNRO16 has no independent action.

## Evidence

Source inspected: `expert-place.py::{read_profile,place,main}`, 1283/1281, MET08/10/11, RPL01, `queue-moe-ep.sh`, upstream gguf-py ReaderTensor, [llama.cpp #30112](https://github.com/ggml-org/llama.cpp/pull/30112), [vLLM ExpertMapManager](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/expert_map_manager.py). Disposable Python: three valid placement bijections passed; negative cap and NaN traffic admitted; different synthetic model layouts produce indistinguishable old plan schema. No repository pytest, HIP/C++ build, real GGUF processing, GPU execution or new benchmark.
