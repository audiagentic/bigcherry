---
id: PNRO16
order: 0
plan: patching-nasone-rdna-optimizations
state: superseded
created-at: '2026-09-11T04:32:00.691672+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# Offline MoE placement feasibility — superseded by MET08/MET11

## Disposition (2026-10-11)

**Retire PNRO16's proposed separate `tools/lab/moe-expert-placement/` inventory/compiler/replay tool.** This is duplicate ownership, not a completed performance optimisation. No new patch, scheduler, cache, placement solver, telemetry or tooling registry.

**Actual implementation:** `tools/lab/flash-next/expert-place.py::read_profile` consumes Strata `STRP` expert rankings; `place` assigns per-layer experts deterministically using `--caps` and weighted-credit `--traffic`; `main` emits `--plan-only` JSON and can reorder expert gate/up/down and router rows in copied GGUF shards. It imports `GGUFReader` from `vendor/llama.cpp/gguf-py`, resolving PNRO16's old import question. `1283_qwen4exp_expert_parallel` already splits whole expert weights by expert axis, with 1281 range-local execution. MET08 owns usage-placement measurements; MET11 owns the future load-time source-GGUF-preserving permutation and post-top-k B2 ID translation; MET10 host tail and RPL01 advisory cost are separate owners.

**Remaining gap transferred to MET11:** Existing `--plan-only` reads no GGUF and emits no tensor-layout, shard, profile digest or verified model identity. It cannot distinguish same-count/different-quant models. Its CLI accepts negative caps whose sum matches n_expert and non-finite traffic. A layout fingerprint alone does not bind a routing profile to the actual weights. No independent PNRO16 implementation is warranted; MET11's authoritative section now specifies strict admission, identity, inverse-map, per-device byte budgets and tests.

**Measurement boundary:** An offline routing trace does not prove actual H2D/D2H, PCIe, RCCL or critical-path cost on no-P2P hardware. Reuse existing Meta/profiling receipts and MET08 measured controls. No new speedup is claimed.

## Terminal gate and historical context

Created 2026-09-11 as a read-only offline inventory/byte accounting/fingerprint/placement/replay concept, refined 2026-09-24. No llama.cpp Edit() package was intended. Existing `tools/lab/flash-next/queue-moe-ep.sh` identifies the three-shard Qwen3.8-Flash-Next UD-IQ4_XS target; compact real evidence belongs in `docs/evidence/`. A new PNRO16 tool would duplicate MET11 and require unnecessary `TOOL_DISPOSITION.md` entries. MET11 must prove profile/model identity and same-permutation correctness or leave the opt-in loader off. No new hardware experiment while QFP49/1360/1361 is active.

References: [MET08](../../active/patching-moe-expert-tiering/MET08.md), [MET11](../../active/patching-moe-expert-tiering/MET11.md), [MET10](../../active/patching-moe-expert-tiering/MET10.md), [RPL01](../../active/patching-runtime-placement/RPL01.md), [llama.cpp #30112](https://github.com/ggml-org/llama.cpp/pull/30112), [vLLM ExpertMapManager](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/expert_map_manager.py).
