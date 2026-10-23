---
id: BCOP190
order: 190
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-13T17:16:00Z'
created-by: agent
priority: P3
work: S
---

# PKC04: close duplicate Vulkan lifecycle ownership gate

## Description

**Discovery/change:** Pinned b11552 already owns Vulkan dense/sparse attention and native matmul routing. BigCherry 1290/PRVP03 owns the default-off Vulkan Meta collective SPI; `backend/identity.py` owns shared software identity, while `vk_autotune_types.py` is an unwired candidate schema, not runtime attestation. A new PKC04 Vulkan lifecycle/registry would duplicate those owners. Known 1290 inactive-rank/read-before-reject hazard is **PRVP03's** correctness issue, not a new PKC04 implementation. Upstream merged #30003 is native sparse-FA correctness baseline; open #25051 is a provider candidate, not a qualified replacement.

**Authoritative owners / independent work:** PKC04 marked `done` **for scope decision only**; TRVP12–15, RRVP02/05, PRVP02/03, TRVP16, PKC03 and EC16/19 retain implementation, driver/route receipts, contract and hardware gates. No neighbouring plan or active implementation changed. PKC04 last substantive independent change 2026-09-10; no relevant independent 12-hour commit/PR/queued Vulkan lane. Active Radiance GPU-mailbox/R9700 helper and patch-prefix/release work excluded.

**Unresolved action / terminal disposition:** None under PKC04. Reopen only with one concrete Vulkan semantic not representable by existing owners. RRVP02's user-directed implementation pause remains in force. On independent resume, PRVP03 must prevalidate inactive ranks and prevent false fallback after writes; TRVP12/13 must prove process-bound ICD/physical-device and actual bound route before RRVP05 qualification. No new provider, dispatcher, identity, scheduler, allocator, cache or hardware queue.

**Evidence:** llama.cpp b11552 `ggml-vulkan.cpp::ggml_vk_flash_attn`, `ggml_vk_get_mul_mat_mat_pipeline_map`, `ggml-backend-meta.cpp::allreduce_fallback`; BigCherry `1290/patch.py`, `backend/identity.py`, `vk_autotune_types.py`; [#30003](https://github.com/ggml-org/llama.cpp/pull/30003), [#29998](https://github.com/ggml-org/llama.cpp/pull/29998), [#25051](https://github.com/ggml-org/llama.cpp/pull/25051). Historical RRVP05 27B RADV and 1290 results are **not new performance measurements**. Actual-source assertions 14/14; disposable host tests 13/13 and Python compilation passed. No repository pytest, SPIR-V build or GPU run.

## Steps


## Detailed Solution & Technical Design


## Code Samples & Guidance


## Files


## Validation


## Effort & Risk


## Standards


## Acceptance Criteria


## Notes


## Change Log

- 2026-10-23: state 'done' (invalid) corrected to superseded
