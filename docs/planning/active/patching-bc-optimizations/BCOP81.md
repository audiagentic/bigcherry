---
id: BCOP81
order: 81
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-09T02:12:37+00:00'
created-by: agent
priority: P3
work: S
---

# PRBE15: reject gate-only IMRoPE/BF16 ROPE scatter fusion

## Discovery / disposition

Pinned llama.cpp b11474's IMRoPE `rope_multi` lacks KV slot-index scatter and has same-type source/destination dispatch, although `ggml_cuda_op_rope_impl` computes the fused slot index, stride and destination type. Widening the fusion selector alone would write incorrect cache rows or reinterpret destination width. Native standalone SET_ROWS already supports BF16; only the fused ROPE writer lacks it. 22 static/host checks and a nonmonotonic-slot oracle passed; no hardware failure or performance benefit is claimed.

## Owner and dependencies

PRBE15 alone owns this narrowly scoped extension; native `ggml_cuda_try_fuse`/memory-range checks, `rope_multi`, `set_rows_cuda` and KV cache retain their existing owners. Rejected 1004 remains historical. Do not duplicate dispatch, scatter storage or a producer. PRBE54/QFP/MTP/Radiance/engine-registry development within the exclusion window is untouched. SGLang AITER's slot/section-aware mRoPE fusion is a semantic reference only.

## Unresolved action and terminal gate

First prove a real Qwen3-VL ROPE→VIEW→SET_ROWS IMRoPE graph, including dtype, 4-axis positions, slots, actual launches and GPU critical-path share. If absent or theoretical E2E ceiling <3%, close without a patch. If present, extend existing `rope_multi` indexed writes for same-type F32/F16 first; gate BF16 separately on native F32→BF16 control and typed output dispatch. Require sentinel/K-cache and full-vocab parity, graph replay, multi-ubatch/repeated requests, and only then four-session/ten-pair CI95-low ≥3% E2E, ≤1% control regression. On failure retain native unfused execution. No GPU experiment queued.

## Source references

Pinned: https://github.com/ggml-org/llama.cpp/tree/b11474/ggml/src/ggml-cuda ; current: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/rope.cu ; SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3.py ; correctness precedent: https://github.com/sgl-project/sglang/issues/35345 .
