---
id: BCOP126
order: 126
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-10T23:07:22+00:00'
created-by: agent
priority: P2
work: S
---

# PRBE29: admit F16 shadows only after complete model uploads

## Discovery / disposition

Pinned b11474 model loader supports async **chunked** uploads through `ggml_backend_tensor_set_async` and synchronizes their events at `load_all_data` completion. AMD-Ecosystem PR #57 creates a persistent F16 shadow from the synchronous `ggml_backend_cuda_buffer_set_tensor` hook only, so that external implementation cannot cover the async route as-is. Its `tensor->extra != nullptr` early return also lacks source-mutation invalidation. These are source-proven contract gaps, not reproduced GPU bugs. Keep PRBE29 default-off pending a typed dense GEMM critical-path and VRAM gate; do not port the hook blindly.

## Ownership / protected work

PRBE29: shadow creation, same-device lifetime and dispatch admission; PRBE30: conditional F16-shadow padding; PRBE31: existing BLAS vs Lt crossover; PRBE70: typed cost census; PRBE28: original float GGUF row padding. No second allocator, global cache, dispatcher or telemetry system. Last independent PRBE29/30/31 file commit 2026-10-09 04:41 UTC (mechanical relocation); substantive planning 2026-10-08. No associated independent patch, PR or queued experiment found in the preceding 12 hours from inspected repository evidence. Excluded active QFP49/1360/1361 cross-card and prefill work, QFP50/1359, QFP41/1356, QFP17/1330, QFP36/1357 and Radiance. No protected file changed.

## Bounded next action / terminal outcome

Gate 0: PRBE70 typed same-device dense GEMM receipt with >=2.9126% removable E2E headroom and safe post-KV VRAM reserve; else `BLOCKED_ON_TYPED_DENSE_GEMM` or reject. Gate 1: source/shadow generation, full/async/host upload and overwrite host/backend fixtures; no READY before events drain, no stale pointer or foreign GPU. Gate 2: one <=256 MiB/device default-off F16 shadow, correct M-per-ubatch route, numerical/greedy/MTP, graph, repeated-request, load and VRAM controls. Gate 3: four-session paired ABBA per architecture, CI95-low >=3% E2E, <=1% control regression; otherwise reject and retain negative evidence. Detailed design belongs only in PRBE29.

## Evidence and actual validation

Pinned b11474 `src/llama-model-loader.cpp::load_all_data`, `src/llama-model.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu`; [AMD #57](https://github.com/AMD-Ecosystem/llama.cpp/pull/57) (open, updated 2026-07-18), [ggml-org #26621](https://github.com/ggml-org/llama.cpp/pull/26621) (open, updated 2026-08-08), [vLLM RDNA3 quantized GEMM](https://github.com/vllm-project/vllm/blob/main/csrc/rocm/moe_q_gemm_rdna3.cu). Eight disposable Python shadow-lifetime assertions passed; arithmetic shadow memory gates checked. No repository pytest, C++/HIP build, real model, GPU benchmark or new BigCherry speedup.
