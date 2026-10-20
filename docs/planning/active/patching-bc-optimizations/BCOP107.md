---
id: BCOP107
order: 107
plan: patching-bc-optimizations
state: done
created-at: '2026-10-10T04:13:30+00:00'
created-by: agent
priority: P2
work: S
---

# PRBE29: gate F16 shadow on complete upload and effective compute type

## Discovery / action

Pinned llama.cpp b11474 has distinct synchronous buffer.set_tensor, chunked backend.set_tensor_async and borrowed mmap paths. AMD-Ecosystem PR #57's load-time F16 shadow hook only intercepts synchronous buffer.set_tensor; it does not prove completion/coverage for the other two. Its MMQ bypass is not sufficient when effective BLAS compute is F32/BF16, and its `GGML_PREFILL_DEQUANT=0/off` parser enables shadows. These are source-level admission/coverage defects in an **external candidate**, not observed BigCherry failures. No local patch or hardware result exists.

## Ownership / recent work

PRBE29 owns one quantized dense shadow and lifetime; PRBE30 K-padding within that owner, PRBE31 the existing MMQ/BLAS crossover, PRBE28 float original-weight row padding and PRBE70 offline GEMM attribution. PRBE29 was outside the 12-hour independent-work exclusion. Radiance gfx1100 MXFP4, QFP35/36/41/43, Flash-Next accuracy/router and MTP paths were excluded and unchanged. No new scheduler, allocator family, registry, cache, queue or config surface.

## Terminal gate

First prove one hot, owned, fit-safe dense tensor and >=3% optimistic end-to-end prefill opportunity using existing profiling. Then prove exactly-once post-upload/event completion before graph capture, effective F16 compute, a <=256 MiB/device **peak** shadow+scratch budget and numerical/graph parity. If no safe finalization seam, close `NO_SAFE_FINALIZE`; if no hot tensor, close `NO_HOT_DENSE`. Only surviving default-off candidate receives four-session paired gfx1100/gfx1201 qualification (CI95-low >=3% E2E prefill, <=1% decode/control regression). Otherwise record `OOM`, `NUMERIC_FAIL` or `NO_GAIN` and retain native MMQ.

## Evidence

Pinned b11474 `src/llama-model-loader.cpp::load_all_data`, `ggml/src/ggml-cuda/ggml-cuda.cu::{ggml_backend_cuda_buffer_set_tensor,ggml_backend_cuda_set_tensor_async,ggml_cuda_mul_mat,ggml_cuda_mul_mat_cublas}`; AMD-Ecosystem/llama.cpp PR #57 actual diff, upstream #26621, AMD ROCm hipBLASLt documentation. Disposable Python source-aligned fixture: 12/12 admission, 4/4 env, 2/2 loader-path cases passed; no repository pytest, compilation, GPU run or benchmark.
