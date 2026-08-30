# 0200_dispatch_hook: Route the dense matmul selector through measured dispatch (HI04)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Inserts a single guarded hook (ggml_hip_dispatch_mul_mat) at the top of upstream's ggml_cuda_mul_mat entry points; the hook returns false whenever it declines, so upstream's own ladder runs untouched. Also exposes the previously-static cuBLAS entry point so the BLAS candidate can reach it.

## Why

Upstream's selector decides and launches in one motion, so there is nothing to measure, store, or replay. A minimal, appended hook keeps the diff tiny and durable across releases while guaranteeing the native fallback is upstream's real code, not a reimplementation.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI04).
