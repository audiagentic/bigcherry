# 0900_pool_workspace_metrics: Measured per-candidate workspace via the pool's own bookkeeping (HI52 part 1)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Adds a bc_requested_size counter to ggml_cuda_pool's base struct, incremented/decremented at the single choke point (ggml_cuda_pool_alloc's two call sites) where every pool alloc/free in the CUDA/HIP backend passes through, tracking each candidate's requested size rather than the pool's actual-size bookkeeping.

## Why

measurement.workspace_bytes previously reported each candidate's declared upper bound, which is constant within a family and has never discriminated anything (HI45's low-memory Pareto profile always reports 0% savings). A device-global hipMemGetInfo delta was tried and failed structurally because the caching pool reuses a high-water-mark allocation. The pool's own bookkeeping is the only place that actually knows the answer.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI52).
