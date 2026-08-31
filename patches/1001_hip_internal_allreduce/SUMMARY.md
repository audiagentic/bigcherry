# 1001_hip_internal_allreduce: Upstream backport: enable the internal (non-RCCL) AllReduce on HIP

**Status:** untested
**Group:** upstream-fixes
**Plan item:** none

## What it does

Removes the `GGML_USE_HIP` compile-out guard on `allreduce.cu`'s pinned-host-
memory AllReduce, substitutes `__builtin_amdgcn_s_sleep(4)` for CUDA's
`__nanosleep(100)` in the cross-GPU spin-wait, and maps the four HIP
host-mapped pinned-memory alloc APIs (`hipHostMalloc` etc.) the
implementation needs in `vendors/hip.h`. `GGML_CUDA_ALLREDUCE=internal`
(already present in the pinned base) selects this path over RCCL at
runtime.

## Why

Real dual-XTX rocprofv3 profiling on this project's own hardware found
RCCL consuming 9.9% of decode wall time (union-of-spans), with only 3.3%
overlapping matmul compute -- ~9.6% of decode wall is pure inter-GPU
communication with zero concurrent compute, at a call rate (~5186
collectives/sec) that matches exactly the small-collective-latency case
this internal path targets. Upstream's own numbers (2x AMD PCIe) show
+2.24% TG over RCCL. Not yet A/B validated on this project's hardware.

## Upstream / provenance

Cherry-picked from open upstream PR
https://github.com/ggml-org/llama.cpp/pull/27825. Scoped to the two
functional edits only; the PR's comment-only wording changes were not
ported (no runtime effect, and awkward to anchor against this project's
comment-blanking patch matcher).
