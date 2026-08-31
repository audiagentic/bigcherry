# 1001_hip_internal_allreduce: Upstream backport: enable the internal (non-RCCL) AllReduce on HIP

**Status:** validated
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
this internal path targets.

## Validation evidence (dual RX 7900 XTX / gfx1100, 2026-08-31)

**Correctness** (BigCherry's HI18 SPLIT_REDUCE probe, `test-hip-reduce`,
6 pattern/seed combinations): `internal` with
`GGML_CUDA_AR_BF16_THRESHOLD=0` matches RCCL and the FP64 reference
within the F32 rounding floor (nmse=7.7e-16, max_abs=5.96e-08,
bit-identical output digests) -- PASS.

**Performance** (fresh single-session, back-to-back 3-arm paired
completion-bench, 24 prompts, same binary, only env vars varied):
- A, `GGML_CUDA_ALLREDUCE=nccl` (control): 59.19 TPS
- B, `GGML_CUDA_ALLREDUCE=internal` + `GGML_CUDA_AR_BF16_THRESHOLD=0`
  (exact FP32 wire): 69.31 TPS -- **B vs A = +17.33%, 95% CI
  [+13.41%, +21.25%]**
- C, `GGML_CUDA_ALLREDUCE=internal` with BF16 wire at its upstream
  default (`GGML_CUDA_AR_BF16_THRESHOLD=1`): 63.78 TPS -- C vs A =
  +8.52%, 95% CI [+4.37%, +12.66%]; **C vs B = -7.27%, 95% CI
  [-10.56%, -3.99%]**

The internal mechanism's win over RCCL is real and validated at exact
precision, independent of the BF16 wire-compression tradeoff. `draft_acceptance`
and `mean_accepted_length` were materially unchanged across all three
arms (0.4946-0.5019, 2.965-2.993) -- no speculative-decoding behavior
shift.

**BF16 wire compression is a separate, negative finding on this
topology**: `GGML_CUDA_AR_BF16_THRESHOLD=1` (upstream's default, applies
BF16 round-trip to every nonzero F32 reduction) is 7.27% *slower* than
exact precision here, not merely an accuracy/throughput tradeoff --
likely because this decode workload's collectives are small and
latency-bound (~5186 calls/sec), so the conversion kernel's own overhead
exceeds any PCIe bytes saved. Not yet profiled to confirm the mechanism.
RCCL itself already reserves BF16 for large/bandwidth-bound reductions
and uses FP32 for small ones, which supports size-dependent precision
being the right design generally -- this project's internal-path default
(BF16 for every reduction regardless of size) just doesn't match that
here.

## Production recommendation -- decode-only workloads

For validated dual-RX7900-XTX/gfx1100 tensor-split configurations where
the workload is decode-dominated (interactive serving, long generations
relative to prompt length):

```
GGML_CUDA_ALLREDUCE=internal
GGML_CUDA_AR_BF16_THRESHOLD=0
```

`GGML_CUDA_AR_BF16_THRESHOLD` must be set explicitly to `0` -- the
upstream internal AllReduce otherwise defaults to `1` (BF16 wire for
every reduction), which this project's own evidence shows is worse
here, not just lower-precision.

## WARNING: internal AllReduce is a severe regression for prefill

Real llama-bench sweep (dual RX 7900 XTX, same tensor-split config, 3
repetitions each, `-b 2048 -ub 512`) across prompt-processing (large
reduction) and text-generation (small reduction) sizes:

| test    | A (nccl) t/s | B (internal, BF16=0) t/s | diff     |
|---------|-------------:|--------------------------:|---------:|
| pp512   |      1504.67 |                     998.21 |  -33.66% |
| pp2048  |      1447.58 |                     980.64 |  -32.26% |
| pp4096  |      1431.77 |                     972.08 |  -32.11% |
| tg128   |        33.69 |                      36.01 |   +6.88% |

**Do not set `GGML_CUDA_ALLREDUCE=internal` globally or for any
prompt-processing/prefill-heavy workload** -- it is ~32-34% slower than
RCCL there, a severe regression, not a minor tradeoff. The internal
path's win is real but decode-only; RCCL remains clearly better for
large/bandwidth-bound reductions, consistent with upstream's own
BF16-for-large/FP32-for-small design intent for its wire encoding.
Real crossover confirmed -- see HI155 (size-adaptive `internal`/`rccl`
provider dispatch), the correct fix for shipping this safely across
both prefill and decode.

## Upstream / provenance

Cherry-picked from open upstream PR
https://github.com/ggml-org/llama.cpp/pull/27825. Scoped to the two
functional edits only; the PR's comment-only wording changes were not
ported (no runtime effect, and awkward to anchor against this project's
comment-blanking patch matcher).
