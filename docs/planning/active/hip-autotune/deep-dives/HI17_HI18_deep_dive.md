# HIP Autotune Deep Dive: HI17 BLAS Execution Plans and HI18 SPLIT_REDUCE

**Date:** 2026-08-11  
**Scope:** Implementation guidance for plan items **HI17** and **HI18** only  
**Audience:** Developer or implementation agent modifying the project's pinned llama.cpp HIP/ROCm path  
**Status:** Design and implementation guidance. No proposed optimization is considered faster until the validation in this document passes.

> HI17 is completed at `4732f47`. The HI17 sections below are historical
> design and implementation guidance retained for audit; current
> implementation work applies only to HI18. Do not reopen HI17 or treat its
> future-tense recommendations as pending acceptance.

---

## 1. Purpose

This document turns HI17 and HI18 into a code-level implementation guide. It covers:

- the real llama.cpp execution paths;
- the HIP/hipBLAS/rocBLAS/hipBLASLt/RCCL interfaces behind them;
- which controls are genuinely per-call and which are process/library state;
- corrected candidate schemas;
- concrete insertion seams and pseudocode;
- requested-versus-effective telemetry;
- numerical and correctness gates;
- complete-path timing;
- profiling and source-archeology workflows;
- validation matrices and stop conditions;
- deeper follow-on tuning opportunities.

The governing rule is:

> **A tuner may only rank choices that it can control, observe, and validate.**

A stable candidate name does not prove a library followed the requested path. An environment variable does not become a per-signature control merely because it can be serialized. A fast timing is not promotable if the effective execution path is unknown.

---

## 2. Evidence and version boundaries

### 2.1 Plan-derived evidence

This guide is based on:

- **HI17 — Decompose the opaque BLAS candidate into a controlled execution plan**
- **HI18 — Tune the multi-device allreduce algorithm (SPLIT_REDUCE)**

Their RV30 contracts remain the baseline, but several abstractions are tightened here after following the actual source and public ROCm APIs.

### 2.2 Current upstream versus the pinned project tree

Current upstream llama.cpp was inspected on 2026-08-11. Important files are:

```text
ggml/src/ggml-cuda/ggml-cuda.cu
ggml/src/ggml-cuda/vendors/hip.h
ggml/src/ggml-cuda/convert.cuh
ggml/src/ggml-cuda/convert.cu
ggml/src/ggml-cuda/allreduce.cuh
ggml/src/ggml-cuda/allreduce.cu
ggml/src/ggml-backend-meta.cpp
ggml/include/ggml-backend.h
```

Do **not** assume current upstream and the project's pinned vendor tree are identical.

A concrete example already exists: HI17's 2026-08-11 verification found no `GGML_CUDA_CUBLAS_COMPUTE_TYPE` in the pinned tree, while current upstream now documents and implements that override.

Always distinguish:

```text
project pinned source
current upstream source
installed ROCm headers/libraries
```

All three are versioned realities.

---

# Part I — HI17: BLAS execution-plan tuning

## 3. Main conclusion

The original HI17 design was technically sound, but its plan schema required
correction before durable candidate identities were introduced. That work and
the required parity evidence are complete at `4732f47`; the material below is
retained as historical rationale and is not an open implementation gate.

The current high-level fields:

```text
compute_type
conversion_route
output_route
api_strategy
```

are not precise enough to describe the effective hipBLAS call.

At minimum distinguish:

```text
operand/internal representation
accumulation type
output type
src0 conversion
src1 conversion
API strategy
numerical class
```

This matters because llama.cpp can keep A/B in F16 while changing the GEMM compute/accumulation and result datatype to F32.

---

## 4. Real BLAS call graph

Conceptually:

```text
matmul dispatch
    |
    v
ggml_cuda_mul_mat_cublas(...)
    |
    +-- resolve internal operand representation
    |       F32 / F16 / BF16
    |
    v
ggml_cuda_mul_mat_cublas_impl<operand_type>(...)
    |
    +-- convert src0 if needed
    |
    +-- convert src1 if needed
    |
    +-- configure A/B/C datatypes
    |
    +-- configure GEMM compute/accumulation type
    |
    +-- choose direct destination or temporary C
    |
    +-- choose GEMM API
    |       |
    |       +-- Sgemm
    |       +-- GemmEx
    |       +-- GemmStridedBatchedEx
    |       `-- GemmBatchedEx
    |
    `-- post-convert temporary C to F32 if needed
```

On HIP, llama.cpp's compatibility layer maps CUDA-style BLAS names to hipBLAS in:

```text
ggml/src/ggml-cuda/vendors/hip.h
```

Relevant aliases correspond to:

```text
cublasGemmEx               -> hipblasGemmEx
cublasGemmBatchedEx        -> hipblasGemmBatchedEx
cublasGemmStridedBatchedEx -> hipblasGemmStridedBatchedEx
cublasSgemm                -> hipblasSgemm
```

This is a strong foundation for HI17: the candidate boundaries correspond to real callable library interfaces.

---

## 5. Why `compute_type` is too ambiguous

The llama.cpp template parameter in a call such as:

```cpp
ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(...)
```

selects the internal operand representation and traits.

Those traits determine defaults such as:

```text
A/B datatype
compute/accumulation type
alpha/beta scalar type
conversion functions
```

but the implementation can subsequently alter the result datatype and compute type.

So one field called `compute_type` can accidentally mix:

1. A/B internal storage type;
2. hipBLAS `computeType`;
3. GGML request precision.

Those must not share durable identity.

### Recommended type model

```cpp
enum ggml_hip_blas_operand_type : uint8_t {
    BLAS_OPERAND_NATIVE = 0,
    BLAS_OPERAND_F32,
    BLAS_OPERAND_F16,
    BLAS_OPERAND_BF16,
};

enum ggml_hip_blas_accum_type : uint8_t {
    BLAS_ACCUM_NATIVE = 0,
    BLAS_ACCUM_F16,
    BLAS_ACCUM_F32,
};

enum ggml_hip_blas_output_type : uint8_t {
    BLAS_OUTPUT_NATIVE = 0,
    BLAS_OUTPUT_F16,
    BLAS_OUTPUT_BF16,
    BLAS_OUTPUT_F32,
};

enum ggml_hip_blas_api_strategy : uint8_t {
    BLAS_API_NATIVE_AUTO = 0,
    BLAS_API_GEMMEX_SINGLE,
    BLAS_API_STRIDED_BATCHED,
    BLAS_API_POINTER_BATCHED,
    BLAS_API_LOOPED_GEMMEX,
};
```

Then:

```cpp
struct ggml_hip_blas_plan_v2 {
    uint16_t schema_version;

    uint8_t operand_type;
    uint8_t accum_type;
    uint8_t output_type;

    uint8_t src0_conversion;
    uint8_t src1_conversion;

    uint8_t api_strategy;
    uint8_t numerical_class;
};
```

### Reason

The public GEMM interfaces expose input/result/compute types separately. Candidate identity should describe what is actually passed to the library.

### Validation

Immediately before the library call, capture:

```text
a_type
b_type
c_type
compute_type
api
```

A promoted result requires those effective values to match the requested plan.

---

## 6. Introduce a resolved-call object

Do not build separate native and tuned executors.

Separate:

```text
native resolution
candidate override
eligibility/numerical validation
execution
observation
```

### Proposed representation

```cpp
struct ggml_hip_blas_resolved_call {
    ggml_type operand_type;

    hipDataType a_type;
    hipDataType b_type;
    hipDataType c_type;
    hipblasComputeType_t compute_type;

    ggml_hip_conversion_route src0_conversion;
    ggml_hip_conversion_route src1_conversion;

    ggml_hip_blas_api_strategy api;

    bool uses_src0_temp;
    bool uses_src1_temp;
    bool uses_dst_temp;

    uint64_t src0_temp_bytes;
    uint64_t src1_temp_bytes;
    uint64_t dst_temp_bytes;
};
```

Use the typedefs already exposed by the project's versioned HIP compatibility layer rather than introducing an unnecessary direct dependency on one specific ROCm typedef revision.

### Proposed functions

```cpp
static ggml_hip_blas_resolved_call
resolve_native_blas_call(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * dst);

static bool
apply_blas_plan(
    const ggml_hip_blas_signature & sig,
    const ggml_hip_blas_plan_v2 & requested,
    ggml_hip_blas_resolved_call & resolved,
    ggml_hip_blas_reject_reason * reject);

static void
execute_blas_call(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst,
    const ggml_hip_blas_resolved_call & resolved,
    ggml_hip_blas_observation * obs);
```

Wrapper shape:

```cpp
static void ggml_cuda_mul_mat_cublas(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst,
    const ggml_hip_blas_plan_v2 * plan = nullptr,
    ggml_hip_blas_observation * obs = nullptr) {

    auto resolved =
        resolve_native_blas_call(ctx, src0, src1, dst);

    if (plan != nullptr) {
        ggml_hip_blas_reject_reason reason;

        if (!apply_blas_plan(
                make_blas_signature(...),
                *plan,
                resolved,
                &reason)) {
            record_rejection(reason);
            return;
        }
    }

    execute_blas_call(
        ctx, src0, src1, dst, resolved, obs);
}
```

### Why this seam matters

`native-auto`, `forced-native`, and all tuned candidates use the same final launcher.

That turns forced-native parity into a structural guarantee rather than a convention.

### Forced-native parity proof

For each representative signature:

1. run native resolution;
2. capture the fully resolved call;
3. serialize that configuration as a forced plan;
4. replay through `apply_blas_plan`;
5. verify all final library arguments/API choices match;
6. verify output against native;
7. compare complete-path timing within noise.

A failure blocks tuning.

---

## 7. Numerical class must be derived from the effective call

HI17 is right that numerical-class rejection must happen before ranking.

Do not trust a candidate label.

Classify the resolved call after the candidate has been applied.

Conceptually:

```cpp
static ggml_hip_numerical_class
classify_numerics(
    const ggml_hip_blas_resolved_call & c,
    const ggml_hip_blas_signature & sig) {

    if (sig.prec == GGML_PREC_F32 &&
        c.compute_type != HIPBLAS_COMPUTE_32F) {
        return NUMERICAL_REJECT_STRICT_PRECISION;
    }

    if (c.a_type == HIPBLAS_R_32F &&
        c.b_type == HIPBLAS_R_32F &&
        c.c_type == HIPBLAS_R_32F &&
        c.compute_type == HIPBLAS_COMPUTE_32F) {
        return NUMERICAL_EXACT_BASELINE;
    }

    if (c.compute_type == HIPBLAS_COMPUTE_32F) {
        return NUMERICAL_BACKEND_TOLERANCE;
    }

    return NUMERICAL_REDUCED_PRECISION;
}
```

Use the project's exact standards for the final policy.

The important invariant is:

> **Numerical classification sees the effective API configuration, not the candidate name.**

Reject before launch/ranking:

```cpp
if (!objective_accepts(classify_numerics(...))) {
    return CANDIDATE_INELIGIBLE;
}
```

---

## 8. RDNA3 output path: separate accumulation from conversion

The gfx1100 path identified by HI17 remains one of the strongest early experiments, but direct F32 output changes more than the destination pointer.

The meaningful experiment is:

### Arm A — native F16-style path

```text
A/B internal: F16
accumulation: F16
C: F16 temporary
post conversion: F16 -> F32
```

### Arm B — F32 accumulation with temporary output, if the concrete API supports it

```text
A/B internal: F16
accumulation: F32
C: F16 temporary
post conversion: F16 -> F32
```

This isolates the cost/effect of F32 accumulation.

### Arm C — direct F32 result

```text
A/B internal: F16
accumulation: F32
C: F32 destination
post conversion: none
```

This isolates removal of the output conversion when compared with Arm B.

If Arm B is not supported by the installed hipBLAS combination, record `unsupported`; do not silently reinterpret it.

### Required observations

```text
complete_path_us
src conversion kernels
output conversion kernel present/absent
src temp bytes
dst temp bytes
workspace bytes
effective provider
observed API
```

Rank on complete path, not GEMM-only timing.

---

## 9. Current upstream compute-type override

Current upstream llama.cpp documents:

```text
GGML_CUDA_CUBLAS_COMPUTE_TYPE
```

with values such as:

```text
auto
f16/fp16
bf16
f32/fp32
```

This is useful as:

- evidence that upstream considers the compute-mode choice worth exposing;
- a source implementation to inspect;
- a process-level sanity test.

It is **not** the final per-signature tuning interface because it is process environment state.

Use it, where present, for matched diagnostic processes:

```text
auto
f16
bf16
f32
```

Then implement the explicit per-call plan in the pinned tree.

---

## 10. Conversion routing requires two fields

Use:

```cpp
enum ggml_hip_conversion_route : uint8_t {
    CONVERT_NONE = 0,
    CONVERT_CONTIG_SPECIALIZED,
    CONVERT_GENERIC_STRIDED,
};
```

with:

```cpp
uint8_t src0_conversion;
uint8_t src1_conversion;
```

### Reason

Contiguous conversion is conceptually:

```cpp
converter(src, dst, element_count, stream);
```

while non-contiguous conversion receives tensor geometry and source strides:

```cpp
converter_nc(
    src, dst,
    ne0, ne1, ne2, ne3,
    s1, s2, s3,
    stream);
```

They have different preconditions and their source-type support can differ.

### Initial rule

Treat native conversion route as telemetry first.

Only make two routes compete if both are proven legal for the same source/destination type and concrete layout.

### Eligibility shape

```cpp
static bool can_convert(
    const ggml_tensor * src,
    ggml_type dst_type,
    ggml_hip_conversion_route route) {

    switch (route) {
        case CONVERT_NONE:
            return src->type == dst_type;

        case CONVERT_CONTIG_SPECIALIZED:
            return ggml_is_contiguously_allocated(src)
                && lookup_contig_converter(
                       src->type, dst_type) != nullptr;

        case CONVERT_GENERIC_STRIDED:
            return lookup_nc_converter(
                       src->type, dst_type) != nullptr;
    }

    return false;
}
```

Do not force a route simply to make the catalog symmetric.

---

## 11. API strategy is a valid per-call dimension

A clean candidate enum is:

```cpp
enum ggml_hip_blas_api_strategy : uint8_t {
    BLAS_API_NATIVE_AUTO = 0,
    BLAS_API_GEMMEX_SINGLE,
    BLAS_API_STRIDED_BATCHED,
    BLAS_API_POINTER_BATCHED,
    BLAS_API_LOOPED_GEMMEX,
};
```

Rules:

- **native auto:** exact reference behavior;
- **single GemmEx:** only true single-matrix shapes;
- **strided batched:** only if regular strides can represent the shape;
- **pointer batched:** include pointer array allocation and pointer-building kernel in timing;
- **looped GemmEx:** separate identity if multiple single GEMMs are intentionally tested.

Never compare a batched library call against a hand-loop while attributing only the library kernel.

---

## 12. HI17 telemetry contract

Recommended shape:

```cpp
struct ggml_hip_blas_observation_v1 {
    uint16_t schema_version;

    uint8_t effective_operand_type;
    uint8_t effective_accum_type;
    uint8_t effective_output_type;

    uint8_t src0_conversion;
    uint8_t src1_conversion;

    uint8_t observed_api;

    uint8_t provider_requested;
    uint8_t provider_effective;

    uint64_t src0_temp_bytes;
    uint64_t src1_temp_bytes;
    uint64_t dst_temp_bytes;
    uint64_t workspace_bytes;

    uint64_t provider_solution_id;
};
```

Some provider fields may remain unknown.

Unknown is acceptable diagnostic evidence. False attribution is not.

### Promotion rule

A winner cannot be promoted when a field required to prove the requested plan is unknown or mismatched.


---

## 13. Three BLAS tuning layers

Keep these separate:

| Layer | Controls | Scope | Recommended owner |
|---|---|---:|---|
| llama.cpp execution plan | operand type, accumulation, output, conversion, GEMM API | per call | HI17 |
| rocBLAS backend policy | Tensile vs preferred hipBLASLt | process/library | HI49-style arm |
| provider solution | rocBLAS solution index / hipBLASLt algorithm | potentially per call | later provider extension |

This separation prevents process-global policy from contaminating per-signature identity.

---

## 14. rocBLAS backend policy

Official rocBLAS documentation exposes:

```text
ROCBLAS_USE_HIPBLASLT
ROCBLAS_USE_HIPBLASLT_BATCHED
```

Broadly:

```text
unset -> automatic
0     -> Tensile
1     -> prefer hipBLASLt, with fallback permitted
```

### Rule

These are execution context/process arms until an explicit safe per-call provider-selection API is demonstrated.

Therefore:

```text
DO NOT put them in per-signature BLAS stable names
DO record them in build/runtime context
DO record effective provider when it can be established
```

A requested hipBLASLt preference is not proof that hipBLASLt actually executed.

---

## 15. Deeper rocBLAS solution tuning

rocBLAS exposes beta APIs that can enumerate solution indices capable of solving a concrete GEMM, including ordinary, batched and strided-batched forms.

This is a **later stage**, not first-pass HI17.

Use only if:

1. Phase 0 proves BLAS is runtime-significant;
2. ordinary plan decomposition has plateaued;
3. the version namespace invalidates solution IDs across incompatible rocBLAS builds.

Conceptually:

```cpp
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>

rocblas_int count = 0;

rocblas_gemm_ex_get_solutions(
    handle,
    ...problem...,
    nullptr,
    &count);

std::vector<rocblas_int> solutions(count);

rocblas_gemm_ex_get_solutions(
    handle,
    ...same problem...,
    solutions.data(),
    &count);
```

Then benchmark only reported valid solutions.

### Risk

AMD labels these APIs beta. Solution indices are not safe durable identities across arbitrary library revisions.

---

## 16. Deeper hipBLASLt algorithm tuning

hipBLASLt exposes extension APIs to enumerate algorithms and test support/workspace.

Conceptually:

```cpp
std::vector<hipblasLtMatmulHeuristicResult_t> algos;

hipblaslt_ext::getAllAlgos(
    handle,
    gemm_type,
    opA,
    opB,
    typeA,
    typeB,
    typeC,
    typeD,
    compute_type,
    algos);

for (auto & result : algos) {
    size_t workspace = 0;

    if (hipblaslt_ext::matmulIsAlgoSupported(
            handle,
            matmul_desc,
            alpha,
            A_desc,
            B_desc,
            beta,
            C_desc,
            D_desc,
            result.algo,
            workspace) == HIPBLAS_STATUS_SUCCESS) {
        benchmark(result.algo, workspace);
    }
}
```

This can eventually move the project from:

```text
prefer hipBLASLt
```

to:

```text
algorithm N is empirically best for this exact gfx1100 problem
```

but only after provider control, telemetry and versioned identity are stable.

---

## 17. Profiling HI17 with ROCprofiler

`rocprofv3` can trace HIP calls, kernels, memory copies and marker ranges. Newer versions also expose broader runtime/system/RCCL tracing.

Development example:

```bash
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --memory-copy-trace \
  --marker-trace \
  -- ./build/bin/llama-bench <args>
```

Depending on installed version:

```bash
rocprofv3 --runtime-trace -- ./build/bin/llama-bench <args>
```

can provide a useful aggregate trace.

### Candidate markers

In a development build:

```cpp
roctxRangePushA(candidate_stable_name);

execute_blas_call(...);

roctxRangePop();
```

Inspect whether the marker contains:

```text
input conversion kernel(s)
pointer construction kernel if applicable
GEMM library kernel(s)
output conversion kernel if applicable
copies/allocations
```

Use profiling to prove path identity. Use uninstrumented runs for promotion timing.

---

## 18. HI17 time×calls prioritization

Do not equate:

```text
19 of 92 signatures
```

with 20% of runtime.

Collect per signature:

```text
call_count
representative complete_path_us
total attributed time
```

Use:

```text
weighted_cost ~= calls × complete-path time
```

to rank engineering effort.

One signature may dominate the 19 BLAS signatures, or all 19 may be minor.

---

## 19. HI17 implementation stages

### BLAS-0 — telemetry only

Record native:

```text
operand type
A/B/C datatype
accumulation type
src0/src1 conversion
API
temporary bytes
precision request
architecture
provider environment
```

**Exit gate:** the effective native plan is reconstructable from observations.

### BLAS-1 — resolved-call seam and forced-native parity

Introduce:

```text
resolve_native_blas_call
apply_blas_plan
execute_blas_call
```

but enumerate only native and forced-native.

**Exit gate:** forced-native reproduces native API, types, conversions, output and timing within noise.

### BLAS-2 — gfx1100 accumulation/output experiment

Test legal combinations separating accumulation precision from output conversion.

**Exit gate:** effective path is proven for every arm.

### BLAS-3 — operand/internal type

Expose F16/BF16/F32 only where converters, numerical class and API support permit.

**Exit gate:** strict precision can never enumerate an inadmissible reduced-precision plan.

### BLAS-4 — conversion route

Only enumerate alternatives where both are legal.

### BLAS-5 — API strategy

Native, single, strided, pointer-batched, and looped single where justified.

**Exit gate:** observed API matches requested API.

### BLAS-6 — provider/process arms

Compare automatic/Tensile/hipBLASLt-preference behavior as matched processes.

### BLAS-7 — provider solution tuning

Only if measured remaining BLAS runtime justifies the complexity.

---

# Part II — HI18: SPLIT_REDUCE

> **Status pointer (2026-08-20, PH01 review triage):** This Part II predates
> HI18's implementation progress. §38's stage checklist is the *original
> decomposition* — read it as plan, not current status. The authoritative
> current state is `HI18.md` (Validation/Notes), summarised here:
> - **REDUCE-0 (telemetry):** done — reduction-signature contract + strict
>   offline loader (37/37 suite), unknown peer access fail-closed for promotion.
> - **REDUCE-1 (explicit AUTO/RCCL/META per-call seam):** done — implemented
>   without mutating the shared comm context; Brutus tensor-split RCCL and
>   forced-meta canaries emit strict-loader-clean signature evidence with
>   requested/effective telemetry.
> - **REDUCE-2 (valid timer):** in progress, exit gate **not met**. The earlier
>   "complete-path timing proven at 57e9c9f" claim was superseded by a
>   timing-status correction: current `elapsed_us` is a host-side
>   submission/control interval, **not** a Standards 7.1 complete-path timing.
>   Remaining work is the `GGML_HIP_REDUCE_TIMING=sync` device-synchronised
>   producer path (`timing_mode=device_synchronized`) before the loader accepts
>   promotable timing evidence.
> - **REDUCE-3 … REDUCE-8:** not started (correctness/microbenchmark matrix,
>   end-to-end holdout, HIP-internal spike/provider/tuning, RCCL policy).
>
> Do not read the future-tense exit gates below as already-passed acceptance;
> they remain the design intent for each stage.

## 20. Main conclusion

HI18's execution-plan idea is good, but the current HIP candidate set needs correction.

On current upstream llama.cpp:

```text
RCCL is real
meta-backend generic/butterfly fallback is real
internal custom AllReduce is stubbed out on HIP
```

Unless the project's branch has independently ported the internal provider, do not tune `(internal, none)` as if it executes a real reduction.

At the same time, the upstream custom CUDA AllReduce reveals a potentially valuable **separate AMD porting project** because it is designed around two PCIe GPUs and host staging.

---

## 21. Real allreduce control flow

```text
meta backend sees a partial tensor
        |
        v
backend-specific comm_allreduce callback
        |
        v
ggml_backend_cuda_comm_allreduce_tensor(...)
        |
        v
comm_ctx->try_allreduce(...)
        |
        +--> RCCL provider -> true
        |
        +--> internal provider -> true/false
        |
        `--> decline -> false
                    |
                    v
meta backend generic fallback
                    |
                    `--> asynchronous copies + ADD butterfly
```

### Important telemetry consequence

The HIP/CUDA callback does not itself perform the generic butterfly when it returns false.

Therefore inside the HIP callback record:

```text
backend-specific provider declined
handoff to meta
```

not:

```text
effective algorithm = butterfly
```

The latter becomes true only when `ggml-backend-meta.cpp` actually runs its fallback.

---

## 22. Interfaces and source locations

### CUDA/HIP communication dispatch

Inspect:

```text
ggml/src/ggml-cuda/ggml-cuda.cu
```

Search:

```bash
rg -n \
  "try_allreduce|comm_allreduce|GGML_CUDA_ALLREDUCE|comm_init_" \
  ggml/src/ggml-cuda/ggml-cuda.cu
```

### Internal AllReduce

Inspect:

```text
ggml/src/ggml-cuda/allreduce.cuh
ggml/src/ggml-cuda/allreduce.cu
```

Search:

```bash
rg -n \
  "ggml_cuda_ar_pipeline_init|ggml_cuda_ar_allreduce|GGML_USE_HIP" \
  ggml/src/ggml-cuda/allreduce.*
```

### Generic fallback

Inspect:

```text
ggml/src/ggml-backend-meta.cpp
```

Search:

```bash
rg -n \
  "allreduce_fallback|backend_allreduce_success|comm_allreduce" \
  ggml/src/ggml-backend-meta.cpp
```

That is the correct point to record actual generic fallback execution.

---

## 23. Do not mutate `comm_ctx->try_allreduce` per candidate

Even though provider selection is represented by a function pointer, avoid:

```cpp
ctx->try_allreduce = selected;
ctx->try_allreduce(ctx, tensors);
```

on every invocation.

That creates shared mutable state.

Prefer direct per-call dispatch:

```cpp
static bool ggml_backend_cuda_comm_allreduce_tensor(
    void * comm_ctx_v,
    ggml_tensor ** tensors) {

    if (comm_ctx_v == nullptr) {
        return false;
    }

    auto * ctx =
        static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);

    const split_reduce_plan_v1 * plan =
        current_split_reduce_plan();

    if (plan == nullptr ||
        plan->preferred_algorithm == REDUCE_AUTO) {
        return ctx->try_allreduce(ctx, tensors);
    }

    switch (plan->preferred_algorithm) {
        case REDUCE_RCCL:
            return try_rccl(ctx, tensors);

        case REDUCE_INTERNAL:
            return try_internal(ctx, tensors);

        case REDUCE_META:
            note_backend_handoff(HANDOFF_FORCED_META);
            return false;
    }

    GGML_ABORT("invalid reduction plan");
}
```

Use the exact existing wrapper names and compile guards in the pinned tree.

---

## 24. Explicit fallback chains

Represent fallback policy as requested data rather than by changing the shared provider.

Example:

```cpp
struct split_reduce_plan_v1 {
    uint16_t schema_version;
    uint8_t n_algorithms;
    uint8_t algorithms[3];
};
```

Potential plans:

```text
[RCCL]
[META]
[RCCL, META]
[AUTO]
```

Add INTERNAL only when a real HIP implementation exists.

Executor:

```cpp
static bool execute_reduce_plan(
    ggml_backend_cuda_comm_context * ctx,
    ggml_tensor ** tensors,
    const split_reduce_plan_v1 & plan,
    split_reduce_observation_v1 & obs) {

    for (uint8_t i = 0; i < plan.n_algorithms; ++i) {
        const auto alg = plan.algorithms[i];

        if (alg == REDUCE_META) {
            obs.handoff_reason =
                HANDOFF_REQUESTED_META;
            return false;
        }

        const bool ok =
            try_reduce_algorithm(ctx, tensors, alg);

        obs.attempts[obs.n_attempts++] = {
            alg,
            ok
        };

        if (ok) {
            obs.backend_specific_success = true;
            obs.effective_algorithm = alg;
            return true;
        }
    }

    obs.backend_specific_success = false;
    obs.handoff_reason =
        HANDOFF_CHAIN_EXHAUSTED;

    return false;
}
```

Instrument meta fallback separately.

---

## 25. Current HIP reality: internal provider is unavailable upstream

Current upstream's real custom internal AllReduce implementation is excluded from HIP/MUSA builds.

The HIP side effectively supplies:

```text
pipeline_init -> nullptr
pipeline_free -> no-op
allreduce     -> false
```

Therefore first verify the project's pinned tree:

```bash
rg -n \
  "GGML_USE_HIP|ggml_cuda_ar_pipeline_init|ggml_cuda_ar_allreduce" \
  vendor/llama.cpp/ggml/src/ggml-cuda/allreduce.cu
```

If the same stubs are present, enumerate:

```text
internal -> unavailable_unported
```

### Initial HIP candidate set

```text
AUTO/upstream
RCCL/no fallback
META/no fallback
```

Fallback-enabled plans come only after requested/effective observations work.

---

## 26. Correct multi-device timing

RCCL work is stream-enqueued. A host call returning is not enough to establish semantic completion on all participating devices.

Use one worker per device.

Worker:

```cpp
void worker(int rank) {
    hipSetDevice(device[rank]);

    ready_barrier.arrive_and_wait();

    enqueue_selected_collective(rank);

    hipEventRecord(
        done_event[rank],
        stream[rank]);

    hipEventSynchronize(
        done_event[rank]);

    complete_barrier.arrive_and_wait();
}
```

Coordinator:

```cpp
ready_barrier.arrive_and_wait();

const auto start =
    std::chrono::steady_clock::now();

complete_barrier.arrive_and_wait();

const auto stop =
    std::chrono::steady_clock::now();

obs.raw_complete_path_us =
    to_us(stop - start);
```

Record separately:

```text
raw_complete_path_us
orchestration_us
estimated_collective_us
```

Rank primarily by raw complete-path time.

### Why not subtract orchestration as the authoritative score?

Subtracting a separately calibrated small cost can add variance and distort very short measurements. All candidates share the same orchestration, so much of it cancels in comparisons.

### HIP events

Keep per-device event measurements as diagnostics.

Never:

```text
gpu0_event_us + gpu1_event_us = wall latency
```

and do not assume cross-device event timestamp domains are directly comparable.

---

## 27. Prove the timer with a deliberately imbalanced run

Before trusting any winner, deliberately make one participant slower.

Possible mechanisms:

```text
extra kernel on rank 1
controlled stream wait
controlled host submission delay
```

The complete-path measurement must follow the slow participant.

A one-stream measurement should undercount it.

This test is mandatory because timing errors are otherwise silent.

---

## 28. Process containment for deadlocks

Run dangerous collective candidates in owned child processes.

```text
parent tuner
    |
    `-- child candidate process
            |
            `-- HIP/RCCL experiment
```

Parent behavior:

```cpp
pid = spawn_candidate_process(candidate);

if (!wait_with_timeout(pid, timeout)) {
    kill(pid, SIGKILL);
    reap(pid);

    record_outcome(
        candidate,
        OUTCOME_HANG);
}
```

The child owns:

```text
communicators
streams
events
temporary buffers
test tensors
```

This is safer than relying only on a watchdog thread inside a process that may be blocked in the GPU runtime.


---

## 29. The upstream internal CUDA AllReduce is highly relevant to AMD

The custom internal provider is not merely another NCCL wrapper. It is a PCIe-oriented reduction design.

Broadly:

### Small-message path

```text
GPU0 / GPU1
    |
    +-- write local contribution into mapped pinned host memory
    +-- system-wide memory fence
    +-- publish arrival token
    +-- wait for peer token
    +-- read peer mapped host memory
    `-- reduce locally
```

### Large-message path

```text
D2H async chunks
    |
host-pinned staging
    |
H2D peer contribution
    |
device-side add
```

The implementation also contains:

```text
chunking
event/resource pools
copy-engine pipelining
BF16 wire representation for some large F32 reductions
thresholds between paths
```

This design is relevant to dual consumer GPUs without working P2P because it deliberately routes data through host-visible PCIe staging.

That makes a HIP port worth a **capability spike** if HI18 Phase 0 shows SPLIT_REDUCE is runtime-significant.

---

## 30. HIP capability proof before porting internal AllReduce

Current HIP documentation exposes relevant primitives including:

```text
hipHostAllocMapped
hipHostAllocPortable
hipHostGetDevicePointer
__threadfence_system()
```

Their existence justifies an experiment; it does not prove the complete CUDA protocol is correct on gfx1100.

Create a standalone proof outside the production tuner.

### 30.1 Allocation/mapping probe

Conceptually:

```cpp
void * host = nullptr;

HIP_CHECK(hipHostMalloc(
    &host,
    bytes,
    hipHostMallocMapped |
    hipHostMallocPortable));

void * device_alias[2] = {};

for (int d = 0; d < 2; ++d) {
    HIP_CHECK(hipSetDevice(d));

    HIP_CHECK(hipHostGetDevicePointer(
        &device_alias[d],
        host,
        0));
}
```

Use the exact allocation flags supported by the installed ROCm version.

If the target release requires coherent/fine-grained memory settings for the desired system-fence semantics, put that requirement into the experiment namespace.

### 30.2 Writer kernel

```cpp
__global__ void writer(
    uint32_t * data,
    uint32_t * signal) {

    const int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    data[i] = expected_value(i);

    __threadfence_system();

    if (i == 0) {
        *signal = 1;
        __threadfence_system();
    }
}
```

### 30.3 Reader kernel

```cpp
__global__ void reader(
    const uint32_t * data,
    const uint32_t * signal,
    uint32_t * result) {

    if (threadIdx.x == 0) {
        while (
            *(volatile const uint32_t *) signal
                != 1) {
            hip_ar_relax();
        }
    }

    __syncthreads();
    __threadfence_system();

    const int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    result[i] = data[i];
}
```

### 30.4 Stress matrix

Run:

```text
GPU0 -> GPU1
GPU1 -> GPU0
simultaneous bidirectional exchange
multiple payload sizes
many iterations
unrelated load on other streams
repeated allocation/free cycles
```

Verify every result.

Any stale-read, permanent-spin or ordering failure blocks the port.

---

## 31. Spin/yield behavior is an architecture experiment

The CUDA implementation uses a sleep/yield mechanism while waiting for peer arrival.

AMD code can expose architecture-specific sleep instructions through compiler builtins, but do not assume one form is automatically safe or optimal for gfx1100.

A probe can start with:

```cpp
__device__ __forceinline__
void hip_ar_relax() {
#if defined(__HIP_PLATFORM_AMD__)
    __builtin_amdgcn_s_sleep(1);
#endif
}
```

Compile for the exact target:

```bash
hipcc \
  --offload-arch=gfx1100 \
  -O3 \
  -c ar_sleep_probe.cpp
```

Inspect generated ISA/code object with the LLVM/ROCm tooling installed on the system.

Benchmark at least:

```text
pure busy poll
sleep 1
sleep 2
sleep 4
```

Measure:

```text
signal-to-observe latency
hang rate
effect on unrelated work
GPU utilization
```

Do not promote a wait strategy until a soak test passes.

---

## 32. If a HIP internal provider works, make it a separate tuning subsystem

Do not repeat HI17's original problem by calling an entire custom AllReduce implementation simply `internal`.

A later plan could look like:

```cpp
struct hip_internal_ar_plan_v1 {
    uint16_t schema_version;

    uint8_t transport;
    uint8_t wire_type;
    uint8_t wait_strategy;

    uint32_t kernel_blocks;
    uint32_t chunk_bytes;

    uint32_t copy_threshold;
    uint32_t bf16_threshold;
};
```

Potential values:

```text
transport:
    mapped_host_kernel
    copy_engine

wire_type:
    f32
    bf16
    f16 where numerically legal
```

Possible tuning dimensions:

```text
small/large route threshold
chunk size
block count
wait strategy
wire type threshold
pipeline depth
```

Do not create this search space before the base provider proves correctness and runtime value.

---

## 33. RCCL has its own deeper tuning layer

At the llama.cpp level, `RCCL` is a good provider identity.

Internally RCCL still chooses collective algorithm/protocol/channel configuration.

### 33.1 Process-level controls

RCCL exposes NCCL-compatible environment tuning controls.

Treat these as matched process/library experiments unless a per-call interface explicitly owns the choice.

### 33.2 RCCL tuner plugin

AMD documents an external tuner-plugin API.

The plugin's `getCollInfo()` is invoked for collective calls and can influence:

```text
algorithm
protocol
number of channels
```

based on inputs including collective type and message size.

This creates a later architecture:

```text
SPLIT_REDUCE
      |
      +-- provider choice = RCCL
      |
      `-- RCCL tuner plugin
              |
              `-- RCCL algorithm /
                  protocol /
                  channels
```

Conceptual callback:

```cpp
ncclResult_t getCollInfo(
    ...,
    ncclFunc_t coll_type,
    size_t nbytes,
    int * algorithm,
    int * protocol,
    int * n_channels,
    ...) {

    const auto * winner =
        lookup_rccl_winner(
            topology_digest,
            coll_type,
            nbytes);

    if (winner == nullptr) {
        // Leave RCCL defaults in control.
        return ncclSuccess;
    }

    *algorithm = winner->algorithm;
    *protocol = winner->protocol;
    *n_channels = winner->channels;

    return ncclSuccess;
}
```

Use the exact interface version from the installed RCCL headers.

### Why this is not Phase 1

Do not build an RCCL plugin until Phase 0 proves:

```text
SPLIT_REDUCE has significant runtime weight
AND
RCCL policy variance is worth controlling
```

The plugin callback itself is per-collective overhead, so lookup must be cheap.

---

## 34. HI18 signature and context

A useful signature shape is:

```cpp
struct split_reduce_signature_v1 {
    uint16_t schema_version;

    uint8_t element_type;
    uint8_t device_count;

    uint64_t element_count;
    uint64_t message_bytes;

    uint64_t topology_digest;
    uint64_t split_digest;
};
```

Important context to record:

```text
device architecture(s)
device set
PCIe topology
peer-access matrix
tensor split
RCCL version
ROCm/HIP version
stream configuration
process-level collective settings
```

Do not use arbitrary device ordinal as the hardware identity.

---

## 35. HI18 observation model

A stronger observation than only `fallback_depth` is:

```cpp
struct split_reduce_attempt_v1 {
    uint8_t algorithm;
    uint8_t result;
};

struct split_reduce_observation_v1 {
    uint16_t schema_version;

    uint8_t requested_algorithm;
    uint8_t effective_algorithm;

    uint8_t n_attempts;
    split_reduce_attempt_v1 attempts[3];

    uint8_t handoff_reason;
    uint8_t meta_fallback_executed;

    uint64_t message_bytes;
    uint64_t peer_access_mask;

    double raw_complete_path_us;
    double orchestration_us;
    double estimated_collective_us;
};
```

Useful attempt outcomes:

```text
success
unsupported
init_unavailable
runtime_failure
declined_to_meta
hang
watchdog_kill
correctness_failure
```

---

## 36. Correctness matrix for SPLIT_REDUCE

For each candidate:

1. generate deterministic per-device inputs;
2. compute a CPU/high-precision reference;
3. run the collective;
4. synchronize all participants;
5. copy every device result back;
6. compare every device against reference;
7. compare against native production policy where useful.

Test:

```text
F32
F16
BF16 where supported
small messages
threshold-boundary messages
large messages
inactive/zero-contributing shards
uneven split geometry where legal
```

A performance winner that changes numerical behavior cannot silently replace a stricter class.

---

## 37. End-to-end holdout

A microbenchmark winner is not enough.

Run the actual tensor-split workload, covering at least:

```text
representative prompt processing
representative token generation
dual-gfx1100 tensor split
```

If:

```text
microbenchmark: +20%
end-to-end: -2%
```

the microbenchmark is not modeling the real path well enough to promote the candidate.

---

## 38. HI18 implementation stages

### REDUCE-0 — telemetry only

Record:

```text
construction-time selected provider
RCCL init success/failure
internal availability
backend-specific attempt
handoff to meta
actual meta fallback execution
message size/type
device set
peer matrix
```

**Exit gate:** every production reduction can be explained as RCCL or meta and why.

### REDUCE-1 — explicit existing-provider candidates

Initially on HIP:

```text
AUTO
RCCL
META
```

**Exit gate:** requested/effective behavior is provable.

### REDUCE-2 — valid timer

Implement multi-device worker/barrier/event timing.

**Exit gate:** deliberately imbalanced validation passes.

### REDUCE-3 — correctness and microbenchmark matrix

**Exit gate:** stable winner direction exists only where correctness passes.

### REDUCE-4 — end-to-end holdout

**Exit gate:** microbenchmark and workload directions agree within the declared interval or discrepancies are explained.

### REDUCE-5 — HIP internal capability spike

Separate item/branch.

Prove mapped-host access, visibility, signalling, bounded waiting and soak stability.

### REDUCE-6 — HIP internal provider

Only if the spike passes.

Start with one conservative route and no internal tuner.

### REDUCE-7 — internal provider tuning

Only then enumerate transport/wire/threshold/chunk/wait choices.

### REDUCE-8 — RCCL policy tuning

Only if measured remaining runtime warrants it.

---

# Part III — Cross-cutting engineering rules

## 39. Build/runtime namespace

Capture at least:

```text
llama.cpp vendor revision
project patchset revision
ROCm version
HIP version
hipBLAS version
rocBLAS version
hipBLASLt version
RCCL version
gfx architecture
build flags
provider environment variables
```

For BLAS also capture, when set:

```text
ROCBLAS_USE_HIPBLASLT
ROCBLAS_USE_HIPBLASLT_BATCHED
```

Provider solution IDs must be invalidated across incompatible library versions.

---

## 40. Requested and effective are distinct records

Prefer:

```cpp
requested_plan
effective_plan
observation
```

over assuming equality.

Examples:

```text
requested hipBLASLt preference
effective Tensile fallback

requested strided GEMM
effective different API branch

requested RCCL
RCCL init failure
effective meta fallback

requested internal
HIP implementation unavailable
effective meta fallback
```

A requested/effective mismatch is diagnostic evidence unless the candidate semantics explicitly include that fallback.

---

## 41. Hard eligibility before launch

### BLAS

Reject before execution for:

```text
strict precision incompatibility
missing converter
layout incompatibility
invalid batch/stride representation
unsupported datatype combination
temporary/workspace limit violation
```

### Reduce

Reject before execution for:

```text
RCCL not built
communicator unavailable
internal unimplemented
unsupported device count/type
topology mismatch
```

Routine ineligibility should not be discovered by crashing or silently falling back.

---

## 42. Complete-path timing

For both families:

```text
candidate cost =
    preparation
  + conversions
  + attributable allocation/workspace
  + helper kernels
  + library operation
  + communication
  + synchronization needed for semantic completion
  + post-conversion
```

Examples that must not be hidden:

```text
pointer-array construction for batched GEMM
F16 temporary -> F32 conversion
BF16 reduction conversion
meta fallback copies and ADD graphs
```

---

## 43. Workspace accounting

For pooled allocations, record where possible:

```text
requested temporary bytes
pool-returned/reserved bytes
```

Do not let a candidate that allocates large temporary buffers appear to consume zero workspace simply because the allocator reused a pool block.

If HI64 changes allocation accounting semantics, version the BLAS evidence accordingly.

---

## 44. Source-archeology workflow

Pin the vendor revision:

```bash
cd vendor/llama.cpp
git rev-parse HEAD
git status --short
```

BLAS:

```bash
rg -n \
  "ggml_cuda_mul_mat_cublas|prefer_f32_output|GemmEx|GemmBatched|GemmStrided" \
  ggml/src/ggml-cuda
```

Conversions:

```bash
rg -n \
  "ggml_get_to_(fp16|fp32|bf16).*cuda" \
  ggml/src/ggml-cuda
```

AllReduce:

```bash
rg -n \
  "try_allreduce|comm_allreduce|allreduce_fallback|GGML_CUDA_ALLREDUCE" \
  ggml
```

History:

```bash
git log -S'GGML_CUDA_CUBLAS_COMPUTE_TYPE' -- \
    ggml/src/ggml-cuda/ggml-cuda.cu

git log -S'ggml_cuda_ar_pipeline_init' -- \
    ggml/src/ggml-cuda

git blame -L <start>,<end> \
    ggml/src/ggml-cuda/ggml-cuda.cu
```

Inspect installed APIs:

```bash
rg -n "hipblasGemmEx" /opt/rocm/include
rg -n "rocblas_gemm_ex_get_solutions" /opt/rocm/include
rg -n "getAllAlgos" /opt/rocm/include
rg -n "ncclTuner" /opt/rocm/include
```

Inspect linked libraries:

```bash
ldd ./build/bin/llama-server |
    grep -Ei 'hip|rocblas|hipblas|rccl'
```

Toolchain:

```bash
hipcc --version
rocminfo | head -n 80
```

---

## 45. Profiling workflow

Path-proof run:

```bash
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --memory-copy-trace \
  --marker-trace \
  -- ./build/bin/llama-bench <args>
```

Where supported:

```bash
rocprofv3 --runtime-trace -- ./build/bin/llama-bench <args>
```

and newer ROCprofiler versions can expose RCCL trace collection.

Do not use heavy profiler runs as final performance samples. Trace to prove path identity, then benchmark with tracing disabled.

---

## 46. Suggested repository boundaries

A clean project-owned split is:

```text
src/ggml/src/ggml-cuda/
    hip-autotune-types.h
    hip-autotune-dispatch.cu
    hip-autotune-tuner.cu

vendor/llama.cpp/ggml/src/ggml-cuda/
    ggml-cuda.cu
    allreduce.cu        # only if HIP internal port is undertaken

tools/bigcherry/
    autotune_catalog.py
    autotune_schema.py

sql/
    dispatch-db.sql
```

Keep vendor patches narrow:

```text
resolve/accept plan
execute
emit observation hook
```

Do not put tuner database/catalog policy into vendor code.

---

## 47. Test matrices

### HI17

Test at minimum:

```text
gfx1100
gfx1201 where available

single GEMM
strided-batched
pointer-batched

contiguous source layouts
real non-contiguous layouts

quantized -> F16
quantized -> BF16
quantized -> F32 where legal

GGML_PREC_DEFAULT
GGML_PREC_F32
```

Correctness:

```text
native
forced-native
candidate
CPU/high-precision reference where practical
```

Performance:

```text
warm-up
counterbalanced candidate order
repeated samples
raw samples retained
end-to-end holdout
```

### HI18

Test at minimum:

```text
dual gfx1100
RCCL enabled
RCCL unavailable/failure path
forced meta

small/boundary/large messages
supported F32/F16/BF16 cases
balanced split
deliberately imbalanced timer proof
hard-kill/restart recovery for experimental providers
```

---

## 48. Stop conditions

### HI17 blockers

Do not mark complete if:

- forced-native does not reproduce native effective path;
- numerical gating can admit an inadmissible reduced-precision candidate;
- requested API cannot be proven effective;
- temporary/workspace attribution is materially wrong;
- promoted winners lack required effective-path evidence;
- process-global provider settings enter per-signature identity.

### HI18 blockers

Do not mark complete if:

- timing only proves one device completed;
- a deadlock can escape without bounded process cleanup;
- requested/effective algorithm cannot be reconciled;
- meta butterfly is claimed without observing meta execution;
- the unavailable HIP internal stub is treated as a measured algorithm;
- topology/library mismatch can reuse a stale winner;
- end-to-end holdout contradicts the microbenchmark without explanation.

---

## 49. Recommended near-term order

1. **HI17 telemetry-only resolved-call observation.**
2. **HI18 telemetry-only provider/handoff observation.**
3. **HI17 resolved-call seam + forced-native parity.**
4. **HI17 gfx1100 accumulation/output experiment.**
5. **HI18 valid RCCL-vs-meta timing harness.**
6. **HI18 RCCL-vs-meta measurements on the actual dual-XTX workload.**
7. **HI17 API-strategy tuning on the highest time×calls BLAS signatures.**
8. **HIP internal-AllReduce capability spike if HI18 cost justifies it.**
9. **Provider-specific BLAS solution tuning only after ordinary HI17 choices plateau.**
10. **RCCL tuner-plugin work only after existing-provider results justify it.**

The engineering priority after Phase 0 should be driven by:

```text
calls × complete-path cost
```

not signature count or theoretical cleverness.


---

# 50. Reference appendix

## 50.1 Plan items

### HI17

**Decompose the opaque BLAS candidate into a controlled execution plan**

Plan-derived requirements preserved here:

- one BLAS family rather than artificial DQ_BLAS/BLAS entry points;
- numerical class and precision gating;
- requested/effective distinction;
- conversion/output cost attribution;
- API strategy;
- complete-path timing;
- forced-native parity;
- provider/version context;
- Phase-0 telemetry before tuning.

### HI18

**Tune the multi-device allreduce algorithm (SPLIT_REDUCE)**

Plan-derived requirements preserved here:

- independent reduction signature;
- topology-aware evidence;
- requested/effective algorithm;
- fallback telemetry;
- complete multi-device timing;
- no-fallback measurements before resilience chains;
- correctness reference;
- watchdog/owned cleanup;
- tensor-split end-to-end holdout.

---

## 50.2 llama.cpp current upstream source

Repository:

- https://github.com/ggml-org/llama.cpp

Important files:

- BLAS and communication dispatch:  
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu

- HIP compatibility aliases:  
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/hip.h

- Conversion interface:  
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/convert.cuh

- Conversion implementation:  
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/convert.cu

- Internal AllReduce interface:  
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/allreduce.cuh

- Internal AllReduce implementation/HIP stubs:  
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/allreduce.cu

- Meta-backend generic reduction fallback:  
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend-meta.cpp

- Meta/scheduler API definitions:  
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h

- Current build/runtime documentation, including `GGML_CUDA_CUBLAS_COMPUTE_TYPE`:  
  https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md

### Useful llama.cpp history

Compute-type override history:

- `f2c0dfb7394b3abb5a5afd1c2a94f621bb64236f`
- https://github.com/ggml-org/llama.cpp/commit/f2c0dfb7394b3abb5a5afd1c2a94f621bb64236f

Internal CUDA AllReduce:

- `f3c3e0e9a087835639733485b8900b195ba4ca47`
- https://github.com/ggml-org/llama.cpp/commit/f3c3e0e9a087835639733485b8900b195ba4ca47

Backend-agnostic tensor parallelism / HIP RCCL work:

- `d6f3030047f85a98b009189e76f441fe818ea44d`
- https://github.com/ggml-org/llama.cpp/commit/d6f3030047f85a98b009189e76f441fe818ea44d

SYCL custom tensor-parallel reduction example:

- `e9fb3b3fc0300501179b0ce61e907be767e8c859`
- https://github.com/ggml-org/llama.cpp/commit/e9fb3b3fc0300501179b0ce61e907be767e8c859

The commit history is useful design evidence, but the project must still verify its own pinned vendor revision before implementation.

---

## 50.3 AMD hipBLAS

Official hipBLAS API:

- https://rocm.docs.amd.com/projects/hipBLAS/en/latest/functions.html
- https://rocm.docs.amd.com/projects/hipBLAS/en/latest/reference/hipblas-api-functions.html

Relevant interfaces:

```text
hipblasGemmEx
hipblasGemmBatchedEx
hipblasGemmStridedBatchedEx
hipblasSgemm
hipblasSetStream
```

Why they matter:

- ordinary, pointer-batched and strided-batched GEMM are distinct public interfaces;
- extended GEMM interfaces expose datatype/compute choices needed by HI17.

Always inspect the headers used by the installed build:

```bash
rg -n "hipblasGemmEx" /opt/rocm/include
```

---

## 50.4 AMD rocBLAS

### Environment/backend policy

Official environment-variable reference:

- https://rocm.docs.amd.com/projects/rocBLAS/en/develop/reference/env-variables.html

Relevant settings:

```text
ROCBLAS_USE_HIPBLASLT
ROCBLAS_USE_HIPBLASLT_BATCHED
```

The documentation explicitly permits fallback from preferred hipBLASLt to Tensile. Therefore a preference setting is not effective-provider proof.

### Design notes

- https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/what-is-rocblas.html
- https://rocm.docs.amd.com/projects/rocBLAS/en/docs-7.0.0/conceptual/rocblas-design-notes.html

These describe rocBLAS use of Tensile and hipBLASLt.

### Beta solution enumeration

- https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/beta-features.html

Relevant APIs include:

```text
rocblas_gemm_ex_get_solutions
rocblas_gemm_batched_ex_get_solutions
rocblas_gemm_strided_batched_ex_get_solutions
```

Why they matter:

- they can expose valid provider solution IDs for a concrete problem;
- they are beta and therefore require strict version namespacing.

---

## 50.5 AMD hipBLASLt

Extension API:

- https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/reference/ext-reference.html

Relevant functions:

```text
hipblaslt_ext::getAllAlgos
hipblaslt_ext::matmulIsAlgoSupported
```

These provide later-stage algorithm enumeration and workspace/support checking.

Environment variables:

- https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/env-variables.html

Useful categories include:

```text
logging
profiling markers
offline tuning
tuning override files
workspace limits
```

These are not automatically per-signature-safe controls.

---

## 50.6 AMD RCCL

### Tuner plugin

- https://rocm.docs.amd.com/projects/rccl/en/docs-7.1.1/how-to/using-rccl-tuner-plugin-api.html

Why it matters:

- `getCollInfo()` is called for collective operations;
- the plugin can influence collective algorithm, protocol and channel count;
- it creates a later per-collective policy layer beneath llama.cpp's RCCL provider choice.

RCCL documentation root:

- https://rocm.docs.amd.com/projects/rccl/

Inspect the installed headers as the final API truth:

```bash
rg -n "ncclAllReduce|ncclTuner" /opt/rocm/include
```

---

## 50.7 AMD HIP mapped host memory

HIP runtime definitions:

- https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___global_defs.html

Relevant concepts:

```text
hipHostAllocMapped
hipHostAllocPortable
hipHostGetDevicePointer
```

Why they matter:

- they justify a capability test for the mapped-host part of the custom internal AllReduce design.

They do **not** by themselves prove that the complete cross-device visibility/signalling protocol is correct for the target machine.

---

## 50.8 AMD HIP system memory fence

HIP C++ language extensions:

- https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/hip_cpp_language_extensions.html

Relevant primitive:

```text
__threadfence_system()
```

Current documentation describes a system scope that makes prior writes visible to other devices and the host.

Older ROCm documentation had additional caveats/workarounds around system fences and coherent/fine-grained host memory. That version history is exactly why the installed ROCm version belongs in experiment identity and why the capability spike is mandatory.

---

## 50.9 AMD HIP events

Event management:

- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/event_management.html

Relevant APIs:

```text
hipEventCreate
hipEventRecord
hipEventQuery
hipEventSynchronize
hipEventElapsedTime
```

Why they matter:

- event synchronization proves preceding work on the recorded device stream has completed;
- per-device event timing is useful diagnostic evidence;
- it is not a substitute for coordinated multi-device wall timing.

---

## 50.10 ROCprofiler-SDK / rocprofv3

Current/develop usage:

- https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html

Quick guide:

- https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick_guide.html

Useful options, depending on installed version:

```text
--hip-trace
--kernel-trace
--memory-copy-trace
--marker-trace
--runtime-trace
--sys-trace
--rccl-trace
```

Why they matter:

- prove candidate path identity;
- correlate conversion/helper kernels with candidate markers;
- inspect copies and allocation activity;
- inspect RCCL activity where supported.

---

# 51. Final decisions

## HI17

**Proceed, but fix the execution-plan schema first.**

Durable identity should distinguish:

```text
operand/internal type
accumulation type
output type
src0 conversion
src1 conversion
API strategy
numerical class
```

The first actual performance experiment after telemetry and forced-native parity should target the known gfx1100 F16-temporary/direct-F32 difference while separating accumulation precision from output conversion wherever the installed API permits.

Provider environment policy stays outside per-signature identity.

## HI18

**Proceed with existing-provider measurement, but correct the initial candidate set.**

On current upstream HIP:

```text
RCCL = real
meta fallback = real
custom internal provider = not implemented
```

Build requested/effective telemetry and a true multi-device timer before ranking anything.

Treat a HIP port of the custom PCIe AllReduce as a **separate high-upside implementation item** gated by a small mapped-host-memory/system-fence/signalling proof.

## Overall priority

After Phase 0, decide engineering priority from measured:

```text
calls × complete-path cost
```

not signature count and not the number of theoretical tuning controls.

A clever control surface is only worth implementing when the runtime attribution shows that changing it can matter.
