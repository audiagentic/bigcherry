# Family model — verified against upstream

Assessment of the expanded family taxonomy, checked against the vendored
`llama.cpp` tree rather than accepted on description. Written 2026-08-05.

**Outcome:** adopt DQ_BLAS/BLAS and SPLIT_REDUCE; reject WMMA as a family;
adopt the four-way separation of *signature / context / candidate / observation*
as a standing rule. Three claims from the first pass are corrected below, one of
them mine.

---

## The structural rule (adopted)

A candidate's durable identity must contain only what the candidate **chose**.
Everything else belongs to one of three other records:

| Record | Holds | Varies with |
| --- | --- | --- |
| `MatmulSignature` | op kind, src0/src1/dst types, `ne*`, layout, contiguity, broadcast, batchedness, split geometry, `prec` | the request |
| `ExecutionContext` | backend, architecture, device count, topology, split mode, ROCm/library versions, enabled capabilities | the machine and the build |
| `Candidate` | family, family parameters, provider, policy | the choice under test |
| `Observation` | actual family/provider/api/algorithm, library backend, solution id, kernel names, conversion kernels, temp+workspace bytes, fallback reason and depth | the run |

The failure mode this prevents is concrete: embedding `source_type` in a
DQ_BLAS candidate would mint one candidate record per quantised type that differ
only in *eligibility*, not in implementation. `dq-blas:f16:hipblas:gemmex:auto:v1`
is one execution plan; which source types it can serve is an eligibility
question answered per signature.

This is not a new principle for the codebase — `hip-autotune-types.h` already
separates `ggml_hip_dispatch_signature_v1` from the candidate descriptor and the
hardware key. The correction is that the *proposed new families* had started to
violate it, and MMQ's existing name needs a justification rather than an
exemption (below).

### Where MMQ stands under the rule

`mmq:q8_0:j64:fb0:...` contains `q8_0`, which looks like a request attribute.
It survives the rule because MMQ kernels are **compile-time specialised per
type**: `q8_0` names a distinct compiled artifact, not the incoming tensor. The
right reading of the field is `kernel_type`, and the docs should say so.

Same for MMVQ/MMVF/MMF. For DQ_BLAS and BLAS it does **not** hold — hipBLAS is
one library call, and the source type only selects which conversion kernel runs
ahead of it. Hence the different treatment.

### `fallback` is overloaded and must be renamed

MMQ's `fb0`/`fb1` field means *a compile-time fallback implementation selected
within the `CASE` row*. It has nothing to do with permission for the dispatcher
to fall back to another family. Only the first belongs in candidate identity;
the second is a property of the selection request or the benchmark mode.

Action: rename to `impl_fallback` in docs and in the catalog's field naming, and
keep the stable-name token `fb` (renaming the token would invalidate every
recorded name — standards 2.1).

---

## Verified: DQ_BLAS is real and expensive

`ggml_cuda_mul_mat_cublas_impl` is up to three operations, not one
([ggml-cuda.cu:1414-1624](../../vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu#L1414-L1624)):

1. **Input conversion** — `if (src0->type == compute_type)` else `convert_func`
   into a pool temp. Applied independently to src0 and src1, and in two variants
   (`convert` for contiguous, `convert_nc` for strided).
2. **The GEMM.**
3. **Output conversion** — `if (cu_data_type != CUDA_R_32F)` then
   `to_fp32_cuda(dst_temp, dst_ddf, ...)`.

On the 27B MTP profile **19 of 92 signatures** resolve to BLAS. Reporting all of
that as one opaque `blas:hipblas-auto:v1` hides a fifth of the workload's
distinct operations behind a single unmeasured token.

Adopted, with the candidate identity the review proposed:

```text
(conversion_route, compute_type, output_conversion_route,
 provider, api_strategy, provider_policy)
```

`conversion_route` rather than `convert_type` is the right call. Today the route
is a bijection with `(source_type → target)`, so the two look equivalent — but
`convert` vs `convert_nc` is *already* two routes for one type pair, so the
degenerate reading is wrong even now.

## Correction: DQ_BLAS and BLAS are not two entry points

Both are the same function. There is exactly one BLAS entry,
[`ggml_cuda_mul_mat_cublas`](../../vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu#L1642),
templated on `compute_type`, called from two sites. Whether a call is "DQ_BLAS"
or "BLAS" is decided by `src0->type == compute_type` — and `compute_type` is
**the candidate's own choice**.

So the same candidate is DQ_BLAS against a q8_0 signature and plain BLAS against
an f16 one. Splitting them into two families re-imports a request attribute at
the family level, which is precisely what the structural rule forbids.

**Recommendation: one `BLAS` family; `dq` is a derived label.** A conversion
route of `none` means no dequantisation happened. Observations still carry
`actual_family = dq_blas | blas` so grouping, reporting and cost attribution all
work as intended — the analysis the split was meant to enable is preserved
without a family whose membership depends on the input.

This is the one point where I'd depart from "keep the seven families exactly as
proposed", and it follows from the review's own rule rather than against it. The
table below records six families; if the seven-family form is preferred, nothing
downstream changes except the label.

## Correction (mine): four BLAS APIs in the matmul path, not six

The earlier pass counted six `cublas*` entry points across `ggml-cuda`. Two of
those live in `out-prod.cu` and `ssm-scan.cu` — different operations, not
MUL_MAT. The matmul path has **four**:

| API | condition |
| --- | --- |
| `cublasSgemm` | `compute_type == F32 && ne12 == 1 && ne13 == 1` |
| `cublasGemmEx` | `ne12 == 1 && ne13 == 1` |
| `cublasGemmStridedBatchedEx` | `r2 == 1 && r3 == 1 && src0/src1 contiguous over dims 2,3` |
| `cublasGemmBatchedEx` | otherwise (builds a pointer array via `k_compute_batched_ptrs`) |

The original "four" was right; my "six" over-counted by ignoring the operation.

## Verified: `api_strategy` is correct, and upstream says why

The API is chosen by a hard `if / else if / else if / else` chain over signature
properties. For a given signature exactly one API is reachable by default — so a
raw `api` field in candidate identity would encode a request attribute *and*
could name an API that cannot serve the shape.

What makes the strategy tunable is upstream's own comment at
[ggml-cuda.cu:1545](../../vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu#L1545):

```text
// Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
// However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
//     probably because the internal kernel selection logic is suboptimal.
```

An admitted guess, hedged with "probably", about "some old" GPUs, deciding the
API for every single-matrix call. That is exactly the class of heuristic this
project exists to measure, and it makes `prefer-strided` a candidate with a
documented reason to exist rather than a speculative one.

```text
api_strategy = native-auto | prefer-strided | prefer-pointer-batched | force-gemmex
observed.api = hipblasGemmStridedBatchedEx | ...
```

## New finding: `output_intermediate_type` is a per-architecture guess — on our hardware

[ggml-cuda.cu:1515-1520](../../vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu#L1515-L1520):

```cpp
if (compute_type == GGML_TYPE_F16) {
    prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
} else if (compute_type == GGML_TYPE_BF16) {
    prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
}
```

For F16 compute, RDNA4 and CDNA get direct F32 output; **RDNA3 does not** — so
on gfx1100, every F16-compute BLAS call allocates an F16 temp and runs a
separate `to_fp32_cuda` pass over the whole output. RDNA4 is one generation
away and gets the opposite treatment.

The 2× XTX rig is gfx1100. This is a measurable A/B on the exact target
hardware, and it is the third leg of DQ_BLAS's cost that the opaque candidate
was hiding. `output_conversion_route` earns its place in candidate identity.

## Verified: `compute_type` is selectable, and upstream already proves it

[ggml-cuda.cu:1649-1688](../../vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu#L1649-L1688):

```cpp
ggml_type compute_type = src0->type;
if (ggml_is_quantized(compute_type)) {
    compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
} else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
    compute_type = GGML_TYPE_F32;
}
if (dst->op_params[0] == GGML_PREC_F32) {
    compute_type = GGML_TYPE_F32;
}
const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");   // f32 | f16 | bf16 | auto
...
switch (compute_type) { case F32: impl<F32>(...); case BF16: impl<BF16>(...); case F16: impl<F16>(...); }
```

Two things fall out:

- Upstream **already ships a global override** for this. Forcing it per candidate
  is the same mechanism at signature granularity, and the templated `switch` is
  the Option C forced-variant pattern the project already uses everywhere else.
  Implementation cost is near zero.
- `dst->op_params[0] == GGML_PREC_F32` **is** the precision equivalence class,
  and our signature already carries it as
  [`prec`](../../src/ggml/src/ggml-cuda/hip-autotune-types.h#L135). The review's
  `required_precision` gate is therefore computable today with no ABI change.

The review's rule is adopted verbatim: *compute_type is a candidate dimension
only within the precision equivalence class permitted by the request* — and the
warning against inferring it from source type is correct. A quantised source
does not make every accumulation mode equivalent; `prec` is the authority.

Candidates carry a `numerical_class`:

```text
exact_baseline
equivalent_within_backend_tolerance
reduced_precision
```

The objective admits or rejects classes **before** performance ranking, so a
reduced-precision candidate can never win by default against a strict request.
This generalises the MMVF accumulator rule already implemented (standards 3.1).

## Rejected: WMMA is not a family

`rocwmma` appears nowhere in the ggml sources — only in CI workflow files. What
exists is `amd_wmma_available(cc)`, a capability predicate used by `mmq.cu`,
`mmf.cu`, `common.cuh` and the FlashAttention files to select code paths
**inside** those families; the intrinsics live in `mmq.cuh` and `mmf.cuh`.

Promoting it would create a family with no entry point and no candidates.
Rejected permanently as a top-level matmul family.

If a nominal MMF configuration ever generates materially different architecture
paths, the right response is an `mma_mode = auto | disable | require` field on
MMF — and even then, keep it observed or architecture-derived first rather than
multiplying the candidate matrix.

**FlashAttention is a different operation**, not a matmul family. Tuning it needs
its own signature space and collection points — a separate subsystem.

## Verified: SPLIT_REDUCE is real, with the identity corrected

[ggml-cuda.cu:1135-1195](../../vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu#L1135-L1195)
selects one of three implementations by assigning `ret->try_allreduce`:

```text
ggml_backend_cuda_comm_try_allreduce_butterfly    (assigned first)
ggml_backend_cuda_comm_try_allreduce_internal     (conditionally overrides)
ggml_backend_cuda_comm_try_allreduce_nccl         (conditionally overrides)
```

`device_count` and `topology` are **execution context**, not candidate identity —
correct, and removed. Candidate identity is:

```text
(preferred_algorithm, fallback_policy)
```

Initial candidate set:

```text
(nccl,      none)
(internal,  none)
(butterfly, none)
(nccl,      internal_then_butterfly)
(internal,  butterfly)
(auto,      upstream_default)
```

The first three are the measurement candidates. The fallback-enabled ones
reproduce production resilience and are **not admissible as evidence about an
algorithm** unless actual-path telemetry is recorded — `winner = nccl` is a lie
if the call fell through to butterfly.

This sits directly on the path of the configuration the project exists to
optimise: 2× XTX, `split_mode tensor`, RCCL enabled.

---

## Telemetry is mandatory, not optional

Any candidate whose execution is chosen by a library or by a fallback chain must
record what actually ran, or its result is uninterpretable:

```text
observed.provider_requested   = hipblas
observed.provider_effective   = rocblas | hipblaslt | ...
observed.api                  = hipblasGemmStridedBatchedEx | ...
observed.reduce_algorithm     = nccl | internal | butterfly
observed.fallback_depth       = 0
observed.fallback_chain       = []
observed.library_backend      = tensile | hipblaslt | unknown
observed.solution_id          = 1234
observed.kernel_name          = ...
observed.conversion_kernels   = [...]
observed.temporary_bytes      = ...
observed.workspace_bytes      = ...
```

A winner recorded without effective-path telemetry must be marked unverified and
excluded from promotion. Standards 2.1 makes the stable name a durable database
identity; a Tensile solution id is not durable, so it is observation only —
unless bigcherry ever *invokes* one explicitly, at which point it becomes a
chosen configuration.

The build namespace must capture ROCm, hipBLAS, rocBLAS and hipBLASLt versions
plus `ROCBLAS_USE_HIPBLASLT`, since internal selection changes between releases
(standards 13.1, extended from source revision to library versions).

## Enumerated but unreachable candidates must not rank

MMVQ is generated but its entry point is still gated. Two independent statuses,
not one:

```text
enumeration_status = implemented | partial | absent
dispatch_status    = wired | unavailable
```

A candidate with `dispatch_status = unavailable` is excluded from ranking and
promotion outright. This is the review's point and it is the correct guard —
the current failure mode is worse than a missing candidate, because "MMQ won"
against an unreachable MMVQ reads as a measurement when it is an artifact.
Upstream still routes small effective batch dimensions to MMVQ ahead of MMQ,
which is why wiring the chooser (HI09) is the blocking item rather than a
finishing touch.

---

## Final family table

| Family | Candidate identity | Eligibility / context | Status |
| --- | --- | --- | --- |
| MMQ | `(kernel_type, J, impl_fallback, full CASE config)` | src0 type, shape, architecture, alignment | implemented |
| MMVQ | `(kernel_type, width, nwarps, rows_per_block, small_k)` | type, effective rows, K, architecture | enumerated; **dispatch unwired** |
| MMVF | `(kernel_type, width, block_size, accumulator)` | type, shape, precision class | implemented |
| MMF | `(kernel_type, width, nwarps)` | type, shape, architecture | implemented |
| BLAS *(incl. DQ)* | `(conversion_route, compute_type, output_conversion_route, provider, api_strategy, provider_policy)` | src0/src1/dst types, layout, batch structure, `prec` | **new — HI17** |
| SPLIT_REDUCE | `(preferred_algorithm, fallback_policy)` | devices, topology, reduction shape/type, peer access | **new — HI18** |

`provider` admits `cublas`/`cublaslt` so the classification is portable, but no
CUDA candidates are generated in the AMD lane. hipBLASLt and cuBLASLt are the
providers worth enumerating properly later — both expose heuristic candidate
enumeration and explicit algorithm selection, which classic hipBLAS/cuBLAS do
not.

`hipblas-auto`, `hipblas-force-tensile` and `hipblas-prefer-hipblaslt` are
**policies** under BLAS, driven by `ROCBLAS_USE_HIPBLASLT` — not families.

## Work items

- **HI17** — BLAS decomposition: conversion routes, `compute_type`,
  `output_conversion_route`, `api_strategy`, provider policies, precision-class
  gating, effective-path telemetry.
- **HI18** — SPLIT_REDUCE: three algorithms plus fallback policies, with
  mandatory actual-path telemetry.
- **HI19** — record/observation split: enforce the four-record separation in the
  ABI and schema, add `numerical_class`, `dispatch_status`, and the observation
  fields above; rename MMQ's `fallback` to `impl_fallback` in documentation and
  catalog fields.
