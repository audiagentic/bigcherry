"""HI04 - route the dense matmul selector through measured dispatch.

Upstream's ``ggml_cuda_mul_mat`` decides and launches in one motion: every
``if (should_use_X)`` ends in a call and a ``return``. The choice exists only as
control flow, so there is nothing to measure, store, or replay.

The edits here are deliberately minimal. Rather than restructure upstream's
ladder -- which would be a large diff, hard to review, and hard to keep
applying across releases -- a single guarded hook is inserted at the top of each
entry point. ``ggml_hip_dispatch_mul_mat`` reproduces the ladder itself and
returns false whenever it declines, in which case upstream's original code runs
untouched.

That shape buys three things:

* a build with the layer compiled in but in native mode executes upstream's own
  code path, not a reimplementation of it (standards 4.2);
* the diff stays four small insertions, so it survives release churn;
* the fallback is the real upstream code, so declining is always safe.

The cuBLAS entry point also needs exposing: it is ``static`` upstream, and the
BLAS candidate has to be able to call it.
"""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch

_INCLUDE = """
#ifdef GGML_HIP_DISPATCH
#include "ggml-cuda/hip-autotune-dispatch.cuh"
#endif
#ifdef GGML_HIP_AUTOTUNE_RECORD
#include "ggml-cuda/hip-autotune-record.h"
#endif
"""

def _record_api(name):
    return (
        "\n#ifdef GGML_HIP_AUTOTUNE_RECORD\n"
        f"        ggml_hip_record_effective_call_api(\"{name}\");\n"
        "#endif\n"
    )

# A non-static forwarder rather than making the original non-static: upstream
# keeps its own linkage, and the exported name is ours to keep stable.
#
# The forwarder is inserted *above* the definition it calls, so it carries a
# forward declaration of that static function with it. Anchoring below the
# definition instead would mean anchoring to whatever function happens to
# follow it, which is a far weaker attachment.
_CUBLAS_FORWARDER = """
#ifdef GGML_HIP_DISPATCH
// bigcherry: the BLAS candidate needs to reach this path, which upstream keeps
// file-local. A forwarder leaves upstream's own linkage untouched.
static void ggml_cuda_mul_mat_cublas(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, ggml_tensor * dst);

void ggml_cuda_mul_mat_cublas_dispatch(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}
#endif

"""

_MUL_MAT_HOOK = """
#ifdef GGML_HIP_DISPATCH
    // bigcherry: measured dispatch. Returns false in native mode, and whenever
    // the dispatch layer declines this operation -- upstream's ladder below
    // then runs exactly as it always did.
    if (ggml_hip_dispatch_mul_mat(ctx, src0, src1, /*ids =*/ nullptr, dst,
                                  /*fusion =*/ nullptr)) {
        return;
    }
#endif
"""

_FLUSH_HOOK = """
#ifdef GGML_HIP_DISPATCH
    // bigcherry: flush record-mode observations and the bounded replay-miss
    // log. Without this the miss log is only ever written by a host that calls
    // ggml_hip_autotune_flush() explicitly -- so `GGML_HIP_DISPATCH_MISS=
    // native-record` would collect misses and silently discard them at exit,
    // which looks exactly like "there were no misses".
    ggml_hip_autotune_flush();
#endif
"""

_MUL_MAT_ID_HOOK = """
#ifdef GGML_HIP_DISPATCH
    // bigcherry: MUL_MAT_ID is a distinct semantic operation with its own
    // signature fields (standards 11.2), not a shape of MUL_MAT.
    if (ggml_hip_dispatch_mul_mat(ctx, dst->src[0], dst->src[1], dst->src[2],
                                  dst, /*fusion =*/ nullptr)) {
        return;
    }
#endif
"""

PATCH = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="route the dense and MoE matmul selectors through dispatch",
    edits=(
        Edit(
            id="dispatch-include",
            # One of upstream's own includes, chosen because the dispatch layer
            # depends on it anyway. Anchoring to the standard-library block
            # below would be more fragile.
            anchor=r'^#include "ggml-cuda/mmvq\.cuh"$',
            rationale="upstream's ggml-cuda include block",
            text=_INCLUDE,
            guard=r'#include "ggml-cuda/hip-autotune-dispatch\.cuh"',
        ),
        Edit(
            id="blas-api-sgemm-telemetry",
            anchor=r"^                    \(const float \*\) beta,  \(float       \*\)  dst_ptr, ne0\)\);$",
            rationale="the completed native single-matrix F32 BLAS call",
            text=_record_api("cublasSgemm"),
            guard=r"ggml_hip_record_effective_call_api\(\"cublasSgemm\"\)",
        ),
        Edit(
            id="blas-api-gemmex-telemetry",
            anchor=r"^                    CUBLAS_GEMM_DEFAULT_TENSOR_OP\)\);$",
            rationale="the completed native single-matrix typed BLAS call",
            text=_record_api("cublasGemmEx"),
            guard=r"ggml_hip_record_effective_call_api\(\"cublasGemmEx\"\)",
        ),
        Edit(
            id="blas-api-strided-telemetry",
            anchor=r"^                CUBLAS_GEMM_DEFAULT_TENSOR_OP\)\);$",
            rationale="the completed native strided-batched BLAS call",
            expect_matches=2,
            occurrence=0,
            text=_record_api("cublasGemmStridedBatchedEx"),
            guard=r"ggml_hip_record_effective_call_api\(\"cublasGemmStridedBatchedEx\"\)",
        ),
        Edit(
            id="blas-api-pointer-batched-telemetry",
            anchor=r"^                CUBLAS_GEMM_DEFAULT_TENSOR_OP\)\);$",
            rationale="the completed native pointer-batched BLAS call",
            expect_matches=2,
            occurrence=1,
            text=_record_api("cublasGemmBatchedEx"),
            guard=r"ggml_hip_record_effective_call_api\(\"cublasGemmBatchedEx\"\)",
        ),
        Edit(
            id="cublas-forwarder",
            # Placed immediately after the definition it forwards to, which is
            # located by its opening line rather than by offset.
            anchor=r"^static void ggml_cuda_mul_mat_cublas\(ggml_backend_cuda_context & ctx, "
                   r"const ggml_tensor \* src0, const ggml_tensor \* src1, ggml_tensor \* dst\) \{$",
            rationale="the cuBLAS dense entry point, which is file-local upstream",
            mode="insert_before",
            text=_CUBLAS_FORWARDER,
            guard=r"void ggml_cuda_mul_mat_cublas_dispatch\(",
        ),
        Edit(
            id="mul-mat-hook",
            # Anchored to the function's opening line plus its first statement,
            # so the hook lands before any of upstream's own decisions.
            anchor=r"^static void ggml_cuda_mul_mat\(ggml_backend_cuda_context & ctx, "
                   r"const ggml_tensor \* src0, const ggml_tensor \* src1, ggml_tensor \* dst\) \{\n"
                   r"    GGML_TENSOR_BINARY_OP_LOCALS$",
            rationale="the top of ggml_cuda_mul_mat, before its family ladder",
            text=_MUL_MAT_HOOK,
            guard=r"ggml_hip_dispatch_mul_mat\(ctx, src0, src1, /\*ids =\*/ nullptr",
        ),
        Edit(
            id="dispatch-flush-hook",
            anchor=r"^static void ggml_backend_cuda_free\(ggml_backend_t backend\) \{$",
            rationale="backend teardown, the last point at which anything the "
                      "dispatch layer accumulated can still be written out",
            text=_FLUSH_HOOK,
            guard=r"ggml_hip_autotune_flush\(\);",
        ),
        Edit(
            id="mul-mat-id-hook",
            anchor=r"^static void ggml_cuda_mul_mat_id\(ggml_backend_cuda_context & ctx, "
                   r"ggml_tensor \* dst\) \{$",
            rationale="the top of ggml_cuda_mul_mat_id",
            text=_MUL_MAT_ID_HOOK,
            guard=r"ggml_hip_dispatch_mul_mat\(ctx, dst->src\[0\]",
        ),
    ),
)
