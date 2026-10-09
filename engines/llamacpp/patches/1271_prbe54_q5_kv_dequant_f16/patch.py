"""PRBE54: FA-only half2 F16 dequantization for legacy Q5 KV cache types."""

import re

from bigcherry.patcher import Edit, FilePatch

_Q5_KERNEL = r'''
template <bool q5_1>
static __global__ void bigcherry_dequantize_block_q5_f16(
        const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    static_assert(QK5_0 == QK5_1, "Q5 legacy block widths must match");
    const int iqs = threadIdx.x & 15;
    const int64_t ib = 2*(int64_t) blockIdx.x + threadIdx.x/16;
    const int64_t nb = k / QK5_0;
    if (ib >= nb) {
        return;
    }

    uint32_t qh;
    half2 out;
    if constexpr (q5_1) {
        const block_q5_1 * x = (const block_q5_1 *) vx;
        memcpy(&qh, x[ib].qh, sizeof(qh));
        const int q = x[ib].qs[iqs];
        const int lo = (q & 0x0f) | (((qh >> iqs)        & 1) << 4);
        const int hi = (q >> 4)   | (((qh >> (iqs + 16)) & 1) << 4);
        const half2 dm = x[ib].dm;
        out = __hadd2(
            __hmul2(make_half2(lo, hi), __half2half2(__low2half(dm))),
            __half2half2(__high2half(dm)));
    } else {
        const block_q5_0 * x = (const block_q5_0 *) vx;
        memcpy(&qh, x[ib].qh, sizeof(qh));
        const int q = x[ib].qs[iqs];
        const int lo = (q & 0x0f) | (((qh >> iqs)        & 1) << 4);
        const int hi = (q >> 4)   | (((qh >> (iqs + 16)) & 1) << 4);
        out = __hmul2(make_half2(lo - 16, hi - 16), __half2half2(x[ib].d));
    }

    y[ib*QK5_0 + iqs]            = __low2half(out);
    y[ib*QK5_0 + iqs + QK5_0/2] = __high2half(out);
#else
    GGML_UNUSED_VARS(vx, y, k);
    NO_DEVICE_CODE;
#endif
}

'''

_Q5_WRAPPERS = r'''
static void bigcherry_dequantize_block_q5_0_f16_cuda(
        const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK5_0;
    bigcherry_dequantize_block_q5_f16<false><<<(nb + 1)/2, 32, 0, stream>>>(vx, y, k);
}

static void bigcherry_dequantize_block_q5_1_f16_cuda(
        const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK5_1;
    bigcherry_dequantize_block_q5_f16<true><<<(nb + 1)/2, 32, 0, stream>>>(vx, y, k);
}

'''

_FATTN_SELECTOR = r'''
to_fp16_cuda_t ggml_get_to_fp16_fattn_cuda(ggml_type type) {
#ifdef GGML_USE_HIP
    if (type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
        if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
            static std::atomic_flag bigcherry_prbe54_logged = ATOMIC_FLAG_INIT;
            if (!bigcherry_prbe54_logged.test_and_set(std::memory_order_relaxed)) {
                GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1271_prbe54_q5_kv_dequant_f16 path=q5_fattn_f16_dequant contract=PRBE54-Q5-KV-DEQUANT-F16 type=%d\n",
                              (int) type);
            }
        }
        return type == GGML_TYPE_Q5_0
            ? bigcherry_dequantize_block_q5_0_f16_cuda
            : bigcherry_dequantize_block_q5_1_f16_cuda;
    }
#endif
    return ggml_get_to_fp16_cuda(type);
}

'''

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/convert.cu",
        description="PRBE54 Q5_0/Q5_1 half2 F16 dequant and FA-only selector",
        edits=(
            Edit(
                id="prbe54-headers",
                anchor=r"#include <cstdint>\n",
                mode="insert_before",
                text="#include <atomic>\n#include <cstdlib>\n",
                guard=r"#include <atomic>\n#include <cstdlib>",
                rationale="Add only the C++ headers used by the trace-once FA selector.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="prbe54-q5-kernel",
                anchor=r"(?=template<typename dst_t>\nstatic __global__ void dequantize_block_q4_0\()",
                mode="insert_before",
                text=_Q5_KERNEL,
                guard=r"bigcherry_dequantize_block_q5_f16",
                rationale="Attach immediately before the existing legacy-Q4 dequant kernels, in the contiguous legacy-quant dequant section.",
                expect_matches=1,
                max_span_lines=1,
            ),
            Edit(
                id="prbe54-q5-wrappers",
                anchor=r"(?=template<typename dst_t>\nstatic void dequantize_row_q2_K_cuda\()",
                mode="insert_before",
                text=_Q5_WRAPPERS,
                guard=r"bigcherry_dequantize_block_q5_0_f16_cuda",
                rationale="Attach wrappers after contiguous legacy dequant helpers and before K-quant row helpers.",
                expect_matches=1,
                max_span_lines=1,
            ),
            Edit(
                id="prbe54-fattn-selector",
                anchor=r"(?=to_bf16_cuda_t ggml_get_to_bf16_cuda\(ggml_type type\) \{)",
                mode="insert_before",
                text=_FATTN_SELECTOR,
                guard=r"to_fp16_cuda_t ggml_get_to_fp16_fattn_cuda",
                rationale="Add an FA-specific selector adjacent to the global conversion selectors; non-FA callers keep ggml_get_to_fp16_cuda.",
                expect_matches=1,
                max_span_lines=1,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/convert.cuh",
        description="Declare the PRBE54 FA-specific F16 converter selector",
        edits=(
            Edit(
                id="prbe54-selector-decl",
                anchor=r"to_fp16_cuda_t ggml_get_to_fp16_cuda\(ggml_type type\);\n",
                mode="insert_after",
                text="to_fp16_cuda_t ggml_get_to_fp16_fattn_cuda(ggml_type type);\n",
                guard=r"ggml_get_to_fp16_fattn_cuda",
                rationale="Expose the FA-only selector without changing the existing global selector API.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/fattn-common.cuh",
        description="Route only FlashAttention contiguous K/V F16 staging through the PRBE54 selector",
        edits=(
            Edit(
                id="prbe54-fattn-k",
                anchor=re.escape("to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);"),
                mode="replace_all",
                text="to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_fattn_cuda(K->type);",
                guard=r"ggml_get_to_fp16_fattn_cuda\(K->type\)",
                rationale="Replace exactly the FA K contiguous-F16 staging selector and no global conversion users.",
                expect_matches=1,
                max_span_lines=1,
            ),
            Edit(
                id="prbe54-fattn-v",
                anchor=re.escape("to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);"),
                mode="replace_all",
                text="to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_fattn_cuda(V->type);",
                guard=r"ggml_get_to_fp16_fattn_cuda\(V->type\)",
                rationale="Replace exactly the FA V contiguous-F16 staging selector and no global conversion users.",
                expect_matches=1,
                max_span_lines=1,
            ),
        ),
    ),
]
