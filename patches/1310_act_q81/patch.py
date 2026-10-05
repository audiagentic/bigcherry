"""1310 (QFP13): gated / plain activations also produce the Q8_1 activation their MMVQ consumer needs.

QFP13 census (Flash-Next decode, per generated token per GPU): unary_gated_op_kernel -> quantize_q8_1 ~30 and
unary_op_kernel -> quantize_q8_1 ~30, each followed by mul_mat_vec_q. Same pattern as 1309 (PRBE06): with
BIGCHERRY_ACT_Q81=1 and GGML_HIP_Q8_1_CACHE_MODE=on (1307), an eligible F32 activation (row length a multiple of
MATRIX_ROW_PADDING so the padded row equals the row, <= 16 rows, contiguous) launches bc_act_q81_kernel, which
writes the normal F32 result and, in the same launch, native-layout Q8_1 blocks into a 1235 cache slab published
under exactly the key ggml_cuda_mul_mat_vec_q builds for src1 == this node. The MMVQ consumer then skips its
quantize launch; other consumers miss and quantize as before. Each 32-element group is 32 consecutive threads of
one block (block size and row length are multiples of 32), so group reductions are width-32 warp reductions and the
Q8_1 math (amax/127, roundf, half2(d, sum)) matches quantize_q8_1 bit for bit. Reservation failure (e.g. during
graph capture) or an ineligible shape runs the unchanged kernel. Requires 1307.

Default on since 2026-10-05 (BIGCHERRY_ACT_Q81 unset = on); BIGCHERRY_ACT_Q81=0 disables.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_INCLUDE_ANCHOR = '#include "convert.cuh"\n'
_INCLUDE_TEXT = ('#if defined(GGML_USE_HIP)\n'
                 '#include "hip-q81-cache.h"  // bigcherry 1310: Q8_1 activation cache (1235/1307)\n'
                 '#include <atomic>\n'
                 '#endif\n')

_HELPER_ANCHOR = "template <float (*op)(float), typename T>\nstatic __global__ void unary_op_kernel(const T * x, T * dst, const int k) {\n"

_HELPER = r"""#if defined(GGML_USE_HIP)
// bigcherry 1310: F32 activation (plain op(x) or gated op(x) * g) that also writes native-layout Q8_1 blocks of its
// output. Threads run over the PADDED index space (rows x n_padded, n_padded = GGML_PAD(n, MATRIX_ROW_PADDING)),
// exactly the layout quantize_row_q8_1_cuda produces: columns past n are zeros in Q8_1 and write no F32. The row
// length must be a multiple of QK8_1 and the block size a multiple of QK8_1, so every 32-element group is 32
// consecutive threads that are all in range together.
template <float (*op)(float), bool gated>
static __global__ void bc_act_q81_kernel(const float * x, const float * g, float * dst, block_q8_1 * yq,
        const int64_t k_padded, const int64_t n, const int64_t n_padded, const int64_t n_src, const int64_t o0,
        const int64_t o1) {
    const int64_t ip = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (ip >= k_padded) {
        return;
    }
    const int64_t row = ip / n_padded;
    const int64_t col = ip % n_padded;
    float v = 0.0f;
    if (col < n) {
        const int64_t i = row * n + col;  // flat index of the contiguous F32 output
        if constexpr (gated) {
            // gate inputs may be strided per SOURCE row (n_src elements; n_src == n unless flattened, 1312)
            const int64_t srow = i / n_src;
            const int64_t scol = i % n_src;
            const int64_t j0 = srow * o0 + scol;
            const int64_t j1 = o0 == o1 ? j0 : srow * o1 + scol;
            v = op(x[j0]) * g[j1];
        } else {
            v = op(x[i]);
        }
        dst[i] = v;
    }
    const int64_t i = ip;  // Q8_1 position in the padded layout
    const float amax = warp_reduce_max<QK8_1>(fabsf(v));
    const float sum  = warp_reduce_sum<QK8_1>(v);
    const float d    = amax / 127.0f;
    const int   iqs  = (int) (i % QK8_1);
    block_q8_1 & b   = yq[i / QK8_1];
    b.qs[iqs] = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
    if (iqs == 0) {
        b.ds = make_half2(d, sum);
    }
}

// Returns true when the activation was produced (F32 + Q8_1 published); false = caller runs the normal kernel.
template <float (*op)(float), bool gated>
// flatten01 (1312): publish dims 0 and 1 merged into one row, the layout of a consumer that reads this node through a
// contiguous reshape_3d(ne0*ne1, ne2, ne3) whose row needs MMVQ padding (1307's flattened-reshape lookup).
static bool bc_act_q81_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float * x, const float * g,
        const int64_t o0, const int64_t o1, const bool flatten01) {
    static const bool enabled = getenv("BIGCHERRY_ACT_Q81") == nullptr || atoi(getenv("BIGCHERRY_ACT_Q81")) != 0;
    if (!enabled || ggml_hip_q81_cache_mode_get() == GGML_HIP_Q81_CACHE_OFF || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(dst)) {
        return false;
    }
    const int64_t ne0 = flatten01 ? dst->ne[0]*dst->ne[1] : dst->ne[0];
    const int64_t ne1 = flatten01 ? dst->ne[2] : dst->ne[1];
    const int64_t ne2 = flatten01 ? dst->ne[3] : dst->ne[2];
    const int64_t ne3 = flatten01 ? 1 : dst->ne[3];
    if (ne0 % QK8_1 != 0 || !ggml_hip_q81_decode_graph || (flatten01 && dst->ne[0] % QK8_1 != 0)) {  // only in decode-shaped graphs (1307): MMVQ consumers
        return false;
    }
    const int64_t ne0_padded = GGML_PAD(ne0, MATRIX_ROW_PADDING);  // MMVQ's padded src1 row
    ggml_hip_q81_cache & q81 = ggml_hip_q81_cache_for_context(ctx);
    const ggml_hip_q81_cache_key key = ggml_hip_q81_cache_make_key(
        q81, dst, dst->data, ctx.curr_stream_no, ne0, ne0_padded, ne1, ne2, ne3, ne0, ne0*ne1, ne0*ne1*ne2);
    if (ggml_hip_q81_cache_find(q81, key) != nullptr) {
        return false;
    }
    const int64_t k_padded = ne0_padded * ne1*ne2*ne3;
    const ggml_hip_q81_cache_reservation r = ggml_hip_q81_cache_reserve(q81, (size_t) (k_padded / QK8_1) * sizeof(block_q8_1));
    if (!r.ok) {
        return false;
    }
    constexpr int block = 256;
    static_assert(block % QK8_1 == 0, "1310: block must cover whole Q8_1 groups");
    const ggml_cuda_kernel_launch_params launch_params((dim3) ((k_padded + block - 1) / block), dim3(block, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(bc_act_q81_kernel<op, gated>, launch_params, x, g, (float *) dst->data, (block_q8_1 *) r.ptr,
                            k_padded, ne0, ne0_padded, dst->ne[0], o0, o1);
    ggml_hip_q81_cache_publish(q81, key, r);
    if (getenv("BIGCHERRY_Q81_TRACE") != nullptr) {  // pair with 1307's miss trace
        GGML_LOG_WARN("BIGCHERRY_Q81 publish-act gen=%llu node=%p(%s) data=%p ne=%lld,%lld,%lld,%lld\n",
            (unsigned long long) ggml_hip_q81_cache_current_generation(q81), (const void *) dst, dst->name, dst->data,
            (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3);
    }
    static std::atomic<bool> logged{false};
    if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr && !logged.exchange(true)) {
        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1310_act_q81 gated=%d ne0=%lld rows=%lld\n", gated ? 1 : 0,
                      (long long) ne0, (long long) (ne1*ne2*ne3));
    }
    return true;
}
#endif

"""

_UNARY_OLD = """    } else {
        unary_cuda<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
"""
_UNARY_NEW = """    } else {
#if defined(GGML_USE_HIP)
        if (bc_act_q81_try<op, false>(ctx, dst, (const float *) src0_d, nullptr, 0, 0, false)) {  // bigcherry 1310
            return;
        }
#endif
        unary_cuda<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
"""

_GATED_OLD = """        unary_gated_cuda<op>(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), stream);
"""
_GATED_NEW = """#if defined(GGML_USE_HIP)
        if (bc_act_q81_try<op, true>(ctx, dst, src0_p, src1_p, src0_o / sizeof(float), src1_o / sizeof(float), false)) {  // bigcherry 1310
            return;
        }
#endif
        unary_gated_cuda<op>(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), stream);
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/unary.cu",
        description="1310: F32 plain/gated activations emit the Q8_1 activation for their MMVQ consumer",
        language="none",
        edits=(
            Edit(
                id="act-q81-include",
                anchor=re.escape(_INCLUDE_ANCHOR),
                mode="insert_after",
                text=_INCLUDE_TEXT,
                guard=r"bigcherry 1310: Q8_1 activation cache",
                rationale="unary.cu's convert.cuh include; the cache API is HIP-only.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="act-q81-helper",
                anchor=re.escape(_HELPER_ANCHOR),
                mode="insert_before",
                text=_HELPER,
                guard=r"static bool bc_act_q81_try\(",
                rationale="Before the plain unary kernel, so both ggml_cuda_op_unary and ggml_cuda_op_unary_gated "
                          "(defined later) can call it.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="act-q81-unary",
                anchor=re.escape(_UNARY_OLD),
                mode="replace",
                text=_UNARY_NEW,
                guard=r"bc_act_q81_try<op, false>",
                rationale="F32 branch of ggml_cuda_op_unary.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="act-q81-gated",
                anchor=re.escape(_GATED_OLD),
                mode="replace",
                text=_GATED_NEW,
                guard=r"bc_act_q81_try<op, true>",
                rationale="F32 launch in ggml_cuda_op_unary_gated (after the swapped/split pointer setup).",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_ACT_Q81', '0|1', '1 (on)',
           'activation ops write the Q8_1 activation directly for the next MMVQ (needs Q8_1 cache); 0 disables'),
)
