"""1312 (QFP13): same-shape F32 MUL also produces the Q8_1 activation its MMVQ consumer needs.

QFP13 Q8_1 miss trace after 1307-1311 (Flash-Next decode): two remaining quantize -> MMVQ chains per layer are fed by a
plain GGML_OP_MUL - the full-attention output gate (attn_gated = attn * sigmoid(gate) -> wo) and the GatedDeltaNet
gated output norm (rms_norm(x)*w * sigmoid(z) -> reshape final_output -> ssm_out). With BIGCHERRY_ACT_Q81=1 and
GGML_HIP_Q8_1_CACHE_MODE=on (1307), a MUL whose operands and result are F32, contiguous and the same shape (no
broadcast), with a row length that is a multiple of QK8_1, in a decode-shaped graph, launches bc_mul_q81_kernel: it
writes the normal F32 product and, in the same launch, native-layout Q8_1 blocks (MMVQ's padded row) into a 1235 cache
slab published under the key ggml_cuda_mul_mat_vec_q builds for src1 == this node (1307's lookup resolves the reshape
view to it). Q8_1 math is 1310's (bit-identical to quantize_q8_1). Anything else runs the unchanged bin_bcast path.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_INCLUDE_ANCHOR = '#include "convert.cuh"\n'
_INCLUDE_TEXT = ('#if defined(GGML_USE_HIP)\n'
                 '#include "hip-q81-cache.h"  // bigcherry 1312: Q8_1 activation cache (1235/1307)\n'
                 '#include <atomic>\n'
                 '#include <cstdlib>\n'
                 '#endif\n')

_MUL_OLD = """void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_mul>>("""

_MUL_NEW = r"""#if defined(GGML_USE_HIP)
// bigcherry 1312: same-shape F32 product that also writes native-layout Q8_1 blocks of its output, over MMVQ's padded
// index space (rows x n_padded): columns past n are zero in Q8_1 and write no F32. n and the block size are multiples
// of QK8_1, so every 32-element group is 32 consecutive threads that are all in range together.
static __global__ void bc_mul_q81_kernel(const float * a, const float * b, float * dst, block_q8_1 * yq,
        const int64_t k_padded, const int64_t n, const int64_t n_padded) {
    const int64_t ip = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (ip >= k_padded) {
        return;
    }
    const int64_t row = ip / n_padded;
    const int64_t col = ip % n_padded;
    float v = 0.0f;
    if (col < n) {
        const int64_t i = row * n + col;
        v = a[i] * b[i];
        dst[i] = v;
    }
    const float amax = warp_reduce_max<QK8_1>(fabsf(v));
    const float sum  = warp_reduce_sum<QK8_1>(v);
    const float d    = amax / 127.0f;
    const int   iqs  = (int) (ip % QK8_1);
    block_q8_1 & q   = yq[ip / QK8_1];
    q.qs[iqs] = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
    if (iqs == 0) {
        q.ds = make_half2(d, sum);
    }
}

// Returns true when the product was produced (F32 + Q8_1 published); false = caller runs the normal bin_bcast.
static bool bc_mul_q81_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool enabled = getenv("BIGCHERRY_ACT_Q81") != nullptr && atoi(getenv("BIGCHERRY_ACT_Q81")) != 0;
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    if (!enabled || !ggml_hip_q81_decode_graph || ggml_hip_q81_cache_mode_get() == GGML_HIP_Q81_CACHE_OFF ||
            dst->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(dst) || !ggml_is_contiguous(a) || !ggml_is_contiguous(b) ||
            !ggml_are_same_shape(a, dst) || !ggml_are_same_shape(b, dst) || dst->ne[0] % QK8_1 != 0) {
        return false;
    }
    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
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
    static_assert(block % QK8_1 == 0, "1312: block must cover whole Q8_1 groups");
    bc_mul_q81_kernel<<<(k_padded + block - 1) / block, block, 0, ctx.stream()>>>(
        (const float *) a->data, (const float *) b->data, (float *) dst->data, (block_q8_1 *) r.ptr, k_padded, ne0, ne0_padded);
    ggml_hip_q81_cache_publish(q81, key, r);
    if (getenv("BIGCHERRY_Q81_TRACE") != nullptr) {  // pair with 1307's miss trace
        GGML_LOG_WARN("BIGCHERRY_Q81 publish-mul gen=%llu node=%p(%s) data=%p ne=%lld,%lld,%lld,%lld\n",
            (unsigned long long) ggml_hip_q81_cache_current_generation(q81), (const void *) dst, dst->name, dst->data,
            (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3);
    }
    static std::atomic<bool> logged{false};
    if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr && !logged.exchange(true)) {
        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1312_mul_q81 ne0=%lld rows=%lld\n", (long long) ne0, (long long) (ne1*ne2*ne3));
    }
    return true;
}
#endif

void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    if (bc_mul_q81_try(ctx, dst)) {  // bigcherry 1312
        return;
    }
#endif
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_mul>>("""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/binbcast.cu",
        description="1312: same-shape F32 MUL emits the Q8_1 activation for its MMVQ consumer",
        language="none",
        edits=(
            Edit(
                id="mul-q81-include",
                anchor=re.escape(_INCLUDE_ANCHOR),
                mode="insert_after",
                text=_INCLUDE_TEXT,
                guard=r"bigcherry 1312: Q8_1 activation cache",
                rationale="binbcast.cu's convert.cuh include; the cache API is HIP-only.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="mul-q81-op",
                anchor=re.escape(_MUL_OLD),
                mode="replace",
                text=_MUL_NEW,
                guard=r"static bool bc_mul_q81_try\(",
                rationale="Entry of ggml_cuda_op_mul (the only GGML_OP_MUL dispatch); helper defined just before it.",
                expect_matches=1,
                max_span_lines=3,
            ),
        ),
    ),
]
