"""1309 (PRBE06): fused RMSNorm*weight also produces the Q8_1 activation its MMVQ consumer needs.

QFP13/PRBE06 Stage 0 (flashnext-v2-1308-census, 1307 + 1308 on): 29.9 rms_norm_f32<1024, true, ...> ->
quantize_q8_1 -> mul_mat_vec_q triples per generated token per GPU - the fused RMS_NORM+MUL output is consumed by
MMVQ and re-read only to be quantized. With GGML_HIP_Q8_1_CACHE_MODE=on (1307) and an eligible decode shape,
ggml_cuda_op_rms_norm_fused launches rms_norm_mul_q81_f32 instead: the same F32 result is still written (other
consumers, residual paths), and in the same launch each 32-column group is quantized with the native Q8_1 math
(amax/127, roundf, half2(d, sum)) into a 1235 cache slab, published under exactly the key
ggml_cuda_mul_mat_vec_q builds for src1 == the MUL node. The MMVQ consumer then hits the cache and skips its
quantize launch; any other consumer shape simply misses and quantizes as before. Q8_1 blocks are laid out as
quantize_row_q8_1_cuda lays them (rows of GGML_PAD(ncols, MATRIX_ROW_PADDING), zero padding blocks), so a hit is
bit-identical to the standalone quantization. Reservation failure (e.g. during graph capture) falls back to the
unchanged kernel. Requires 1307 (cache wiring) and 1235 (cache).
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_INCLUDE_ANCHOR = '#include "norm.cuh"\n'
_INCLUDE_TEXT = ('#if defined(GGML_USE_HIP)\n'
                 '#include "hip-q81-cache.h"  // bigcherry 1309: Q8_1 activation cache (1235/1307)\n'
                 '#include <atomic>\n'
                 '#endif\n')

_KERNEL_ANCHOR = "template <int block_size>\nstatic __global__ void rms_norm_back_f32(\n"

_KERNEL = r"""// bigcherry 1309 (PRBE06): RMSNorm * weight that also writes the Q8_1 blocks of its output (same layout and math
// as quantize_row_q8_1_cuda, rows padded to ncols_padded with zero blocks). Each 32-column group is handled by 32
// consecutive threads of one iteration, so the group reductions are width-32 warp reductions.
template <int block_size>
static __global__ void rms_norm_mul_q81_f32(const float * x, const float * mul, float * dst, block_q8_1 * yq,
        const int ncols, const int ncols_padded,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
        const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed, const float eps) {
    static_assert(block_size % QK8_1 == 0, "1309: block must cover whole Q8_1 groups");
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;
    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    const int64_t row_flat = ((int64_t) sample*nchannels + channel)*nrows + row;
    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += row_flat*ncols;
    yq  += row_flat*(ncols_padded/QK8_1);
    mul += fastmodulo(sample, mul_nsamples_packed)*mul_stride_sample + fastmodulo(channel, mul_nchannels_packed)*mul_stride_channel
         + fastmodulo(row, mul_nrows_packed)*mul_stride_row;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float scale = rsqrtf(tmp / ncols + eps);

    for (int col = tid; col < ncols_padded; col += block_size) {  // uniform per 32-thread group
        float v = 0.0f;
        if (col < ncols) {
            v = scale * x[col] * mul[fastmodulo(col, mul_ncols_packed)];
            dst[col] = v;
        }
        const float amax = warp_reduce_max<QK8_1>(fabsf(v));
        const float sum  = warp_reduce_sum<QK8_1>(v);
        const float d    = amax / 127.0f;
        const int   iqs  = col % QK8_1;
        block_q8_1 & b   = yq[col / QK8_1];
        b.qs[iqs] = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
        if (iqs == 0) {
            b.ds = make_half2(d, sum);
        }
    }
}

"""

_FUSED_CALL = "    rms_norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,\n"

_FUSED_GATE = r"""#if defined(GGML_USE_HIP)
    // bigcherry 1309 (PRBE06): decode-shaped fused RMSNorm*weight also produces its Q8_1 activation into the 1307
    // cache, keyed exactly as ggml_cuda_mul_mat_vec_q keys src1 == mul_tensor, so its MMVQ consumer skips the
    // standalone quantize launch.
    static const bool bigcherry_rms_q81 = getenv("BIGCHERRY_RMS_Q81") != nullptr && atoi(getenv("BIGCHERRY_RMS_Q81")) != 0;
    if (bigcherry_rms_q81 && ggml_hip_q81_cache_mode_get() != GGML_HIP_Q81_CACHE_OFF && ne00 >= 1024 && ne00 % QK8_1 == 0 &&
            ggml_hip_q81_decode_graph && ggml_is_contiguous(mul_tensor)) {  // decode-shaped graph (1307), not a row cap
        const int64_t ne00_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
        const size_t  q8_bytes    = (size_t) (ne01*ne02*ne03) * (ne00_padded/QK8_1) * sizeof(block_q8_1);
        ggml_hip_q81_cache & q81 = ggml_hip_q81_cache_for_context(ctx);
        const ggml_hip_q81_cache_key key = ggml_hip_q81_cache_make_key(
            q81, mul_tensor, mul_tensor->data, ctx.curr_stream_no,
            ne00, ne00_padded, ne01, ne02, ne03, ne00, ne00*ne01, ne00*ne01*ne02);
        if (ggml_hip_q81_cache_find(q81, key) == nullptr) {
            const ggml_hip_q81_cache_reservation r = ggml_hip_q81_cache_reserve(q81, q8_bytes);
            if (r.ok) {
                const dim3 blocks_num(ne01, ne02, ne03);
                const ggml_cuda_kernel_launch_params launch_params(blocks_num, dim3(1024, 1, 1), 32*sizeof(float), stream);
                ggml_cuda_kernel_launch(rms_norm_mul_q81_f32<1024>, launch_params,
                    src0_d, mul_d, dst_d, (block_q8_1 *) r.ptr, (int) ne00, (int) ne00_padded,
                    s01, s02, s03, mul_s01, mul_s02, mul_s03,
                    init_fastdiv_values(mul_ncols), init_fastdiv_values(mul_nrows),
                    init_fastdiv_values(mul_nchannels), init_fastdiv_values(mul_nsamples), eps);
                ggml_hip_q81_cache_publish(q81, key, r);
                if (getenv("BIGCHERRY_Q81_TRACE") != nullptr) {  // pair with 1307's miss trace
                    GGML_LOG_WARN("BIGCHERRY_Q81 publish-rms gen=%llu node=%p(%s) data=%p ne=%lld,%lld,%lld,%lld\n",
                        (unsigned long long) ggml_hip_q81_cache_current_generation(q81), (const void *) mul_tensor,
                        mul_tensor->name, mul_tensor->data, (long long) ne00, (long long) ne01, (long long) ne02,
                        (long long) ne03);
                }
                static std::atomic<bool> bigcherry_1309_logged{false};  // one context per GPU thread
                if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr && !bigcherry_1309_logged.exchange(true)) {
                    GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1309_rms_norm_mul_q81 ncols=%d rows=%d\n",
                                  (int) ne00, (int) (ne01*ne02*ne03));
                }
                return;
            }
        }
    }
#endif
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/norm.cu",
        description="1309 (PRBE06): fused RMSNorm*weight emits the Q8_1 activation for its MMVQ consumer",
        language="none",
        edits=(
            Edit(
                id="rms-q81-include",
                anchor=re.escape(_INCLUDE_ANCHOR),
                mode="insert_after",
                text=_INCLUDE_TEXT,
                guard=r"bigcherry 1309: Q8_1 activation cache",
                rationale="norm.cu's own header include; the cache API is HIP-only.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="rms-q81-kernel",
                anchor=re.escape(_KERNEL_ANCHOR),
                mode="insert_before",
                text=_KERNEL,
                guard=r"static __global__ void rms_norm_mul_q81_f32\(",
                rationale="Next to the RMSNorm kernels, before rms_norm_back_f32.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="rms-q81-dispatch",
                anchor=re.escape(_FUSED_CALL),
                mode="insert_before",
                text=_FUSED_GATE,
                guard=r"bigcherry 1309 \(PRBE06\): decode-shaped fused RMSNorm\*weight",
                rationale="ggml_cuda_op_rms_norm_fused's single launch (RMS_NORM+MUL without ADD); all of its "
                          "strides and the mul broadcast shape are already computed above it.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
]
