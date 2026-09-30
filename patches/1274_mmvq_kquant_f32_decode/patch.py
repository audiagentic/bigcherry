"""1274: dense Q4_K/Q6_K decode without Q8_1 activation quantization.

Requires 1241_rd33_mmvq_q8_0_f32_decode and extends its f32_act kernel path.
Q4_K/Q6_K keep the existing MMVQ lane/block mapping but dot dequantized weight
values directly against the original F32 activation for dense ncols_dst==1 on
gfx1100. Q5_K is intentionally excluded pending separate evidence because its
extra high-bit-plane unpack raises integer/register cost for the same fixed
activation-quantization saving.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_TEMPLATE_1241 = (
    "template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false,\n"
    "          bool halve_iters = false, int nwarps_explicit = 0, int rows_per_block_explicit = 0,\n"
    "          bool f32_act = false>"
)

_KQUANT_HELPERS = r"""
// bigcherry 1274: raw-F32 decode helpers for K-quants. Preserve the native
// MMVQ lane mapping, but replace the Q8_1 activation loads/dp4a with F32 FMAs.
static __device__ __forceinline__ float kquant_dot_u8x4_f32(
        const int packed, const float * __restrict__ y) {
    const uint8_t * q = (const uint8_t *) &packed;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum = fmaf((float) q[i], y[i], sum);
    }
    return sum;
}

static __device__ __forceinline__ float kquant_dot_i8x4_f32(
        const int packed, const float * __restrict__ y) {
    const int8_t * q = (const int8_t *) &packed;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum = fmaf((float) q[i], y[i], sum);
    }
    return sum;
}

static __device__ __forceinline__ float kquant_sum4_f32(const float * __restrict__ y) {
    return (y[0] + y[1]) + (y[2] + y[3]);
}

static __device__ __forceinline__ float vec_dot_q4_K_f32(
        const void * __restrict__ vbq, const float * __restrict__ y_block,
        const int & kbx, const int & iqs) {
    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
    const int * q4 = (const int *) (bq4_K->qs + 16*bq8_offset + 4*((iqs/2)%4));
    const int v0 = q4[0];
    const int v1 = q4[4];

    const uint16_t * scales = (const uint16_t *) bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *) aux;
    const uint8_t * m  = sc + 2;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v0 >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v1 >> (4*i)) & 0x0F0F0F0F;
        const float * yi = y_block + (bq8_offset + i)*QK8_1 + 4*((iqs/2)%4);
        const float dot = kquant_dot_u8x4_f32(v0i, yi)
                        + kquant_dot_u8x4_f32(v1i, yi + 16);
        const float sumy = kquant_sum4_f32(yi) + kquant_sum4_f32(yi + 16);
        sumf_d = fmaf((float) sc[i], dot,  sumf_d);
        sumf_m = fmaf((float) m[i],  sumy, sumf_m);
    }

    const float2 dm = __half22float2(bq4_K->dm);
    return fmaf(dm.x, sumf_d, -dm.y*sumf_m);
}

static __device__ __forceinline__ float vec_dot_q6_K_f32(
        const void * __restrict__ vbq, const float * __restrict__ y_block,
        const int & kbx, const int & iqs) {
    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    const int bq8_offset = 2*QR6_K*(iqs/(QI6_K/2)) + (iqs%(QI6_K/2))/(QI6_K/4);
    const int scale_offset = (QI6_K/4)*(iqs/(QI6_K/2)) + (iqs%(QI6_K/2))/(QI6_K/8);
    const int vh_shift = 2*((iqs%(QI6_K/2))/(QI6_K/4));

    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(
        bq6_K->qh, (QI6_K/4)*(iqs/(QI6_K/2)) + iqs%(QI6_K/4)) >> vh_shift;
    const int8_t * scales = bq6_K->scales + scale_offset;

    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi = __vsubss4((vil | vih), 0x20202020);
        const float * yi = y_block + (bq8_offset + 2*i)*QK8_1 + 4*(iqs % QI8_1);
        sumf = fmaf(
            (float) scales[4*i],
            kquant_dot_i8x4_f32(vi, yi),
            sumf);
    }

    const float d = bq6_K->d;
    return d * sumf;
}

template <ggml_type type>
static __device__ __forceinline__ float vec_dot_f32_decode(
        const void * __restrict__ vbq, const float * __restrict__ y_block,
        const int & kbx, const int & iqs) {
    if constexpr (type == GGML_TYPE_Q8_0) {
        return vec_dot_q8_0_f32(vbq, y_block, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        return vec_dot_q4_K_f32(vbq, y_block, kbx, iqs);
    } else {
        static_assert(type == GGML_TYPE_Q6_K, "1274 f32 decode supports Q8_0/Q4_K/Q6_K only");
        return vec_dot_q6_K_f32(vbq, y_block, kbx, iqs);
    }
}

"""

_F32_VX_OLD = (
    "                    tmp[j][i] += vec_dot_q8_0_f32(\n"
    "                        vx, y_f32 + j*stride_col_y + kbx*qk, kbx_offset + i*stride_row_x + kbx, kqs);"
)
_F32_VX_NEW = _F32_VX_OLD.replace("vec_dot_q8_0_f32(", "vec_dot_f32_decode<type>(")

_F32_GATE_OLD = (
    "                            tmp_gate[j][i] += vec_dot_q8_0_f32(\n"
    "                                vgate, y_f32 + j*stride_col_y + kbx*qk, kbx_offset + i*stride_row_x + kbx, kqs);"
)
_F32_GATE_NEW = _F32_GATE_OLD.replace("vec_dot_q8_0_f32(", "vec_dot_f32_decode<type>(")

_LAUNCHER_HEAD_OLD = """template <int c_ncols_dst>
static void ggml_cuda_mmvq_q8_0_f32_decode(
        const void * vx, const float * vy, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst,
        const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream) {

    constexpr ggml_type type = GGML_TYPE_Q8_0;
"""

_LAUNCHER_HEAD_NEW = """template <ggml_type type, int c_ncols_dst>
static void ggml_cuda_mmvq_f32_decode(
        const void * vx, const float * vy, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst,
        const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream) {
"""

_Q8_LAUNCH_OLD = "                ggml_cuda_mmvq_q8_0_f32_decode<decltype(ncols)::value>(\n"
_Q8_LAUNCH_NEW = "                ggml_cuda_mmvq_f32_decode<GGML_TYPE_Q8_0, decltype(ncols)::value>(\n"

_Q8_GATE = "    if (!ids && src0->type == GGML_TYPE_Q8_0 && ne1 == 1 && !forced.requested()) {\n"

_KQUANT_GATE = r"""    // bigcherry 1274: extend RD33's no-Q8_1 single-token path to Q4_K/Q6_K.
    // Dense only, gfx1100 only, and never intercept a forced MMVQ candidate.
    if (!ids && ne1 == 1 && !forced.requested() &&
            (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K)) {
        const int kquant_f32_cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        if (GGML_CUDA_CC_IS_RDNA3_0(kquant_f32_cc)) {
            if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
                if (src0->type == GGML_TYPE_Q4_K) {
                    static std::once_flag q4_k_f32_logged;
                    std::call_once(q4_k_f32_logged, [] {
                        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1274_kquant_f32 path=f32_decode type=q4_k ncols=1\n");
                    });
                } else {
                    static std::once_flag q6_k_f32_logged;
                    std::call_once(q6_k_f32_logged, [] {
                        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1274_kquant_f32 path=f32_decode type=q6_k ncols=1\n");
                    });
                }
            }

            const auto kquant_f32_launch = [&](auto type_tag) {
                constexpr ggml_type ktype = decltype(type_tag)::value;
                ggml_cuda_mmvq_f32_decode<ktype, 1>(
                    src0->data, src1_d, fusion_local, dst_d,
                    (int) ne00, (int) ne01,
                    (int) (nb01 / ts_src0), (int) (nb11 / ts_src1), (int) (nb1 / ts_dst),
                    (int) ne02, (int) ne2,
                    (int) (nb02 / ts_src0), (int) (nb12 / ts_src1), (int) (nb2 / ts_dst),
                    (int) ne03, (int) ne3,
                    (int) (nb03 / ts_src0), (int) (nb13 / ts_src1), (int) (nb3 / ts_dst),
                    stream);
            };
            if (src0->type == GGML_TYPE_Q4_K) {
                kquant_f32_launch(std::integral_constant<ggml_type, GGML_TYPE_Q4_K>{});
            } else {
                kquant_f32_launch(std::integral_constant<ggml_type, GGML_TYPE_Q6_K>{});
            }
            return;
        }
    }

"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/mmvq.cu",
        description="1274: Q4_K/Q6_K dense gfx1100 ncols=1 decode directly against F32 activation",
        language="none",
        edits=(
            Edit(
                id="kquant-f32-helpers",
                anchor=re.escape(_TEMPLATE_1241),
                mode="insert_before",
                text=_KQUANT_HELPERS,
                guard=r"static __device__ __forceinline__ float vec_dot_q4_K_f32\(",
                rationale="Attach to 1241's post-patch f32_act template header so the dependency is explicit and fail-closed.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="kquant-f32-vx-dispatch",
                anchor=re.escape(_F32_VX_OLD),
                mode="replace",
                text=_F32_VX_NEW,
                guard=r"tmp\[j\]\[i\] \+= vec_dot_f32_decode<type>\(\s*\n\s*vx,",
                rationale="Generalize 1241's dense F32 dot call from Q8_0 to the compile-time weight type.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="kquant-f32-gate-dispatch",
                anchor=re.escape(_F32_GATE_OLD),
                mode="replace",
                text=_F32_GATE_NEW,
                guard=r"tmp_gate\[j\]\[i\] \+= vec_dot_f32_decode<type>\(\s*\n\s*vgate,",
                rationale="Keep fused gate weights on the same compile-time Q4_K/Q6_K F32 helper.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="kquant-f32-generic-launcher",
                anchor=re.escape(_LAUNCHER_HEAD_OLD),
                mode="replace",
                text=_LAUNCHER_HEAD_NEW,
                guard=r"template <ggml_type type, int c_ncols_dst>\s*\nstatic void ggml_cuda_mmvq_f32_decode\(",
                rationale="Generalize 1241's launcher without changing its launch-parameter machinery.",
                expect_matches=1,
                max_span_lines=20,
            ),
            Edit(
                id="kquant-f32-q8-launch-call",
                anchor=re.escape(_Q8_LAUNCH_OLD),
                mode="replace",
                text=_Q8_LAUNCH_NEW,
                guard=r"ggml_cuda_mmvq_f32_decode<GGML_TYPE_Q8_0, decltype\(ncols\)::value>",
                rationale="Retarget 1241's existing Q8_0 gate to the generalized launcher; behavior remains Q8_0-specific.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="kquant-f32-eligibility-gate",
                anchor=re.escape(_Q8_GATE),
                mode="insert_before",
                text=_KQUANT_GATE,
                guard=r"BIGCHERRY_PATCH_HIT patch=1274_kquant_f32 path=f32_decode type=q6_k ncols=1",
                rationale="Run before 1241's Q8_0 gate and before the shared Q8_1 activation allocation/quantization.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
]
