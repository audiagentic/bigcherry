"""1311 (QFP13): the hyper-connection pre-mix (GGML_OP_DSV4_HC_PRE) also produces its MMVQ consumer's Q8_1 input.

BIGCHERRY_Q81_TRACE on Flash-Next profile v2 (flashnext-v2-q81c-trace) after 1307-1310: the largest remaining
class of Q8_1-cache misses is MMVQ consuming the output of dsv4_hc_pre_f32 (~2958 of ~8056 misses; census pair
mul_mat_vec_q -> dsv4_hc_pre at ~30/token). Same mechanism as 1309/1310: with BIGCHERRY_HC_Q81=1 and
GGML_HIP_Q8_1_CACHE_MODE=on (1307), an eligible pre-mix (contiguous F32 output, n_embd a multiple of QK8_1, <= 16
tokens) launches dsv4_hc_pre_q81_f32, which computes the same weighted sum, writes the normal F32 result and, in
the same launch, native-layout Q8_1 blocks (rows padded to GGML_PAD(n_embd, MATRIX_ROW_PADDING), zero padding
blocks, quantize_q8_1 math) into a 1235 cache slab published under the key ggml_cuda_mul_mat_vec_q builds for
src1 == this node (1307's reshape-aware lookup also covers contiguous reshapes of it). Threads run over the padded
index space so each 32-element group is 32 consecutive threads. Ineligible shapes or a failed reservation launch
the unchanged kernel. Requires 1307.

Default on since 2026-10-05 (BIGCHERRY_HC_Q81 unset = on); BIGCHERRY_HC_Q81=0 disables. Only hyper-connection
models have the op, so it is inert elsewhere.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_INCLUDE_ANCHOR = '#include "dsv4-hc.cuh"\n'
_INCLUDE_TEXT = ('#if defined(GGML_USE_HIP)\n'
                 '#include "hip-q81-cache.h"  // bigcherry 1311: Q8_1 activation cache (1235/1307)\n'
                 '#include <atomic>\n'
                 '#endif\n')

_KERNEL_ANCHOR = "template <bool has_comb>\nstatic __global__ void dsv4_hc_post_f32(\n"

_KERNEL = r"""#if defined(GGML_USE_HIP)
// bigcherry 1311: dsv4_hc_pre_f32 over the PADDED output index space that also writes native-layout Q8_1 blocks.
// Output must be contiguous ([n_embd, n_tokens]); columns past n_embd are Q8_1 zeros and write no F32.
template <bool gated>
static __global__ void dsv4_hc_pre_q81_f32(
        const float * x, const float * weights, float * dst, block_q8_1 * yq,
        int64_t n_embd, int64_t n_embd_padded, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2, int64_t sw0, int64_t sw1, int64_t sw2, float scale) {
    const int64_t ip = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_embd_padded * n_tokens) {
        return;
    }
    const int64_t it = ip / n_embd_padded;
    const int64_t i0 = ip % n_embd_padded;
    float v = 0.0f;
    if (i0 < n_embd) {
        float sum = 0.0f;
        for (int64_t ih = 0; ih < hc; ++ih) {
            const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
            float wv;
            if constexpr (gated) {
                wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
            } else {
                wv = weights[ih*sw0 + it*sw1];
            }
            sum += xv * wv;
        }
        v = scale * sum;
        dst[it*n_embd + i0] = v;
    }
    const float amax = warp_reduce_max<QK8_1>(fabsf(v));
    const float qsum = warp_reduce_sum<QK8_1>(v);
    const float d    = amax / 127.0f;
    const int   iqs  = (int) (ip % QK8_1);
    block_q8_1 & b   = yq[ip / QK8_1];
    b.qs[iqs] = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
    if (iqs == 0) {
        b.ds = make_half2(d, qsum);
    }
}
#endif

"""

_LAUNCH_OLD = "    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;\n"

_LAUNCH_GATE = r"""#if defined(GGML_USE_HIP)
    // bigcherry 1311: emit the Q8_1 activation for this node's MMVQ consumer into the 1307 cache.
    {
        static const bool bigcherry_hc_q81 = getenv("BIGCHERRY_HC_Q81") == nullptr || atoi(getenv("BIGCHERRY_HC_Q81")) != 0;
        if (bigcherry_hc_q81 && ggml_hip_q81_cache_mode_get() != GGML_HIP_Q81_CACHE_OFF && ggml_is_contiguous(dst) &&
                n_embd % QK8_1 == 0 && ggml_hip_q81_decode_graph && dst->ne[0] == n_embd && dst->ne[1] == n_tokens) {
            const int64_t n_embd_padded = GGML_PAD(n_embd, MATRIX_ROW_PADDING);
            ggml_hip_q81_cache & q81 = ggml_hip_q81_cache_for_context(ctx);
            const ggml_hip_q81_cache_key key = ggml_hip_q81_cache_make_key(
                q81, dst, dst->data, ctx.curr_stream_no, n_embd, n_embd_padded, n_tokens, dst->ne[2], dst->ne[3],
                n_embd, n_embd*n_tokens, n_embd*n_tokens*dst->ne[2]);
            if (ggml_hip_q81_cache_find(q81, key) == nullptr) {
                const ggml_hip_q81_cache_reservation r = ggml_hip_q81_cache_reserve(
                    q81, (size_t) (n_embd_padded * n_tokens / QK8_1) * sizeof(block_q8_1));
                if (r.ok) {
                    const dim3 qgrid((n_embd_padded * n_tokens + block_size - 1) / block_size, 1, 1);
                    const ggml_cuda_kernel_launch_params qparams(qgrid, block_dims, 0, ctx.stream());
                    auto qkernel = gated ? dsv4_hc_pre_q81_f32<true> : dsv4_hc_pre_q81_f32<false>;
                    ggml_cuda_kernel_launch(qkernel, qparams,
                        (const float *) x->data, (const float *) weights->data, (float *) dst->data, (block_q8_1 *) r.ptr,
                        n_embd, n_embd_padded, hc, n_tokens,
                        (int64_t) (nbx0 / sizeof(float)), (int64_t) (nbx1 / sizeof(float)), (int64_t) (nbx2 / sizeof(float)),
                        (int64_t) (nbw0 / sizeof(float)), (int64_t) (nbw1 / sizeof(float)), (int64_t) (nbw2 / sizeof(float)),
                        scale);
                    ggml_hip_q81_cache_publish(q81, key, r);
                    static std::atomic<bool> logged{false};
                    if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr && !logged.exchange(true)) {
                        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1311_hc_pre_q81 n_embd=%lld tokens=%lld\n",
                                      (long long) n_embd, (long long) n_tokens);
                    }
                    return;
                }
            }
        }
    }
#endif
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/dsv4-hc.cu",
        description="1311: hyper-connection pre-mix emits the Q8_1 activation for its MMVQ consumer",
        language="none",
        edits=(
            Edit(
                id="hc-q81-include",
                anchor=re.escape(_INCLUDE_ANCHOR),
                mode="insert_after",
                text=_INCLUDE_TEXT,
                guard=r"bigcherry 1311: Q8_1 activation cache",
                rationale="dsv4-hc.cu's own header include; the cache API is HIP-only.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="hc-q81-kernel",
                anchor=re.escape(_KERNEL_ANCHOR),
                mode="insert_before",
                text=_KERNEL,
                guard=r"static __global__ void dsv4_hc_pre_q81_f32\(",
                rationale="After dsv4_hc_pre_f32 and before dsv4_hc_post_f32.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="hc-q81-dispatch",
                anchor=re.escape(_LAUNCH_OLD),
                mode="insert_before",
                text=_LAUNCH_GATE,
                guard=r"bigcherry 1311: emit the Q8_1 activation",
                rationale="ggml_cuda_op_dsv4_hc_pre's launch: shapes, strides, scale, block_size and block_dims are in scope.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_HC_Q81', '0|1', '1 (on)',
           'hyper-connection pre-mix writes the Q8_1 activation directly (needs Q8_1 cache); 0 disables'),
)
