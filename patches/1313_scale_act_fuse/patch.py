"""1313 (QFP13): fuse SCALE -> UNARY(SILU|SIGMOID) [-> SCALE] into one launch, keeping 1310's Q8_1 output.

QFP13 census: ~98 scale_f32 launches per generated token per GPU, most in the Flash-Next hyper-connection blocks
(qwen4exp.cpp build_hc_pre / build_hc_combine): lo = silu(scale(w_down @ xn, 1/hc)) feeding the w_up MMVQ, and
w = scale(sigmoid(scale(inject, 1/hc)), 2) feeding hc_post. Each is 2-3 elementwise launches on a few-KB vector, i.e.
pure launch gap. With BIGCHERRY_SCALE_ACT_FUSE=1, ggml_cuda_try_fuse matches the exact chains (F32, contiguous,
same shape, each intermediate used once, as ggml_can_fuse checks) and runs bc_scale_act_kernel: per element
v = op(s0 * x + b0), then optionally v = s1 * v + b1 - the same expressions, in the same order and precision, as the
separate scale_f32 / unary_op kernels, so the result is bit-identical. When the chain ends at the activation (the silu
feeding MMVQ) and 1310's rules hold (BIGCHERRY_ACT_Q81=1, cache on, decode-shaped graph, row length a multiple of
QK8_1), the same launch also writes MMVQ-padded native Q8_1 blocks under the key MMVQ looks up, exactly as 1310 does
for the unfused activation. Anything else is not fused. Requires 1310 (shared Q8_1 cache include in unary.cu).
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_UNARY_ANCHOR = "/* fused relu + sqr */\n"

_UNARY_TEXT = r"""/* bigcherry 1313: fused SCALE -> UNARY(SILU|SIGMOID) [-> SCALE] */
#if defined(GGML_USE_HIP)  // uses the HIP-only Q8_1 cache (1235/1307/1310)

// Threads run over the padded index space when q81 (rows x n_padded, as 1310) and over the plain range otherwise
// (n_padded == n). n_padded % QK8_1 == 0 and the block size is a multiple of QK8_1, so each Q8_1 group is 32
// consecutive threads that are all in range together.
template <float (*op)(float), bool q81>
static __global__ void bc_scale_act_kernel(const float * x, float * dst, block_q8_1 * yq, const int64_t k_padded,
        const int64_t n, const int64_t n_padded, const float s0, const float b0, const float s1, const float b1,
        const bool post) {
    const int64_t ip = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (ip >= k_padded) {
        return;
    }
    const int64_t row = ip / n_padded;
    const int64_t col = ip % n_padded;
    float v = 0.0f;
    if (col < n) {
        const int64_t i = row * n + col;
        const float scaled = s0 * x[i] + b0;  // scale_f32: scale * x + bias
        v = op(scaled);
        if (post) {
            v = s1 * v + b1;
        }
        dst[i] = v;
    }
    if constexpr (q81) {
        const float amax = warp_reduce_max<QK8_1>(fabsf(v));
        const float sum  = warp_reduce_sum<QK8_1>(v);
        const float d    = amax / 127.0f;
        const int   iqs  = (int) (ip % QK8_1);
        block_q8_1 & b   = yq[ip / QK8_1];
        b.qs[iqs] = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
        if (iqs == 0) {
            b.ds = make_half2(d, sum);
        }
    }
}

template <float (*op)(float)>
static void bc_scale_act_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * scale0, ggml_tensor * dst,
        const ggml_tensor * scale1) {
    const float s0 = ggml_get_op_params_f32(scale0, 0);
    const float b0 = ggml_get_op_params_f32(scale0, 1);
    const float s1 = scale1 ? ggml_get_op_params_f32(scale1, 0) : 1.0f;
    const float b1 = scale1 ? ggml_get_op_params_f32(scale1, 1) : 0.0f;
    const float * x = (const float *) scale0->src[0]->data;
    constexpr int block = 256;
    static_assert(block % QK8_1 == 0, "1313: block must cover whole Q8_1 groups");
    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    static const bool act_q81 = getenv("BIGCHERRY_ACT_Q81") != nullptr && atoi(getenv("BIGCHERRY_ACT_Q81")) != 0;
    if (scale1 == nullptr && act_q81 && ggml_hip_q81_decode_graph && ne0 % QK8_1 == 0 &&
            ggml_hip_q81_cache_mode_get() != GGML_HIP_Q81_CACHE_OFF) {
        const int64_t ne0_padded = GGML_PAD(ne0, MATRIX_ROW_PADDING);  // MMVQ's padded src1 row
        ggml_hip_q81_cache & q81 = ggml_hip_q81_cache_for_context(ctx);
        const ggml_hip_q81_cache_key key = ggml_hip_q81_cache_make_key(
            q81, dst, dst->data, ctx.curr_stream_no, ne0, ne0_padded, ne1, ne2, ne3, ne0, ne0*ne1, ne0*ne1*ne2);
        if (ggml_hip_q81_cache_find(q81, key) == nullptr) {
            const int64_t k_padded = ne0_padded * ne1*ne2*ne3;
            const ggml_hip_q81_cache_reservation r =
                ggml_hip_q81_cache_reserve(q81, (size_t) (k_padded / QK8_1) * sizeof(block_q8_1));
            if (r.ok) {
                bc_scale_act_kernel<op, true><<<(k_padded + block - 1) / block, block, 0, ctx.stream()>>>(
                    x, (float *) dst->data, (block_q8_1 *) r.ptr, k_padded, ne0, ne0_padded, s0, b0, s1, b1, false);
                ggml_hip_q81_cache_publish(q81, key, r);
                if (getenv("BIGCHERRY_Q81_TRACE") != nullptr) {  // pair with 1307's miss trace
                    GGML_LOG_WARN("BIGCHERRY_Q81 publish-scale-act gen=%llu node=%p(%s) data=%p ne=%lld,%lld,%lld,%lld\n",
                        (unsigned long long) ggml_hip_q81_cache_current_generation(q81), (const void *) dst, dst->name,
                        dst->data, (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3);
                }
                return;
            }
        }
    }
    const int64_t k = ggml_nelements(dst);
    bc_scale_act_kernel<op, false><<<(k + block - 1) / block, block, 0, ctx.stream()>>>(
        x, (float *) dst->data, nullptr, k, k, k, s0, b0, s1, b1, scale1 != nullptr);
}

// scale0 = SCALE node, act = UNARY(SILU|SIGMOID) of it, scale1 = optional SCALE of act (output node).
void bc_scale_act_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * scale0, ggml_tensor * act,
        ggml_tensor * scale1) {
    ggml_tensor * dst = scale1 ? scale1 : act;
    switch (ggml_get_unary_op(act)) {
        case GGML_UNARY_OP_SILU:    bc_scale_act_launch<op_silu>(ctx, scale0, dst, scale1);    break;
        case GGML_UNARY_OP_SIGMOID: bc_scale_act_launch<op_sigmoid>(ctx, scale0, dst, scale1); break;
        default: GGML_ABORT("1313: unsupported unary op");
    }
    static std::atomic<bool> logged{false};
    if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr && !logged.exchange(true)) {
        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1313_scale_act_fuse op=%s post=%d n=%lld\n",
                      ggml_unary_op_name(ggml_get_unary_op(act)), scale1 ? 1 : 0, (long long) ggml_nelements(dst));
    }
}
#endif

"""

_DECL_ANCHOR = "void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node);\n"
_DECL_TEXT = ("\n#if defined(GGML_USE_HIP)\n"
              "// bigcherry 1313: fused SCALE -> UNARY(SILU|SIGMOID) [-> SCALE]; scale1 may be nullptr.\n"
              "void bc_scale_act_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * scale0, ggml_tensor * act,\n"
              "        ggml_tensor * scale1);\n"
              "#endif\n")

_FUSE_ANCHOR = ("    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {\n"
                "        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);\n"
                "        return 2;\n"
                "    }\n")

_FUSE_TEXT = r"""
#if defined(GGML_USE_HIP)
    // bigcherry 1313: SCALE -> UNARY(SILU|SIGMOID) [-> SCALE] in one launch (BIGCHERRY_SCALE_ACT_FUSE=1).
    static const bool bc_scale_act = getenv("BIGCHERRY_SCALE_ACT_FUSE") != nullptr && atoi(getenv("BIGCHERRY_SCALE_ACT_FUSE")) != 0;
    if (bc_scale_act && node->op == GGML_OP_SCALE && i + 1 < cgraph->n_nodes) {
        ggml_tensor * act = cgraph->nodes[i + 1];
        const auto plain_f32 = [](const ggml_tensor * t) { return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t); };
        const bool act_ok = act->op == GGML_OP_UNARY && act->src[0] == node &&
            (ggml_get_unary_op(act) == GGML_UNARY_OP_SILU || ggml_get_unary_op(act) == GGML_UNARY_OP_SIGMOID) &&
            plain_f32(node->src[0]) && plain_f32(node) && plain_f32(act) && ggml_are_same_shape(node->src[0], act);
        if (act_ok) {
            const enum ggml_op ops3[] = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE };
            ggml_tensor * post = i + 2 < cgraph->n_nodes ? cgraph->nodes[i + 2] : nullptr;
            if (post && post->op == GGML_OP_SCALE && post->src[0] == act && plain_f32(post) &&
                    ggml_can_fuse(cgraph, i, ops3, 3)) {
                bc_scale_act_fused(*cuda_ctx, node, act, post);
                return 2;
            }
            // leave UNARY -> MUL to upstream's unary_mul fusion (and 1312's Q8_1 output on it)
            const bool act_then_mul = post && post->op == GGML_OP_MUL && (post->src[0] == act || post->src[1] == act);
            if (!act_then_mul && ggml_can_fuse(cgraph, i, ops3, 2)) {
                bc_scale_act_fused(*cuda_ctx, node, act, nullptr);
                return 1;
            }
        }
    }
#endif
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/unary.cu",
        description="1313: fused SCALE -> UNARY(SILU|SIGMOID) [-> SCALE] kernel with 1310-style Q8_1 output",
        language="none",
        edits=(
            Edit(
                id="scale-act-kernel",
                anchor=re.escape(_UNARY_ANCHOR),
                mode="insert_before",
                text=_UNARY_TEXT,
                guard=r"void bc_scale_act_fused\(ggml_backend_cuda_context & ctx",
                rationale="Before the relu+sqr fusion: after op_silu/op_sigmoid and 1310's Q8_1 cache include.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/unary.cuh",
        description="1313: declare bc_scale_act_fused",
        language="none",
        edits=(
            Edit(
                id="scale-act-decl",
                anchor=re.escape(_DECL_ANCHOR),
                mode="insert_after",
                text=_DECL_TEXT,
                guard=r"bigcherry 1313: fused SCALE",
                rationale="Next to the other fused-unary declarations used by ggml_cuda_try_fuse.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1313: match SCALE -> UNARY(SILU|SIGMOID) [-> SCALE] in ggml_cuda_try_fuse",
        language="none",
        edits=(
            Edit(
                id="scale-act-match",
                anchor=re.escape(_FUSE_ANCHOR),
                mode="insert_after",
                text=_FUSE_TEXT,
                guard=r"bigcherry 1313: SCALE -> UNARY",
                rationale="Right after upstream's SCALE/TANH/SCALE softcap fusion, the last matcher in ggml_cuda_try_fuse.",
                expect_matches=1,
                max_span_lines=5,
            ),
        ),
    ),
]
