"""1355 (QFP35): fuse SCALE -> SIGMOID -> SCALE directly into DSV4_HC_POST."""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_A_CUH = r"""void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
"""
_N_CUH = r"""void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// BigCherry 1355: consume a SCALE -> SIGMOID -> SCALE gate inside HC_POST.
void bc_cuda_op_dsv4_hc_post_gate(
        ggml_backend_cuda_context & ctx, const ggml_tensor * scale0, ggml_tensor * dst);
"""

_A_KERNEL_SITE = r"""void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
"""
_N_KERNEL = r"""// BigCherry 1355 (QFP35): same HC_POST arithmetic as the 1344 grid kernel, but compute the
// single-use SCALE -> SIGMOID -> SCALE gate from its source instead of materializing an F32 tensor.
template <bool has_comb>
static __global__ void bc_dsv4_hc_post_gate_grid_f32(
        const float * x,
        const float * residual,
        const float * gate_src,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sg0,
        int64_t sg1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float gate_s0,
        float gate_b0,
        float gate_s1,
        float gate_b1) {
    ggml_cuda_pdl_lc();
    const int64_t i0   = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t idst = blockIdx.y;
    const int64_t it   = blockIdx.z;

    if (i0 >= n_embd || idst >= hc || it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float gate_scaled = gate_s0 * gate_src[idst*sg0 + it*sg1] + gate_b0;
    const float gate_sigmoid = 1.0f / (1.0f + expf(-gate_scaled));
    const float gate = gate_s1 * gate_sigmoid + gate_b1;

    float sum = x[i0*sx0 + it*sx1] * gate;
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void bc_cuda_op_dsv4_hc_post_gate(
        ggml_backend_cuda_context & ctx, const ggml_tensor * scale0, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * scale1   = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];
    const ggml_tensor * gate_src = scale0->src[0];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(gate_src->type == GGML_TYPE_F32);
    GGML_ASSERT(scale0->type == GGML_TYPE_F32);
    GGML_ASSERT(scale1->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbg, gate_src, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const float gate_s0 = ggml_get_op_params_f32(scale0, 0);
    const float gate_b0 = ggml_get_op_params_f32(scale0, 1);
    const float gate_s1 = ggml_get_op_params_f32(scale1, 0);
    const float gate_b1 = ggml_get_op_params_f32(scale1, 1);

    constexpr int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_embd + block_size - 1) / block_size, hc, n_tokens);
    const ggml_cuda_kernel_launch_params launch_params =
            ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    if (comb) {
        ggml_cuda_kernel_launch(bc_dsv4_hc_post_gate_grid_f32<true>, launch_params,
                (const float *) x->data, (const float *) residual->data, (const float *) gate_src->data,
                (const float *) comb->data, (float *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / sizeof(float), nbx1 / sizeof(float),
                nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
                nbg0 / sizeof(float), nbg1 / sizeof(float),
                nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
                nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
                gate_s0, gate_b0, gate_s1, gate_b1);
    } else {
        ggml_cuda_kernel_launch(bc_dsv4_hc_post_gate_grid_f32<false>, launch_params,
                (const float *) x->data, (const float *) residual->data, (const float *) gate_src->data,
                nullptr, (float *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / sizeof(float), nbx1 / sizeof(float),
                nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
                nbg0 / sizeof(float), nbg1 / sizeof(float),
                0, 0, 0,
                nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
                gate_s0, gate_b0, gate_s1, gate_b1);
    }
}

"""

_A_FUSE_SITE = r"""#if defined(GGML_USE_HIP)
    // bigcherry 1313: SCALE -> UNARY(SILU|SIGMOID) [-> SCALE] in one launch (BIGCHERRY_SCALE_ACT_FUSE=1).
"""
_N_FUSE = r"""#if defined(GGML_USE_HIP)
    // BigCherry 1355 (QFP35): consume the HC gate directly in DSV4_HC_POST before 1313 materializes it.
    static const bool bc_hc_post_gate_fuse = [] {
        const char * s = getenv("BIGCHERRY_HC_POST_GATE_FUSE");
        return s == nullptr || atoi(s) != 0;
    }();
    if (bc_hc_post_gate_fuse && node->op == GGML_OP_SCALE && i + 3 < cgraph->n_nodes) {
        ggml_tensor * act    = cgraph->nodes[i + 1];
        ggml_tensor * scale1 = cgraph->nodes[i + 2];
        ggml_tensor * hcpost = cgraph->nodes[i + 3];
        const bool chain =
            act->op == GGML_OP_UNARY && ggml_get_unary_op(act) == GGML_UNARY_OP_SIGMOID && act->src[0] == node &&
            scale1->op == GGML_OP_SCALE && scale1->src[0] == act &&
            hcpost->op == GGML_OP_DSV4_HC_POST && hcpost->src[2] == scale1 &&
            node->src[0]->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
            act->type == GGML_TYPE_F32 && scale1->type == GGML_TYPE_F32 && hcpost->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(node->src[0]) && ggml_is_contiguous(node) &&
            ggml_is_contiguous(act) && ggml_is_contiguous(scale1) &&
            ggml_are_same_shape(node->src[0], scale1) &&
            hcpost->src[1] != nullptr && hcpost->src[1]->ne[1] <= 65535 &&
            hcpost->src[0] != nullptr && hcpost->src[0]->ne[1] <= 65535;
        const ggml_op ops[] = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };
        const int output = i + 3;
        if (chain && ggml_can_fuse_subgraph(cgraph, i, 4, ops, &output, 1) &&
                ggml_cuda_check_fusion_memory_ranges(cgraph, i, 4, &output, 1)) {
            bc_cuda_op_dsv4_hc_post_gate(*cuda_ctx, node, hcpost);
            static std::atomic<bool> bc_logged{false};
            if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr && !bc_logged.exchange(true)) {
                GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1355_hc_post_gate_fuse tokens=%lld hc=%lld\n",
                        (long long) hcpost->src[0]->ne[1], (long long) hcpost->src[1]->ne[1]);
            }
            return 3;
        }
    }

#endif
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/dsv4-hc.cuh",
        description="1355: declare HC_POST gate-consumer fusion",
        language="none",
        edits=(
            Edit(
                id="hc-post-gate-decl",
                anchor=re.escape(_A_CUH),
                mode="replace",
                text=_N_CUH,
                guard=r"bc_cuda_op_dsv4_hc_post_gate",
                rationale="Adjacent to the native HC_POST declaration used by ggml-cuda.cu.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/dsv4-hc.cu",
        description="1355: compute SCALE/SIGMOID/SCALE gate inside HC_POST grid kernel",
        language="none",
        edits=(
            Edit(
                id="hc-post-gate-kernel",
                anchor=re.escape(_A_KERNEL_SITE),
                mode="insert_before",
                text=_N_KERNEL,
                guard=r"bc_dsv4_hc_post_gate_grid_f32",
                rationale="Narrow function-signature anchor after 1344's grid variants and before HC comb dispatch.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1355: recognize the four-node HC gate/POST chain before 1313",
        language="none",
        edits=(
            Edit(
                id="hc-post-gate-match",
                anchor=re.escape(_A_FUSE_SITE),
                mode="insert_before",
                text=_N_FUSE,
                guard=r"patch=1355_hc_post_gate_fuse",
                rationale="Immediately before 1313's SCALE/activation matcher so the longer single-consumer chain wins first.",
                expect_matches=1,
                max_span_lines=3,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc(
        "BIGCHERRY_HC_POST_GATE_FUSE",
        "0|1",
        "1 (on)",
        "fuse SCALE->SIGMOID->SCALE directly into DSV4_HC_POST; 0 restores composed 1313/1344 behavior",
    ),
)
