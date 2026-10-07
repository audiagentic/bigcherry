"""1347: thin F32 matmuls (2..8 output rows) through the vector kernel with the roles swapped (QFP34).

Flash-Next's hyper-connection blocks project the 10240-wide normalised state onto 4 values per token
(hc_attn_inject / hc_ffn_inject, F32 [10240, 4]), twice per layer. For a prefill batch that is a GEMM with a 4-row
weight and 512 columns, and ggml_cuda_mul_mat sends it to rocBLAS SGEMM, whose tiled kernel is built for large square
problems. In the prefill kernel profile of the production build (run gate0-d24576) the SGEMM kernel that serves these
calls and the router (Cijk_..._MT64x64x8) is 12.7% of the kernel time of an XTX at 332 us per call, 141 calls per
512-token chunk; the 4-row projections are 96 of them.

Upstream already handles the one-row case of this shape by swapping the roles ("A transposed vector can still use
MMVQ"): the activation matrix is treated as the matrix and the single weight row as the vector, through
mul_mat_vec_f. This patch does the same for 2..8 weight rows, the batch width the vector kernel supports: one launch
computes the result transposed ([tokens, rows], from the pool) and a small kernel writes it into dst.

The sums are formed in another order than rocBLAS forms them, so the result is equal within float tolerance, not
bit-identical. On by default; BIGCHERRY_F32_THIN_MMVF=0 restores SGEMM.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_A_FN = "static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {\n"
_N_HELPERS = r"""// bigcherry 1347 (QFP34): dst[m, n] = src[n, m] for the thin transposed matmul below (n_rows = 2..8)
static __global__ void bc_f32_thin_transpose(const float * __restrict__ src, float * __restrict__ dst,
        const int64_t n_cols, const int64_t n_rows, const int64_t stride_col_dst) {
    const int64_t n = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= n_cols) {
        return;
    }
    for (int64_t m = 0; m < n_rows; ++m) {
        dst[m + n*stride_col_dst] = src[n + m*n_cols];
    }
}

// bigcherry 1347: on by default, BIGCHERRY_F32_THIN_MMVF=0 restores SGEMM
static bool bc_f32_thin_mmvf() {
    static const bool on = [] {
        const char * s = getenv("BIGCHERRY_F32_THIN_MMVF");
        return s == nullptr || atoi(s) != 0;
    }();
    return on;
}

"""

_A_MMF = "    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {\n        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);\n"
_N_THIN = r"""    // bigcherry 1347 (QFP34): a thin F32 weight (2..8 rows) against many columns - the one-row role swap above for the
    // batch width the vector kernel supports. The activation matrix is the matrix, the weight rows are the vectors;
    // the result comes out transposed ([columns, rows]) and a small kernel writes it into dst.
    if (bc_f32_thin_mmvf() && ne01 >= 2 && ne01 <= MMVF_MAX_BATCH_SIZE && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, warp_size, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        static bool bc_logged = false;
        if (!bc_logged && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
            bc_logged = true;
            fprintf(stderr, "BIGCHERRY_PATCH_HIT patch=1347_f32_thin_transposed_mmvf k=%lld rows=%lld cols=%lld\n",
                    (long long) ne00, (long long) ne01, (long long) ne11);
        }
        ggml_cuda_pool_alloc<float> bc_dst_t(ctx.pool(), ne11*ne01);
        ggml_tensor dst_t = *dst;
        dst_t.data  = bc_dst_t.get();
        dst_t.ne[0] = ne11;
        dst_t.ne[1] = ne01;
        dst_t.nb[0] = sizeof(float);
        dst_t.nb[1] = dst_t.nb[0]*ne11;
        dst_t.nb[2] = dst_t.nb[1]*ne01;
        dst_t.nb[3] = dst_t.nb[2];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_t);

        const int bc_block = 256;
        const dim3 bc_grid((ne11 + bc_block - 1)/bc_block, 1, 1);
        bc_f32_thin_transpose<<<bc_grid, bc_block, 0, ctx.stream()>>>(
                bc_dst_t.get(), (float *) dst->data, ne11, ne01, dst->nb[1]/sizeof(float));
        return;
    }
""" + _A_MMF

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1347: F32 matmuls with a 2..8-row weight and many columns through the vector kernel (roles swapped)",
        language="none",
        edits=(
            Edit(
                id="f32-thin-helpers",
                anchor=re.escape(_A_FN),
                mode="insert_before",
                text=_N_HELPERS,
                guard=r"static __global__ void bc_f32_thin_transpose\(",
                rationale="Directly before the matmul dispatch function, the only user.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="f32-thin-dispatch",
                anchor=re.escape(_A_MMF),
                mode="replace",
                text=_N_THIN,
                guard=r"BIGCHERRY_PATCH_HIT patch=1347_f32_thin_transposed_mmvf",
                rationale="The plain-matmul float GEMM test in ggml_cuda_mul_mat (mul_mat_id = false), which follows "
                          "upstream's one-row role swap; the thin case goes in front of it.",
                expect_matches=1,
                max_span_lines=3,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_F32_THIN_MMVF", "0|1", "1 (on)",
           "F32 matmuls with a 2..8-row weight and more than 8 columns (Qwen4Exp hyper-connection inject projections) "
           "run through the vector kernel with the roles swapped instead of SGEMM; 0 restores SGEMM"),
)
