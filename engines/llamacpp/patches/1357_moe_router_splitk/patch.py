"""1357: the MoE router matmul through a split-K F32 GEMM (QFP36 part a).

Qwen4Exp's router is an F32 weight [n_embd, n_expert] ([2560, 512] in Flash-Next) multiplied by the activation of a
prefill chunk, once per layer. ggml_cuda_mul_mat sends it to rocBLAS SGEMM. In the kernel census of the released build
(run census-rel1-d24576, 75 chunks of 512 tokens) that call is 0.165 ms on an RX 7900 XTX (Cijk_..._MT64x64x8) and
0.432 ms on the R9700 (Cijk_..._MT16x16x16): 7.9 ms and 20.7 ms per chunk. For a 512 x 512 result the SGEMM kernel
launches 64 workgroups, one per 64 x 64 output tile, each walking all of K - fewer workgroups than the card has
compute units.

This patch computes the same product with the same 64 x 64 tile (256 threads, a 4 x 4 block of outputs per thread)
but cuts K into 8 parts, one workgroup per (tile, part), into a pool workspace [parts, N, M]; a second kernel adds the
parts in ascending order into dst. No atomics: the result is the same on every run.

Only the router is taken: src/llama-graph.cpp marks the router's MUL_MAT node (op-param slot 6), and the dispatch
requires the marker. The sums are formed in another order than SGEMM forms them, so the scores are equal within float
tolerance, not bit-identical; the top-k expert choice is discontinuous near ties, which is why this is off by default
(BIGCHERRY_MOE_ROUTER_SPLITK=1 turns it on) until identity is shown on hardware.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_A_LOGITS = "        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]\n"
_N_MARK = _A_LOGITS + r"""        // bigcherry 1357 (QFP36): mark the router matmul for the HIP split-K kernel. build_lora_mm can wrap the
        // matmul (weight scale, LoRA); only the bare MUL_MAT over gate_inp is marked. Slots 0 and 1 (precision, hint)
        // are left alone.
        if (logits->op == GGML_OP_MUL_MAT && logits->src[0] == gate_inp) {
            ((int32_t *) logits->op_params)[6] = 0x42435253; // "BCRS", read by bc_moe_router_marked()
        }
"""

_A_FN = "static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {\n"
_N_HELPERS = r"""// bigcherry 1357 (QFP36): split-K F32 GEMM for the MoE router. W is [K, M] (one weight row per expert), X is [K, N]
// (one column per token); partial[z][n][m] = sum over the z-th part of K of W[k, m]*X[k, n].
// One workgroup per (64 x 64 output tile, part): 16 x 16 threads, each owning a 4 x 4 block of outputs.
#define BC_ROUTER_TILE 64
#define BC_ROUTER_KD   16
#define BC_ROUTER_PARTS 8

static __global__ void bc_moe_router_splitk_partial(const float * __restrict__ W, const float * __restrict__ X,
        float * __restrict__ partial, const int M, const int N, const int K, const int parts) {
    __shared__ float sW[BC_ROUTER_KD*BC_ROUTER_TILE];
    __shared__ float sX[BC_ROUTER_KD*BC_ROUTER_TILE];

    const int m0 = blockIdx.x*BC_ROUTER_TILE;
    const int n0 = blockIdx.y*BC_ROUTER_TILE;
    const int z  = blockIdx.z;
    const int k0 = (int) (((int64_t) K*z)/parts);
    const int k1 = (int) (((int64_t) K*(z + 1))/parts);
    const int tx = threadIdx.x; // 4 expert rows: m0 + 4*tx ..
    const int ty = threadIdx.y; // 4 token columns: n0 + 4*ty ..
    const int tid = ty*16 + tx;

    float acc[4][4] = {};

    for (int kb = k0; kb < k1; kb += BC_ROUTER_KD) {
        // 64 rows x 16 k of each operand; consecutive threads read consecutive k of one row
        for (int l = 0; l < 4; ++l) {
            const int e  = tid + l*256;
            const int r  = e/BC_ROUTER_KD;
            const int kk = e%BC_ROUTER_KD;
            const int k  = kb + kk;
            const int m  = m0 + r;
            const int n  = n0 + r;
            sW[kk*BC_ROUTER_TILE + r] = (k < k1 && m < M) ? W[k + (int64_t) m*K] : 0.0f;
            sX[kk*BC_ROUTER_TILE + r] = (k < k1 && n < N) ? X[k + (int64_t) n*K] : 0.0f;
        }
        __syncthreads();
        for (int kk = 0; kk < BC_ROUTER_KD; ++kk) {
            float a[4];
            float b[4];
            for (int i = 0; i < 4; ++i) {
                a[i] = sW[kk*BC_ROUTER_TILE + 4*tx + i];
                b[i] = sX[kk*BC_ROUTER_TILE + 4*ty + i];
            }
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 4; ++i) {
                    acc[j][i] += a[i]*b[j];
                }
            }
        }
        __syncthreads();
    }

    for (int j = 0; j < 4; ++j) {
        const int n = n0 + 4*ty + j;
        if (n >= N) {
            continue;
        }
        for (int i = 0; i < 4; ++i) {
            const int m = m0 + 4*tx + i;
            if (m < M) {
                partial[((int64_t) z*N + n)*M + m] = acc[j][i];
            }
        }
    }
}

// dst[m, n] = partial[0][n][m] + partial[1][n][m] + ... in ascending part order
static __global__ void bc_moe_router_splitk_reduce(const float * __restrict__ partial, float * __restrict__ dst,
        const int M, const int N, const int parts, const int64_t stride_col_dst) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) M*N) {
        return;
    }
    const int n = (int) (i/M);
    const int m = (int) (i%M);
    float sum = 0.0f;
    for (int z = 0; z < parts; ++z) {
        sum += partial[((int64_t) z*N + n)*M + m];
    }
    dst[m + n*stride_col_dst] = sum;
}

// bigcherry 1357: off by default, BIGCHERRY_MOE_ROUTER_SPLITK=1 turns the split-K router on
static bool bc_moe_router_splitk() {
    static const bool on = [] {
        const char * s = getenv("BIGCHERRY_MOE_ROUTER_SPLITK");
        return s != nullptr && atoi(s) != 0;
    }();
    return on;
}

// the marker src/llama-graph.cpp::build_moe_ffn puts on the router matmul; Meta copies op params to its device nodes
static bool bc_moe_router_marked(const ggml_tensor * dst) {
    return dst->op == GGML_OP_MUL_MAT && ((const int32_t *) dst->op_params)[6] == 0x42435253; // "BCRS"
}

static void bc_moe_router_splitk_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst) {
    const int K = (int) src0->ne[0];
    const int M = (int) src0->ne[1];
    const int N = (int) src1->ne[1];
    ggml_cuda_pool_alloc<float> partial(ctx.pool(), (size_t) BC_ROUTER_PARTS*M*N);

    const dim3 grid((M + BC_ROUTER_TILE - 1)/BC_ROUTER_TILE, (N + BC_ROUTER_TILE - 1)/BC_ROUTER_TILE, BC_ROUTER_PARTS);
    const dim3 block(16, 16, 1);
    bc_moe_router_splitk_partial<<<grid, block, 0, ctx.stream()>>>(
            (const float *) src0->data, (const float *) src1->data, partial.get(), M, N, K, BC_ROUTER_PARTS);

    const int reduce_block = 256;
    const dim3 reduce_grid((unsigned) (((int64_t) M*N + reduce_block - 1)/reduce_block), 1, 1);
    bc_moe_router_splitk_reduce<<<reduce_grid, reduce_block, 0, ctx.stream()>>>(
            partial.get(), (float *) dst->data, M, N, BC_ROUTER_PARTS, (int64_t) (dst->nb[1]/sizeof(float)));
}

"""

_A_MMF = "    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {\n        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);\n"
_N_DISPATCH = r"""    // bigcherry 1357 (QFP36): the marked MoE router (F32 weight [K, experts] against a chunk of tokens) through the
    // split-K GEMM instead of SGEMM. Shape window: at least one 64 x 64 tile in each direction and 64 k per part.
    if (bc_moe_router_splitk() && bc_moe_router_marked(dst) && GGML_CUDA_CC_IS_AMD(cc)
            && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
            && ne00 >= BC_ROUTER_PARTS*64 && ne01 >= BC_ROUTER_TILE && ne11 >= BC_ROUTER_TILE && ne2 == 1 && ne3 == 1
            && src1->ne[0] == ne00 && src1->ne[2] == 1 && src1->ne[3] == 1
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)) {
        static bool bc_logged = false;
        if (!bc_logged && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
            bc_logged = true;
            fprintf(stderr, "BIGCHERRY_PATCH_HIT patch=1357_moe_router_splitk mechanism=router-splitk k=%lld experts=%lld tokens=%lld parts=%d\n",
                    (long long) ne00, (long long) ne01, (long long) ne11, BC_ROUTER_PARTS);
        }
        bc_moe_router_splitk_launch(ctx, src0, src1, dst);
        return;
    }
"""

PATCHES = [
    FilePatch(
        path="src/llama-graph.cpp",
        description="1357: mark the MoE router matmul",
        language="none",
        edits=(
            Edit(
                id="router-mark",
                anchor=re.escape(_A_LOGITS),
                mode="replace",
                text=_N_MARK,
                guard=r"bigcherry 1357 \(QFP36\): mark the router matmul",
                rationale="The one place build_moe_ffn forms the router logits from gate_inp.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/bc-moe-router-splitk.cuh",
        description="1357: BigCherry-owned split-K router kernels, marker test and runtime gate",
        language="none",
        create=True,
        edits=(
            Edit(
                id="router-splitk-owned-file",
                anchor=r"\A",
                mode="insert_after",
                text=_N_HELPERS,
                guard=r"static __global__ void bc_moe_router_splitk_partial\(",
                rationale="BigCherry-owned implementation file; upstream must not provide this path.",
                expect_matches=1,
                max_span_lines=1,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1357: one-line implementation include plus the router dispatch hook",
        language="none",
        edits=(
            Edit(
                id="router-splitk-hook",
                anchor=re.escape(_A_FN),
                mode="insert_before",
                text='#include "bc-moe-router-splitk.cuh"  // bigcherry 1357 implementation\n\n',
                guard=r'#include "bc-moe-router-splitk\.cuh"',
                rationale="Stable ggml_cuda_mul_mat signature immediately before the dispatch using the owned helper.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="router-splitk-dispatch",
                anchor=re.escape(_A_MMF),
                mode="insert_before",
                text=_N_DISPATCH,
                guard=r"BIGCHERRY_PATCH_HIT patch=1357_moe_router_splitk",
                rationale="The plain-matmul float GEMM test in ggml_cuda_mul_mat (mul_mat_id = false); the marked router "
                          "is taken in front of it and of the SGEMM fallback.",
                expect_matches=1,
                max_span_lines=3,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_MOE_ROUTER_SPLITK", "0|1", "0 (off)",
           "The MoE router matmul (F32 weight, at least 64 experts and 64 tokens) runs through a split-K GEMM with a "
           "fixed-order reduction instead of SGEMM; scores equal within float tolerance, not bit-identical"),
)
