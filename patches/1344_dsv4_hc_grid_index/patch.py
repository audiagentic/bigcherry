"""1344: hyper-connection PRE / POST kernels indexed by the launch grid instead of 64-bit division (QFP35).

dsv4_hc_pre_f32 and dsv4_hc_post_f32 (ggml-cuda/dsv4-hc.cu) are launched over a flat 1-D grid and every thread
recovers its coordinates with 64-bit `%` and `/` by run-time dimensions (pre: i0 = ir % n_embd, it = ir / n_embd;
post: three of them). AMD GPUs have no 64-bit integer divide: each one is an emulated software loop, executed once
per output element, in kernels that otherwise do a handful of multiply-adds.

This adds grid variants: PRE runs on a 2-D grid (x: n_embd in blocks of 256, y: token), POST on a 3-D grid (x: n_embd,
y: destination stream, z: token), so the coordinates are blockIdx components and no division is left. Each thread
computes exactly the expression the flat kernel computes for the same element, so the result is bit-identical.

On by default (BIGCHERRY_HC_GRID_INDEX=0 restores the flat kernels); a grid dimension is limited to 65535, larger
batches use the flat kernels.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_A_COMB_OP = "void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {\n"
_N_KERNELS = r"""// bigcherry 1344 (QFP35): the same PRE arithmetic per element as dsv4_hc_pre_f32, with the coordinates taken from a
// 2-D launch grid (x: embedding index, y: token) instead of a 64-bit modulo and division per thread.
template <bool gated>
static __global__ void dsv4_hc_pre_grid_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t it = blockIdx.y;

    if (i0 >= n_embd || it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

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

    dst[i0*sd0 + it*sd1] = scale * sum;
}

// bigcherry 1344 (QFP35): POST on a 3-D launch grid (x: embedding index, y: destination stream, z: token).
template <bool has_comb>
static __global__ void dsv4_hc_post_grid_f32(
        const float * x,
        const float * residual,
        const float * post,
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
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t i0   = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t idst = blockIdx.y;
    const int64_t it   = blockIdx.z;

    if (i0 >= n_embd || idst >= hc || it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

// bigcherry 1344: on by default, BIGCHERRY_HC_GRID_INDEX=0 restores the flat kernels
static bool bc_hc_grid_index() {
    static const bool on = [] {
        const char * s = getenv("BIGCHERRY_HC_GRID_INDEX");
        return s == nullptr || atoi(s) != 0;
    }();
    return on;
}

"""

_A_PRE_LAUNCH = """\
    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
"""
_N_PRE_LAUNCH = """\
    // bigcherry 1344 (QFP35): 2-D grid (embedding blocks x tokens), no 64-bit division in the kernel
    const bool bc_grid = bc_hc_grid_index() && n_tokens <= 65535;
    const dim3 bc_grid_dims((n_embd + block_size - 1) / block_size, bc_grid ? n_tokens : 1, 1);
    const ggml_cuda_kernel_launch_params bc_launch_params =
            bc_grid ? ggml_cuda_kernel_launch_params(bc_grid_dims, block_dims, 0, ctx.stream()) : launch_params;
    if (bc_grid) {
        static bool bc_logged = false;
        if (!bc_logged && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
            bc_logged = true;
            fprintf(stderr, "BIGCHERRY_PATCH_HIT patch=1344_dsv4_hc_grid_index path=pre n_embd=%lld tokens=%lld\\n",
                    (long long) n_embd, (long long) n_tokens);
        }
    }
    auto kernel = bc_grid ? (gated ? dsv4_hc_pre_grid_f32<true> : dsv4_hc_pre_grid_f32<false>)
                          : (gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>);
    ggml_cuda_kernel_launch(kernel, bc_launch_params,
"""

_A_POST_LAUNCH = """\
    auto kernel = comb ? dsv4_hc_post_f32<true> : dsv4_hc_post_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
"""
_N_POST_LAUNCH = """\
    // bigcherry 1344 (QFP35): 3-D grid (embedding blocks x streams x tokens), no 64-bit division in the kernel
    const bool bc_grid = bc_hc_grid_index() && hc <= 65535 && n_tokens <= 65535;
    const dim3 bc_grid_dims((n_embd + block_size - 1) / block_size, bc_grid ? hc : 1, bc_grid ? n_tokens : 1);
    const ggml_cuda_kernel_launch_params bc_launch_params =
            bc_grid ? ggml_cuda_kernel_launch_params(bc_grid_dims, block_dims, 0, ctx.stream()) : launch_params;
    if (bc_grid) {
        static bool bc_logged = false;
        if (!bc_logged && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
            bc_logged = true;
            fprintf(stderr, "BIGCHERRY_PATCH_HIT patch=1344_dsv4_hc_grid_index path=post n_embd=%lld hc=%lld tokens=%lld\\n",
                    (long long) n_embd, (long long) hc, (long long) n_tokens);
        }
    }
    auto kernel = bc_grid ? (comb ? dsv4_hc_post_grid_f32<true> : dsv4_hc_post_grid_f32<false>)
                          : (comb ? dsv4_hc_post_f32<true> : dsv4_hc_post_f32<false>);
    ggml_cuda_kernel_launch(kernel, bc_launch_params,
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/dsv4-hc.cu",
        description="1344: hyper-connection PRE / POST kernels on 2-D / 3-D launch grids (no 64-bit division)",
        language="none",
        edits=(
            Edit(
                id="hc-grid-include",
                anchor=re.escape('#include "dsv4-hc.cuh"\n'),
                mode="insert_after",
                text="\n#include <cstdio>   // bigcherry 1344: fprintf\n#include <cstdlib>  // bigcherry 1344: getenv / atoi\n",
                guard=r"#include <cstdlib>  // bigcherry 1344",
                rationale="The file's own header include, the last include of dsv4-hc.cu.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="hc-grid-kernels",
                anchor=re.escape(_A_COMB_OP),
                mode="insert_before",
                text=_N_KERNELS,
                guard=r"static __global__ void dsv4_hc_post_grid_f32\(",
                rationale="After the flat kernels and before the first op function, which is the first to launch one.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="hc-grid-pre-launch",
                anchor=re.escape(_A_PRE_LAUNCH),
                mode="replace",
                text=_N_PRE_LAUNCH,
                guard=r"BIGCHERRY_PATCH_HIT patch=1344_dsv4_hc_grid_index path=pre",
                rationale="The F32 PRE launch: kernel choice and launch call.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="hc-grid-post-launch",
                anchor=re.escape(_A_POST_LAUNCH),
                mode="replace",
                text=_N_POST_LAUNCH,
                guard=r"BIGCHERRY_PATCH_HIT patch=1344_dsv4_hc_grid_index path=post",
                rationale="The POST launch: kernel choice and launch call.",
                expect_matches=1,
                max_span_lines=3,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_HC_GRID_INDEX", "0|1", "1 (on)",
           "Qwen4Exp hyper-connection PRE / POST kernels take their coordinates from a 2-D / 3-D launch grid instead "
           "of a 64-bit division per element (bit-identical); 0 restores the flat kernels"),
)
