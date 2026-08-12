"""Upstream backport: fuse rms_norm + mul + rope (+ view + set_rows).

Cherry-picked from upstream commit 687e7789271ec1276e3470f158428e11a4f80b6f
(merged to ggml-org/llama.cpp master 2026-08-08, PR #26767 -- already merged
and upstream-CI-tested, not an open/unreviewed proposal).

Fuses a chain of per-layer ops that runs every transformer layer, every
token -- rms_norm, mul (scale), rope (rotary position embedding), and
optionally the following view/set_rows -- into a single kernel launch,
instead of the four-to-five separate launches upstream's unfused graph
issues today. Not gated to CUDA-only (no ``GGML_USE_HIP`` guard anywhere in
the diff), applies to HIP builds as-is.

Same category as the already-validated RDNA4 Q6_K/Q2_K MMQ fix in spirit --
a change to what runs, not which candidate this project's autotune dispatch
engine picks, so the tuner cannot reach or discover this on its own. But
it's not a MUL_MAT-family change at all: this is the norm/rope pipeline,
entirely outside anything this session's HI44-56 dispatch-engine work
touches.

All three files this backport touches were checked against the current
pinned base (22dc605) before writing anchors: the new code is purely
additive (new functions appended, new declaration appended, two new
`ggml_cuda_can_fuse`/`ggml_cuda_try_fuse` checks inserted ahead of existing
ones) and every helper it calls (`ggml_cuda_check_fusion_memory_ranges`,
`ggml_cuda_pdl_lc`/`_sync`, `fastmodulo`, `init_fastdiv_values`,
`block_reduce_method`, `rope_yarn`) already exists in this project's base
commit, confirmed by direct grep before any edit was written -- this is not
introducing a dependency our base predates.

Checked for anchor conflicts against this project's own three existing
`ggml-cuda.cu` patches (`0200_dispatch_hook.py`, `0700_coverage_counters.py`,
`0900_pool_workspace_metrics.py`): none of their anchors fall anywhere near
`ggml_cuda_can_fuse`/`ggml_cuda_try_fuse` (~line 2650-3950) -- they touch
`ggml_backend_cuda_free`, `ggml_cuda_mul_mat_id`, and pool alloc/free
functions instead. No real overlap, despite all four patches sharing a file.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "upstream-fixes"


def re_escape_literal(s: str) -> str:
    return re.escape(s)

# ---------------------------------------------------------------- rope.cuh

_ROPE_CUH_DECL = (
    "void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);\n"
    "\n"
    "void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, "
    "ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows);"
)

ROPE_CUH_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/rope.cuh",
    description="Declare ggml_cuda_op_rms_norm_mul_rope_fused (upstream PR #26767)",
    edits=(
        Edit(
            id="rope-cuh-decl",
            anchor=r"^void ggml_cuda_op_rope_fused\(ggml_backend_cuda_context & ctx, "
                   r"ggml_tensor \* dst, ggml_tensor \* set_rows\);$",
            rationale="the declaration of the existing rope+set_rows fusion entry point",
            mode="replace",
            text=_ROPE_CUH_DECL,
            guard=r"void ggml_cuda_op_rms_norm_mul_rope_fused\(",
        ),
    ),
)

# ----------------------------------------------------------------- rope.cu

# Verbatim from the upstream diff -- a purely new, self-contained kernel plus
# its two callers. Reproduced in full rather than reconstructed, since this
# is real numerical kernel code (block reduction, rope_yarn application,
# fastmod-based broadcast indexing for the mul operand) where a
# paraphrase risks a subtle transcription bug the compiler won't catch.
_ROPE_CU_BODY = '''

// fused RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS)
// one block per row: block_reduce gives the norm scale, then each thread applies mul and rope to the elements it owns
template <int block_size, bool has_ff, typename D>
static __global__ void rms_norm_mul_rope_f32(
        const float * x, D * dst, const int ncols,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
        const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox) {
    ggml_cuda_pdl_lc();
    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    x += sample*s03 + channel*s02 + row*s01;

    const uint32_t mul_row     = fastmodulo(row,     mul_nrows_packed);
    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t mul_sample  = fastmodulo(sample,  mul_nsamples_packed);
    mul += mul_sample*mul_s03 + mul_channel*mul_s02 + mul_row*mul_s01;

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float scale = rsqrtf(tmp/ncols + eps);

    int64_t idst = sample*s3 + channel*s2 + row*s1;
    if (set_rows_stride != 0) {
        idst = row*s1 + row_indices[channel]*set_rows_stride;
    }
    dst += idst;

    for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
        int ix0;
        int ix1;
        if (is_neox && i0 < n_dims) {
            ix0 = i0/2;
            ix1 = i0/2 + n_dims/2;
        } else {
            ix0 = i0 + 0;
            ix1 = i0 + 1;
        }

        const float x0 = scale * x[ix0] * mul[fastmodulo(ix0, mul_ncols_packed)];
        const float x1 = scale * x[ix1] * mul[fastmodulo(ix1, mul_ncols_packed)];

        if (i0 >= n_dims) {
            dst[ix0] = ggml_cuda_cast<D>(x0);
            dst[ix1] = ggml_cuda_cast<D>(x1);
            continue;
        }

        const float theta_base  = pos[channel]*powf(theta_scale, i0/2.0f);
        const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

        float cos_theta;
        float sin_theta;
        rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

        dst[ix0] = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
        dst[ix1] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
    }
}

template <typename D>
static void rms_norm_mul_rope_cuda(
        const float * x, D * dst,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint32_t mul_ncols, const uint32_t mul_nrows,
        const uint32_t mul_nchannels, const uint32_t mul_nsamples,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, cudaStream_t stream) {
    GGML_ASSERT(ncols % 2 == 0);

    const dim3 blocks_num(nrows, nchannels, nsamples);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, false, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, true, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox);
        }
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, false, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, true, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox);
        }
    }
}

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx,
        ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * mul_src = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_src->type == GGML_TYPE_F32);
    GGML_ASSERT(rope->type == GGML_TYPE_F32);

    void *          dst_d           = rope->data;
    ggml_type       dst_type        = rope->type;
    const int64_t * row_indices     = nullptr;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        dst_d           = set_rows->data;
        dst_type        = set_rows->type;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }

    const int n_dims     = ((const int32_t *) rope->op_params)[1];
    const int mode       = ((const int32_t *) rope->op_params)[2];
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) rope->op_params + 10, sizeof(float));

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    const int32_t * pos = (const int32_t *) rope->src[1]->data;

    const float * freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    const size_t ts0 = ggml_type_size(x->type);
    GGML_ASSERT(x->nb[0] == ts0);
    const int64_t s01 = x->nb[1] / ts0;
    const int64_t s02 = x->nb[2] / ts0;
    const int64_t s03 = x->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const size_t ts_dst = ggml_type_size(rope->type);
    const int64_t s1 = rope->nb[1] / ts_dst;
    const int64_t s2 = rope->nb[2] / ts_dst;
    const int64_t s3 = rope->nb[3] / ts_dst;

    cudaStream_t stream = ctx.stream();

    if (dst_type == GGML_TYPE_F32) {
        rms_norm_mul_rope_cuda((const float *) x->data, (float *) dst_d,
            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, row_indices, set_rows_stride, is_neox, stream);
    } else if (dst_type == GGML_TYPE_F16) {
        rms_norm_mul_rope_cuda((const float *) x->data, (half *) dst_d,
            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, row_indices, set_rows_stride, is_neox, stream);
    } else {
        GGML_ABORT("fatal error");
    }
}'''

ROPE_CU_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/rope.cu",
    description="Add the fused rms_norm+mul+rope(+set_rows) kernel and its "
                "entry point (upstream PR #26767)",
    edits=(
        Edit(
            id="rope-cu-fused-kernel",
            anchor=r"^void ggml_cuda_op_rope_fused\(ggml_backend_cuda_context & ctx, "
                   r"ggml_tensor \* rope, ggml_tensor \* set_rows\) \{\n"
                   r"    ggml_cuda_op_rope_impl<true>\(ctx, rope, set_rows\);\n"
                   r"\}",
            rationale="the last function in the file, ggml_cuda_op_rope_fused",
            mode="insert_after",
            text=_ROPE_CU_BODY,
            guard=r"void ggml_cuda_op_rms_norm_mul_rope_fused\(ggml_backend_cuda_context & ctx,",
        ),
    ),
)

# ------------------------------------------------------------- ggml-cuda.cu

_SHOULD_FUSE_FN = '''

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    return true;
}'''

_CAN_FUSE_BLOCK_OLD = '''    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            return true;
        }
    }'''

_CAN_FUSE_BLOCK_NEW = '''    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }'''

_TRY_FUSE_OLD = "    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {"

_TRY_FUSE_NEW = '''    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {'''

GGML_CUDA_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="Wire the rms_norm+mul+rope fusion into ggml_cuda_can_fuse "
                "and ggml_cuda_try_fuse (upstream PR #26767)",
    edits=(
        Edit(
            id="ggml-cuda-should-fuse-fn",
            # The trailing comment (`// match gated_delta_net`) is blanked in
            # the noise-stripped copy the patcher matches against, so it can't
            # be part of the anchor even though it stays in the file
            # unchanged. Instead span from the function's own (unique)
            # signature, non-greedily, to its first `return true;` + closing
            # brace -- which is its own end, since the body only contains
            # `return false;` before that point -- and insert_after so the
            # body never has to be restated.
            anchor=r"^static bool ggml_cuda_should_fuse_rope_set_rows\(const ggml_tensor \* rope,\n"
                   r"[\s\S]*?"
                   r"    return true;\n\}",
            rationale="the whole of ggml_cuda_should_fuse_rope_set_rows, "
                      "ending at its own closing brace",
            mode="insert_after",
            text=_SHOULD_FUSE_FN,
            guard=r"ggml_cuda_should_fuse_rms_norm_mul_rope\(const ggml_tensor \* rms_norm,",
            max_span_lines=40,
        ),
        Edit(
            id="ggml-cuda-can-fuse-block",
            anchor=r"    std::initializer_list<enum ggml_op> rope_set_rows_ops = "
                   r"\{ GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS \};\n"
                   r"\n"
                   r"    if \(is_equal\(rope_set_rows_ops, ops\) && "
                   r"ggml_can_fuse_subgraph\(cgraph, node_idx, ops, \{ node_idx \+ 2 \}\)\) \{\n"
                   r"        const ggml_tensor \* rope     = cgraph->nodes\[node_idx\];\n"
                   r"        const ggml_tensor \* view     = cgraph->nodes\[node_idx \+ 1\];\n"
                   r"        const ggml_tensor \* set_rows = cgraph->nodes\[node_idx \+ 2\];\n"
                   r"\n"
                   r"        if \(ggml_cuda_should_fuse_rope_set_rows\(rope, view, set_rows\)\) \{\n"
                   r"            return true;\n"
                   r"        \}\n"
                   r"    \}",
            rationale="the rope+view+set_rows fusion check inside "
                      "ggml_cuda_can_fuse, which the new rms_norm+mul+rope "
                      "checks are inserted ahead of, plus its own return "
                      "true replaced with the memory-range-checked form",
            mode="replace",
            text=_CAN_FUSE_BLOCK_NEW,
            guard=r"rms_norm_mul_rope_set_rows_ops = \{ GGML_OP_RMS_NORM, GGML_OP_MUL, "
                  r"GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS \};",
            max_span_lines=15,
        ),
        Edit(
            id="ggml-cuda-try-fuse-block",
            anchor=re_escape_literal(_TRY_FUSE_OLD),
            rationale="the RMS_NORM+MUL+ADD fusion attempt inside "
                      "ggml_cuda_try_fuse, ahead of which the two new "
                      "rms_norm+mul+rope attempts are inserted",
            mode="replace",
            text=_TRY_FUSE_NEW,
            guard=r"ggml_cuda_op_rms_norm_mul_rope_fused\(\*cuda_ctx, node, "
                  r"cgraph->nodes\[i \+ 1\], cgraph->nodes\[i \+ 2\], cgraph->nodes\[i \+ 4\]\);",
        ),
    ),
)


PATCHES = [ROPE_CUH_PATCH, ROPE_CU_PATCH, GGML_CUDA_PATCH]
