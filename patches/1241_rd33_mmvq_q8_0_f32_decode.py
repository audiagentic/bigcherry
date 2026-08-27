"""RD33 (AMD-MMV-001): dense Q8_0 decode without activation quantization.

``ggml_cuda_mul_mat_vec_q`` (mmvq.cu) unconditionally quantizes the F32
activation into ``block_q8_1`` format (via ``quantize_row_q8_1_cuda``, a
separate GPU kernel launch plus a pool allocation) before every MMVQ call,
including ``ncols_dst == 1`` -- plain single-token decode, the dominant
shape for ordinary autoregressive generation and MTP speculative-verify.
That quantization step exists to let both weight and activation be
dot-producted as packed int8 (``vec_dot_q8_0_q8_1``, integer accumulation
via dp4a). For n=1 decode this is pure overhead: there is no batching to
amortize it across, and the weight is already the only quantized operand
that matters for a bandwidth-bound matvec.

This patch adds a second code path for that one case: dequantize the Q8_0
weight block directly and dot it against the ORIGINAL F32 activation, with
no q8_1 quantization stage at all. Confirmed against the real pinned
source (2026-08-27, mmvq.cu:1384-1487) that this stage is unconditional and
un-gated for ncols_dst==1 today -- see docs/planning/active/rdna-boost-
experiments/RD33.md for the audit trail and the acceptance methodology.

Design (dev-gpt-agent, verified against our real materialized source
before writing -- session ses_fec603fc33ee4089, req_9b28610e225a45df):

  - Reuses ``mul_mat_vec_q``'s existing template and launch machinery
    rather than duplicating the kernel: a new ``bool f32_act = false``
    template parameter (defaulted, so every existing instantiation is
    unaffected) selects a raw-F32 activation read instead of the
    ``block_q8_1`` read, inside the same kernel body.
  - New device helper ``vec_dot_q8_0_f32`` computes
    ``d_weight * sum(q_weight[i] * y[i])`` in F32 accumulation -- NOT
    equivalent to the existing int32-accumulated q8_1 path within simple
    rounding tolerance; it removes activation-quantization error and
    changes accumulation order. Correctness gate is CPU-reference
    tolerance, not old-path bit parity.
  - ``mul_mat_vec_q_switch_fusion`` gains the same defaulted template
    parameter, threaded into its two existing kernel instantiations --
    the only place ``mul_mat_vec_q<...>`` is actually launched for the
    ncols_dst==1 case this patch targets.
  - New host launcher ``ggml_cuda_mmvq_q8_0_f32_decode`` calls the
    EXISTING ``calc_launch_params``/``mul_mat_vec_q_switch_fusion``
    machinery with ``f32_act=true`` -- no new launch-dimension logic.
  - Eligibility gate in ``ggml_cuda_mul_mat_vec_q``, inserted before the
    q8_1 allocation/quantize call it exists to skip: ``!ids`` (dense
    MUL_MAT only -- MUL_MAT_ID/MoE routing is explicitly out of scope for
    this patch), ``src0->type == GGML_TYPE_Q8_0``, ``ncols_dst == 1``
    (dst's own ne[1], matching the exact ternary the un-patched function
    already uses to derive ncols_dst for the !ids case), and
    ``GGML_CUDA_CC_IS_RDNA3_0(cc)`` (gfx1100 exactly, matching RD30's
    precedent of excluding RDNA3.5 pending its own hardware evidence).
  - ``!forced.requested()`` is part of the gate: a forced autotune
    candidate must never silently execute this structural fast path
    instead of the geometry it was asked to measure, or every MMVQ
    candidate's measured timing would collapse to the same number and the
    "winner" would mean nothing. This is NOT registered as an autotune
    candidate -- it is a compile-time-gated structural bypass, checked
    only when nothing has been forced.

Validation status (2026-08-27): none yet. This patch has not been built,
compiled, or measured on real hardware. STATE stays 'untested' until a
real gfx1100 build (native + tune dispatch modes), MUL_MAT correctness
(test-backend-ops Q8_0, tolerance-based per the note above, not byte
parity), a fused-gate/GLU case, a forced-MMVQ-candidate case (proving the
gate correctly stays off), an ncols_dst>1 case (proving the gate correctly
stays off), and real production-model timing exist -- see RD33.md's
acceptance methodology (interleaved paired A/B, thresholds derived from
this project's own measured noise floor, not a generic percentage).
"""

GROUP = "rdna-boosts"
STATE = "untested"

import re

from bigcherry.patcher import Edit, FilePatch

# Deliberately no PROVENANCE dict: this is not an external backport, so
# there is no source-id/commit to cite. Unlike RD30 (a real, cited external
# commit, AMD-Ecosystem PR #63), this backlog entry (AMD-MMV-001) links only
# to a discussion (ggml-org/llama.cpp#26349), not a specific commit or diff
# -- "source status: recheck before port" in the backlog doc itself. No
# matching implementation was ever found to port; this is an ORIGINAL
# bigcherry design against the current source, inspired by the discussion's
# general hypothesis, not a verbatim port. A PROVENANCE dict citing
# "amd-ecosystem-llama-cpp" here would misrepresent this as a port with a
# real traceable commit, which it is not (tools/bigcherry/source/sources.py
# cross_check_patches() treats any module with a PROVENANCE dict as a claim
# of external backport and validates it against the registry). Traceability
# to RD33 is instead recorded in patches/catalog.toml's plan-item field
# (informational only, not cross-checked against external-sources.toml) --
# the same pattern already used by other locally-designed patches (e.g.
# 1222/1223, HI67).
#
# Design notes:
#   - Original bigcherry design, not a ported diff -- see the module
#     docstring and RD33.md for the real-source audit that confirmed the
#     underlying premise (activation quantization is unconditional for
#     ncols_dst==1 in the current tree) without a specific external commit
#     to port from.
#   - Scoped to GGML_TYPE_Q8_0 only and ncols_dst == 1 only (dense
#     single-token decode) -- the backlog's original description was
#     broader (multiple quant types, small ncols_dst range); this patch
#     targets the one case confirmed live in the current tree and matching
#     this project's actual production workload (dense Qwen 27B + MTP, not
#     MoE).
#   - MUL_MAT_ID (ids != nullptr) explicitly excluded -- out of scope, not
#     attempted.
#   - Gated to cc == GGML_CUDA_CC_RDNA3 exactly (gfx1100), matching RD30's
#     own precedent of excluding RDNA3.5 pending separate hardware
#     evidence.
#   - Implemented by extending the existing mul_mat_vec_q template and
#     mul_mat_vec_q_switch_fusion machinery with one defaulted bool
#     parameter, rather than a standalone kernel/launcher -- reuses all
#     existing launch-dimension and warp/table derivation logic instead of
#     re-deriving it.

# ---------------------------------------------------------------------------
# ggml/src/ggml-cuda/mmvq.cu
# ---------------------------------------------------------------------------

_HELPER = """// bigcherry (RD33/AMD-MMV-001): dequantize a Q8_0 weight block directly
// against a raw F32 activation slice, skipping q8_1 activation
// quantization entirely. Only valid for ncols_dst == 1 (single-token
// decode) -- see the eligibility gate in ggml_cuda_mul_mat_vec_q.
//
// Not bit-equivalent to vec_dot_q8_0_q8_1: that path accumulates in int32
// then scales once; this accumulates d_weight * sum(q_weight * y) in F32.
// Correctness gate is CPU-reference tolerance, not old-path parity.
static __device__ __forceinline__ float vec_dot_q8_0_f32(
        const void * __restrict__ vbq, const float * __restrict__ y_block,
        const int & kbx, const int & iqs) {
    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq + kbx;

    const int elem0 = 4 * iqs;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4 * VDR_Q8_0_Q8_1_MMVQ; ++i) {
        sum = fmaf((float) bq8_0->qs[elem0 + i], y_block[elem0 + i], sum);
    }

    return __half2float(bq8_0->d) * sum;
}

"""

_TEMPLATE_LINE_OLD = (
    "template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false,\n"
    "          bool halve_iters = false, int nwarps_explicit = 0, int rows_per_block_explicit = 0>"
)

# The helper is folded into this same edit (rather than its own, anchored on
# the comment above this template line) because anchors are matched against
# a noise-stripped copy of the file -- comments are blanked before matching,
# so an anchor built from comment text can never match (standards: patch/
# apply.py, "No anchoring into comments"). Anchoring on this code line and
# prepending the helper in the replacement text sidesteps that entirely.
_TEMPLATE_LINE_NEW = (
    _HELPER +
    "template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false,\n"
    "          bool halve_iters = false, int nwarps_explicit = 0, int rows_per_block_explicit = 0,\n"
    "          bool f32_act = false>"
)

# Split from the surrounding kby/kqs lines deliberately: those lines carry a
# trailing/standalone comment, which strip_noise blanks before matching (see
# note on _TEMPLATE_LINE_NEW above). This anchor and _LOOP_BODY_OLD below
# bracket that comment-bearing region without needing to match through it --
# the kby/kqs lines themselves are untouched by this patch.
_Y_DECL_OLD = (
    "    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;\n"
    "    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;"
)

_Y_DECL_NEW = (
    "    const block_q8_1 * y = nullptr;\n"
    "    const float * y_f32 = nullptr;\n"
    "    if constexpr (f32_act) {\n"
    "        y_f32 = ((const float *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;\n"
    "    } else {\n"
    "        y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;\n"
    "    }\n"
    "    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;"
)

_LOOP_BODY_OLD = (
    "#pragma unroll\n"
    "        for (int j = 0; j < ncols_dst; ++j) {\n"
    "#pragma unroll\n"
    "            for (int i = 0; i < rows_per_cuda_block; ++i) {\n"
    "                tmp[j][i] += vec_dot_q_cuda(\n"
    "                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);\n"
    "                if constexpr (has_fusion) {\n"
    "                    if (use_gate) {\n"
    "                        tmp_gate[j][i] += vec_dot_q_cuda(\n"
    "                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }"
)

_LOOP_BODY_NEW = (
    "#pragma unroll\n"
    "        for (int j = 0; j < ncols_dst; ++j) {\n"
    "#pragma unroll\n"
    "            for (int i = 0; i < rows_per_cuda_block; ++i) {\n"
    "                if constexpr (f32_act) {\n"
    "                    tmp[j][i] += vec_dot_q8_0_f32(\n"
    "                        vx, y_f32 + kbx*qk, kbx_offset + i*stride_row_x + kbx, kqs);\n"
    "                    if constexpr (has_fusion) {\n"
    "                        if (use_gate) {\n"
    "                            tmp_gate[j][i] += vec_dot_q8_0_f32(\n"
    "                                vgate, y_f32 + kbx*qk, kbx_offset + i*stride_row_x + kbx, kqs);\n"
    "                        }\n"
    "                    }\n"
    "                } else {\n"
    "                    tmp[j][i] += vec_dot_q_cuda(\n"
    "                        vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);\n"
    "                    if constexpr (has_fusion) {\n"
    "                        if (use_gate) {\n"
    "                            tmp_gate[j][i] += vec_dot_q_cuda(\n"
    "                                vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);\n"
    "                        }\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }"
)

# The switch_fusion template header, its "true"-branch instantiation line,
# and its "false"-branch instantiation are three SEPARATE edits, not one
# block replace: the GGML_ASSERT between them carries a string literal,
# which strip_noise blanks before anchor matching -- same reason the y/kby
# region above is split (see note on _TEMPLATE_LINE_NEW). The GGML_ASSERT
# line and has_fusion computation are untouched by this patch.
_SWITCH_FUSION_TEMPLATE_OLD = (
    "template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false,\n"
    "         int nwarps_explicit = 0, int rows_per_block_explicit = 0>\n"
    "static void mul_mat_vec_q_switch_fusion("
)

_SWITCH_FUSION_TEMPLATE_NEW = (
    "template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false,\n"
    "         int nwarps_explicit = 0, int rows_per_block_explicit = 0, bool f32_act = false>\n"
    "static void mul_mat_vec_q_switch_fusion("
)

_SWITCH_FUSION_TRUE_OLD = (
    "            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, nwarps_explicit, rows_per_block_explicit>, launch_params,"
)

_SWITCH_FUSION_TRUE_NEW = (
    "            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, nwarps_explicit, rows_per_block_explicit, f32_act>, launch_params,"
)

_SWITCH_FUSION_FALSE_OLD = (
    "    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, nwarps_explicit, rows_per_block_explicit>, launch_params,\n"
    "        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,\n"
    "        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,\n"
    "        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);\n"
    "}"
)

_SWITCH_FUSION_FALSE_NEW = (
    "    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, nwarps_explicit, rows_per_block_explicit, f32_act>, launch_params,\n"
    "        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,\n"
    "        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,\n"
    "        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);\n"
    "}\n"
    "\n"
    "// bigcherry (RD33/AMD-MMV-001): host launcher for the Q8_0/f32-activation\n"
    "// decode path. Reuses calc_launch_params and mul_mat_vec_q_switch_fusion\n"
    "// exactly as the native ncols_dst==1 path does -- only f32_act=true and a\n"
    "// null ids differ. Caller (ggml_cuda_mul_mat_vec_q) is responsible for the\n"
    "// full eligibility gate (type, ncols_dst, !ids, cc, !forced).\n"
    "static void ggml_cuda_mmvq_q8_0_f32_decode(\n"
    "        const void * vx, const float * vy, const ggml_cuda_mm_fusion_args_device fusion, float * dst,\n"
    "        const int ncols_x, const int nrows_x,\n"
    "        const int stride_row_x, const int stride_col_dst,\n"
    "        const int nchannels_x, const int nchannels_dst,\n"
    "        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,\n"
    "        const int nsamples_x, const int nsamples_dst,\n"
    "        const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,\n"
    "        cudaStream_t stream) {\n"
    "\n"
    "    constexpr ggml_type type = GGML_TYPE_Q8_0;\n"
    "    constexpr int c_ncols_dst = 1;\n"
    "\n"
    "    const int device = ggml_cuda_get_device();\n"
    "    const int cc = ggml_cuda_info().devices[device].cc;\n"
    "    const int warp_size = ggml_cuda_info().devices[device].warp_size;\n"
    "    const mmvq_parameter_table_id table_id = get_device_table_id(cc);\n"
    "\n"
    "    const uint3 nchannels_y_fd   = make_uint3(0, 0, 0);\n"
    "    const uint3 channel_ratio_fd = init_fastdiv_values(nchannels_dst / nchannels_x);\n"
    "    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);\n"
    "\n"
    "    const std::pair<dim3, dim3> dims = calc_launch_params<type>(\n"
    "        c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);\n"
    "\n"
    "    mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, 0, 0, true>(\n"
    "        vx, (const void *) vy, /*ids=*/nullptr, fusion, dst,\n"
    "        (uint32_t) ncols_x, nchannels_y_fd, (uint32_t) stride_row_x, /*stride_col_y=*/0u,\n"
    "        (uint32_t) stride_col_dst, channel_ratio_fd, (uint32_t) stride_channel_x,\n"
    "        (uint32_t) stride_channel_y, (uint32_t) stride_channel_dst, sample_ratio_fd,\n"
    "        (uint32_t) stride_sample_x, (uint32_t) stride_sample_y, (uint32_t) stride_sample_dst,\n"
    "        dims.first, dims.second, /*nbytes_shared=*/0, /*ids_stride=*/0u, stream);\n"
    "}"
)

_GATE_ANCHOR_OLD = (
    "    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);\n"
    "    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);"
)

_GATE_ANCHOR_NEW = (
    "    // bigcherry (RD33/AMD-MMV-001): dense single-token decode without\n"
    "    // activation quantization. Never taken for a forced autotune\n"
    "    // candidate -- this is a structural fast path, not a dispatch\n"
    "    // candidate, and must not affect measured candidate timing.\n"
    "    if (!ids && src0->type == GGML_TYPE_Q8_0 && ne1 == 1 && !forced.requested()) {\n"
    "        const int rd33_cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;\n"
    "        if (GGML_CUDA_CC_IS_RDNA3_0(rd33_cc)) {\n"
    "            ggml_cuda_mmvq_q8_0_f32_decode(\n"
    "                src0->data, src1_d, fusion_local, dst_d,\n"
    "                (int) ne00, (int) ne01,\n"
    "                (int) (nb01 / ts_src0), (int) (nb1 / ts_dst),\n"
    "                (int) ne02, (int) ne2,\n"
    "                (int) (nb02 / ts_src0), (int) (nb12 / ts_src1), (int) (nb2 / ts_dst),\n"
    "                (int) ne03, (int) ne3,\n"
    "                (int) (nb03 / ts_src0), (int) (nb13 / ts_src1), (int) (nb3 / ts_dst),\n"
    "                stream);\n"
    "            return;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);\n"
    "    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);"
)

MMVQ_CU_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmvq.cu",
    description="RD33: dense Q8_0 decode (ncols_dst==1, no MUL_MAT_ID) "
                "without q8_1 activation quantization",
    edits=(
        Edit(
            id="rd33-template-param",
            anchor=re.escape(_TEMPLATE_LINE_OLD),
            rationale="add a defaulted f32_act template parameter to "
                      "mul_mat_vec_q (with the new device helper prepended "
                      "in the replacement text -- anchors are matched "
                      "against a noise-stripped file and comments are "
                      "blanked before matching, so the helper cannot be "
                      "anchored on the comment above this line). Every "
                      "pre-existing instantiation is unaffected since the "
                      "new parameter defaults to false",
            mode="replace",
            text=_TEMPLATE_LINE_NEW,
            guard=r"bool f32_act = false>",
        ),
        Edit(
            id="rd33-y-decl",
            anchor=re.escape(_Y_DECL_OLD),
            rationale="declare y_f32 alongside y, populating whichever the "
                      "f32_act flag selects; split from the loop body below "
                      "so this anchor does not need to span the kby line's "
                      "trailing comment (blanked before matching)",
            mode="replace",
            text=_Y_DECL_NEW,
            guard=r"float \* y_f32 = nullptr;",
        ),
        Edit(
            id="rd33-loop-body",
            anchor=re.escape(_LOOP_BODY_OLD),
            rationale="branch the activation read and dot-product call on "
                      "f32_act at compile time; the else branch is the "
                      "original code, byte-for-byte",
            mode="replace",
            text=_LOOP_BODY_NEW,
            guard=r"vec_dot_q8_0_f32\(\s*\n\s*vx, y_f32 \+ kbx\*qk",
        ),
        Edit(
            id="rd33-switch-fusion-template",
            anchor=re.escape(_SWITCH_FUSION_TEMPLATE_OLD),
            rationale="add a defaulted f32_act template parameter to "
                      "mul_mat_vec_q_switch_fusion",
            mode="replace",
            text=_SWITCH_FUSION_TEMPLATE_NEW,
            guard=r"rows_per_block_explicit = 0, bool f32_act = false>",
        ),
        Edit(
            id="rd33-switch-fusion-true",
            anchor=re.escape(_SWITCH_FUSION_TRUE_OLD),
            rationale="thread f32_act into the has_fusion=true kernel "
                      "instantiation",
            mode="replace",
            text=_SWITCH_FUSION_TRUE_NEW,
        ),
        Edit(
            id="rd33-switch-fusion-false-and-launcher",
            anchor=re.escape(_SWITCH_FUSION_FALSE_OLD),
            rationale="thread f32_act into the has_fusion=false kernel "
                      "instantiation, and add the host launcher that calls "
                      "this function with f32_act=true directly after it -- "
                      "placed here because both depend on the same "
                      "calc_launch_params/table_id machinery already in "
                      "scope in this part of the file",
            mode="replace",
            text=_SWITCH_FUSION_FALSE_NEW,
            guard=r"ggml_cuda_mmvq_q8_0_f32_decode\(",
        ),
        Edit(
            id="rd33-eligibility-gate",
            anchor=re.escape(_GATE_ANCHOR_OLD),
            rationale="insert the eligibility gate before the q8_1 "
                      "allocation/quantization it exists to skip; falls "
                      "through unchanged to the existing quantize path "
                      "whenever ineligible",
            mode="replace",
            text=_GATE_ANCHOR_NEW,
            guard=r"if \(!ids && src0->type == GGML_TYPE_Q8_0 && ne1 == 1",
        ),
    ),
)

PATCHES = [MMVQ_CU_PATCH]
