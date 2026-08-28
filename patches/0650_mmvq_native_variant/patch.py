"""HI09 (part 2) - route a forced MMVQ geometry to its compiled instance.

Part 1 (patch 0600) compiles the instances and emits `ggml_hip_mmvq_find_instance`
to resolve a geometry to one. This patch is what makes them reachable.

The routing is threaded down the native chain rather than reimplemented beside
it. `ggml_cuda_mul_mat_vec_q` quantises src1 to q8_1, computes a dozen strides
and three fastdiv triples; `mul_mat_vec_q_switch_ncols_dst` adds the warp size,
the parameter table and the channel/sample ratios. A forced path that derived
those itself would be a second copy of upstream logic, silently drifting on
every release, and any drift would show up as a wrong answer rather than a
build failure. So the forced geometry travels as one small struct down to the
point where everything is already computed, and diverges only at the launch.

That is principle 6 -- native and forced share one launcher -- applied to the
one family that needed genuinely new compiled code.

Three refusals worth keeping:

- No instance for the requested geometry aborts. It cannot mean "run native":
  the tuner would then time the native geometry under an explicit name, and
  every candidate in the family would report the same number.
- MUL_MAT_ID with width > 1 aborts when forced. Upstream routes it to a
  dedicated MoE kernel with no geometry dimension, so there is nothing to force
  and quietly taking that path would misattribute the measurement.
- Fusion is left to the instance. `mul_mat_vec_q_switch_fusion` already branches
  on it at runtime and one compiled instance serves both, so fusion belongs to
  the signature, not the candidate.
"""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch

_FORCED_ROUTE = """
#ifdef GGML_HIP_DISPATCH
    // bigcherry (HI09): a forced geometry diverges here, where every launch
    // argument upstream computes is already in hand and nothing has been
    // launched yet.
    if (forced.requested()) {
        if (has_ids && ncols_dst > 1) {
            GGML_ABORT("bigcherry: forced MMVQ geometry requested for a "
                       "multi-token MUL_MAT_ID, which upstream serves with a "
                       "dedicated MoE kernel that has no geometry dimension. "
                       "ggml_hip_mmvq_can_execute should have rejected this.");
        }

        const ggml_hip_mmvq_instance_fn instance = ggml_hip_mmvq_find_instance(
            type, ncols_dst, forced.nwarps, forced.rows_per_block,
            forced.small_k);
        if (instance == nullptr) {
            // Deliberately fatal rather than a fallback to native. A silent
            // fallback would give every MMVQ candidate the native timing and a
            // winner that means nothing -- far worse than not measuring.
            // Eligibility (ggml_hip_mmvq_can_execute) is what is supposed to
            // keep an unbuildable geometry away from here, so reaching this is
            // a catalog/registry disagreement, not a user error.
            GGML_ABORT("bigcherry: no compiled MMVQ instance for type=%s "
                       "width=%d nwarps=%d rows_per_block=%d small_k=%d. "
                       "Regenerate the catalog for this variant set.",
                       ggml_type_name(type), ncols_dst, forced.nwarps,
                       forced.rows_per_block, (int) forced.small_k);
        }

        ggml_hip_mmvq_launch_args args = {};
        args.vx                 = vx;
        args.vy                 = vy;
        args.ids                = ids;
        args.fusion             = fusion;
        args.dst                = dst;
        args.ncols_x            = ncols_x;
        args.nchannels_y        = nchannels_y_fd;
        args.stride_row_x       = stride_row_x;
        args.stride_col_y       = stride_col_y;
        args.stride_col_dst     = stride_col_dst;
        args.channel_ratio      = channel_ratio_fd;
        args.stride_channel_x   = stride_channel_x;
        args.stride_channel_y   = stride_channel_y;
        args.stride_channel_dst = stride_channel_dst;
        args.sample_ratio       = sample_ratio_fd;
        args.stride_sample_x    = stride_sample_x;
        args.stride_sample_y    = stride_sample_y;
        args.stride_sample_dst  = stride_sample_dst;
        args.ids_stride         = ids_stride;
        args.nrows_x            = nrows_x;
        args.nchannels_dst      = nchannels_dst;
        args.nsamples_or_ntokens = nsamples_dst;
        args.warp_size          = warp_size;
        // Native passes 0 at every call site below; the instance recomputes its
        // own launch dims but not this, so it has to match.
        args.nbytes_shared      = 0;
        args.stream             = stream;

        instance(args);
        return;
    }
#endif
"""

_VARIANT_ENTRY = """
// bigcherry (HI09): forced-variant entry point for measured dispatch.
//
// forced_nwarps == 0 && forced_rows_per_block == 0 means "native policy", and
// forced_small_k is then ignored -- upstream derives small_k from the shape.
void ggml_cuda_mul_mat_vec_q_variant(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion,
        int forced_nwarps, int forced_rows_per_block, bool forced_small_k) {
    ggml_hip_mmvq_forced forced;
    forced.nwarps         = forced_nwarps;
    forced.rows_per_block = forced_rows_per_block;
    forced.small_k        = forced_small_k;

    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst, fusion, forced);
}

"""

_HEADER = FilePatch(
    path="ggml/src/ggml-cuda/mmvq.cuh",
    description="declare the forced-geometry parameter on the MMVQ entry point",
    edits=(
        Edit(
            id="mmvq-cuh-include",
            anchor=r'^#include "common\.cuh"$',
            rationale="the single include at the top of mmvq.cuh",
            # mmvq.cuh has no include guard upstream, but mmvq-autotune.cuh has
            # `#pragma once`, so pulling it in here is safe in both directions.
            text='\n#include "mmvq-autotune.cuh"',
            guard=r'#include "mmvq-autotune\.cuh"',
        ),
        Edit(
            id="mmvq-cuh-signature",
            anchor=r"^void ggml_cuda_mul_mat_vec_q\(ggml_backend_cuda_context & ctx,\n"
                   r"    const ggml_tensor \* src0, const ggml_tensor \* src1, "
                   r"const ggml_tensor \* ids, ggml_tensor \* dst, "
                   r"const ggml_cuda_mm_fusion_args_host \* fusion = nullptr\);",
            rationale="the declaration of the dense MMVQ entry point",
            mode="replace",
            text="void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,\n"
                 "    const ggml_tensor * src0, const ggml_tensor * src1, "
                 "const ggml_tensor * ids, ggml_tensor * dst, "
                 "const ggml_cuda_mm_fusion_args_host * fusion = nullptr,\n"
                 "    ggml_hip_mmvq_forced forced = {});",
            guard=r"ggml_hip_mmvq_forced forced = \{\}\);",
        ),
    ),
)

_SOURCE = FilePatch(
    path="ggml/src/ggml-cuda/mmvq.cu",
    description="route forced MMVQ geometry to its compiled instance (HI09 part 2)",
    edits=(
        # --- thread the forced geometry down the native chain ----------------
        Edit(
            id="mmvq-switch-ncols-signature",
            anchor=r"^        const int ids_stride, cudaStream_t stream\) \{\n"
                   r"\n"
                   r"    GGML_ASSERT\(ncols_x % ggml_blck_size\(type\) == 0\);",
            rationale="mul_mat_vec_q_switch_ncols_dst's parameter list, "
                      "identified by the assertion that opens its body",
            mode="replace",
            text="        const int ids_stride, cudaStream_t stream,\n"
                 "        ggml_hip_mmvq_forced forced = {}) {\n"
                 "\n"
                 "    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);",
            guard=r"ggml_hip_mmvq_forced forced = \{\}\) \{\n"
                  r"\n"
                  r"    GGML_ASSERT\(ncols_x % ggml_blck_size\(type\) == 0\);",
        ),
        Edit(
            id="mmvq-switch-type-signature",
            anchor=r"^        const int ids_stride, cudaStream_t stream\) \{\n"
                   r"    switch \(type_x\) \{",
            rationale="mul_mat_vec_q_switch_type's parameter list, identified "
                      "by the type switch that opens its body",
            mode="replace",
            text="        const int ids_stride, cudaStream_t stream,\n"
                 "        ggml_hip_mmvq_forced forced = {}) {\n"
                 "    switch (type_x) {",
            guard=r"ggml_hip_mmvq_forced forced = \{\}\) \{\n"
                  r"    switch \(type_x\) \{",
        ),
        Edit(
            id="mmvq-switch-type-forward",
            # Every case of the type switch ends with this identical line, which
            # is why the forced geometry is one struct and not three arguments:
            # a future dimension changes the struct, not 24 call sites.
            anchor=r"^                 nsamples_x, nsamples_dst, stride_sample_x, "
                   r"stride_sample_y, stride_sample_dst, ids_stride, stream\);$",
            rationale="the forwarding call in each case of "
                      "mul_mat_vec_q_switch_type",
            mode="replace_all",
            # One per quantised type MMVQ supports. Stated rather than inferred:
            # if a release adds or drops a type the count changes and the patch
            # fails loudly, which is the point.
            expect_matches=23,
            text="                 nsamples_x, nsamples_dst, stride_sample_x, "
                 "stride_sample_y, stride_sample_dst, ids_stride, stream, forced);",
            guard=r"stride_sample_dst, ids_stride, stream, forced\);",
        ),

        # --- diverge to the compiled instance --------------------------------
        Edit(
            id="mmvq-forced-route",
            anchor=r"^    const bool has_ids = ids != nullptr;$",
            rationale="the last local mul_mat_vec_q_switch_ncols_dst computes "
                      "before it starts choosing a launch",
            mode="insert_after",
            text=_FORCED_ROUTE,
            guard=r"if \(forced\.requested\(\)\) \{",
        ),

        # --- carry it in from the public entry points ------------------------
        Edit(
            id="mmvq-public-signature",
            anchor=r"^void ggml_cuda_mul_mat_vec_q\(\n"
                   r"        ggml_backend_cuda_context & ctx, const ggml_tensor \* src0, "
                   r"const ggml_tensor \* src1, const ggml_tensor \* ids, ggml_tensor \* dst,\n"
                   r"        const ggml_cuda_mm_fusion_args_host \* fusion\) \{",
            rationale="ggml_cuda_mul_mat_vec_q, the dense MMVQ entry point",
            mode="replace",
            text="void ggml_cuda_mul_mat_vec_q(\n"
                 "        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, "
                 "const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,\n"
                 "        const ggml_cuda_mm_fusion_args_host * fusion,\n"
                 "        ggml_hip_mmvq_forced forced) {",
            guard=r"const ggml_cuda_mm_fusion_args_host \* fusion,\n"
                  r"        ggml_hip_mmvq_forced forced\) \{",
        ),
        Edit(
            id="mmvq-public-forward",
            # The single call in ggml_cuda_mul_mat_vec_q. The MoE entry point
            # (ggml_cuda_op_mul_mat_vec_q) has its own call and is left native:
            # it is the split-buffer path, which measured dispatch does not
            # reach.
            anchor=r"^        ne03,              ne3,           s03, s13,              "
                   r"s3,               ids_stride, stream\);$",
            rationale="the switch_type call at the end of ggml_cuda_mul_mat_vec_q",
            mode="replace",
            text="        ne03,              ne3,           s03, s13,              "
                 "s3,               ids_stride, stream, forced);",
            guard=r"s3,               ids_stride, stream, forced\);",
        ),
        Edit(
            id="mmvq-variant-entry",
            anchor=r"^void ggml_cuda_op_mul_mat_vec_q\($",
            rationale="the function following ggml_cuda_mul_mat_vec_q, so the "
                      "variant entry point sits after the function it calls",
            mode="insert_before",
            text=_VARIANT_ENTRY,
            guard=r"void ggml_cuda_mul_mat_vec_q_variant\(",
        ),
    ),
)

# The header must land before the source: mmvq.cu's definition has to agree with
# a declaration that already carries the defaulted parameter.
PATCHES = (_HEADER, _SOURCE)
