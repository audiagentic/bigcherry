// bigcherry: predefined routing transformations (HI28).
//
// Each transformation rewrites the operands of one matmul into an arrangement a
// different family can serve, together with the argument for why the rewrite
// computes the same thing. The tuner then measures both arrangements and keeps
// whichever is faster, so nothing here asserts that a rewrite is an
// improvement -- only that it is *legal*.
//
// The two below are the shapes that fall to BLAS for a reason that is about
// layout rather than arithmetic:
//
//   transpose_weight_for_mmvf -- upstream already does exactly this rewrite by
//       hand (ggml-cuda.cu:1879). Modelling it makes the shape tunable instead
//       of skipped; see the note on that function.
//   batch_for_mmvf -- upstream does not do this, and it is the genuinely open
//       question of the two.
//
// Every rewritten signature is derived by running ggml_hip_make_signature over
// the rewritten tensor headers, never by copying the original signature and
// editing fields. The two are supposed to agree, and the only way to be sure
// they do is to build the second one the same way the first was built: a
// hand-edited copy silently keeps the original's contiguity flags, alignment
// class and padding bits, all of which describe operands that no longer exist.

#include "hip-autotune-transform.cuh"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH) && defined(GGML_HIP_ROUTING_TRANSFORM)

#include "hip-autotune-dispatch.cuh"
#include "hip-autotune-signature.h"

#include "mmvf.cuh"   // MMVF_MAX_BATCH_SIZE

#include <string.h>

namespace {

// Rebuild `out_sig` from the rewritten tensor headers in `ctx`.
//
// `fusion` is deliberately not carried across: a fused pattern is never
// eligible for rewriting in the first place (see
// ggml_hip_transform_signature_is_eligible), so passing it would be describing
// a state that cannot occur.
// Null `out_sig` means the caller does not need one. That is the dispatch path
// (HI31), which re-applies the transform on every launch of a transformed
// binding and only wants the rewritten tensors -- the signature was settled
// when the binding was made, and rebuilding it per launch would be work on the
// hot path whose only consumer is the tuner.
void signature_from_ctx(ggml_hip_transform_ctx * ctx,
                        ggml_hip_dispatch_signature_v1 * out_sig) {
    if (out_sig == nullptr) {
        return;
    }
    *out_sig = ggml_hip_make_signature(&ctx->src0, &ctx->src1,
                                       /*ids =*/ nullptr, &ctx->dst,
                                       /*fusion =*/ nullptr);
}

void launch_context_from_ctx(const ggml_hip_launch_context & orig_lc,
                             ggml_hip_transform_ctx * ctx,
                             ggml_hip_launch_context * out_lc) {
    *out_lc      = orig_lc;
    out_lc->src0 = &ctx->src0;
    out_lc->src1 = &ctx->src1;
    out_lc->dst  = &ctx->dst;
    out_lc->ids  = nullptr;
}

// ------------------------------------------- TRANSPOSE_WEIGHT_FOR_MMVF
//
// F32[1xK] x F32[KxN] with N above MMVF's batch bound. Swapping the operands
// turns the wide-but-single-row product into a single-column one, which MMVF
// serves for any N.
//
// Equivalence is exact, not approximate. With ne01 == 1,
//     dst[0][n] = sum_k src0[k][0] * src1[k][n]
// and after the swap
//     dst'[n][0] = sum_k src1[k][n] * src0[k][0]
// which is the same sum of the same products in the same order -- so the two
// agree bit for bit, not merely to within tolerance. dst is contiguous and
// [1 x N] and [N x 1] have identical linear layouts, so the result also lands
// in the same bytes and the writeback is genuinely nothing.
//
// This is the one rewrite here that upstream already performs
// (ggml-cuda.cu:1879-1891), and the conditions below are that branch's
// conditions. Bigcherry currently *declines* this shape at dispatch time
// (hip-autotune-dispatch.cu:497) precisely because the launch context upstream
// builds is not the one the signature describes -- which is the gap this
// transformation closes. So the gain here is not a new route; it is that the
// route upstream already takes becomes measurable, and the MMVF block-size and
// accumulator variants along it become tunable.

bool transpose_weight_can_apply(const ggml_hip_dispatch_signature_v1 & sig) {
    if (sig.flags & GGML_HIP_SIG_HAS_IDS)          return false;
    if (sig.src0_type != GGML_TYPE_F32)            return false;
    if (sig.src1_type != GGML_TYPE_F32)            return false;
    if (sig.ne0[1] != 1)                           return false;  // ne01 == 1
    if (sig.ned[2] != 1 || sig.ned[3] != 1)        return false;

    // All three contiguous: the rewrite reinterprets dst's strides, which is
    // only the same memory if there were no gaps in it to begin with.
    const uint16_t contig = GGML_HIP_SIG_SRC0_CONTIGUOUS
                          | GGML_HIP_SIG_SRC1_CONTIGUOUS
                          | GGML_HIP_SIG_DST_CONTIGUOUS;
    if ((sig.flags & contig) != contig)            return false;

    // Only worth doing when MMVF cannot already serve the shape as it stands;
    // below the bound the untransformed signature is MMVF-eligible and the
    // rewrite would be measuring the same family twice.
    return sig.ne1[1] > MMVF_MAX_BATCH_SIZE;
}

bool transpose_weight_apply(const ggml_hip_launch_context &  orig_lc,
                            ggml_hip_transform_ctx *         ctx,
                            ggml_hip_dispatch_signature_v1 * out_sig,
                            ggml_hip_launch_context *        out_lc) {
    // The swap itself.
    ctx->src0 = *orig_lc.src1;
    ctx->src1 = *orig_lc.src0;

    // dst [1 x N] reinterpreted as [N x 1]. Strides recomputed rather than
    // swapped: nb[1] of the original describes a row that no longer exists.
    const int64_t n = orig_lc.src1->ne[1];
    ctx->dst        = *orig_lc.dst;
    ctx->dst.ne[0]  = n;
    ctx->dst.ne[1]  = 1;
    ctx->dst.nb[1]  = ctx->dst.nb[0] * n;
    ctx->dst.nb[2]  = ctx->dst.nb[1];
    ctx->dst.nb[3]  = ctx->dst.nb[1];

    ctx->n_batches   = 1;
    ctx->batch_width = n;
    ctx->total_width = n;

    signature_from_ctx(ctx, out_sig);
    launch_context_from_ctx(orig_lc, ctx, out_lc);
    return true;
}

size_t transpose_weight_overhead(const ggml_hip_dispatch_signature_v1 & sig) {
    GGML_UNUSED(sig);
    return 0;   // three rewritten headers on the caller's stack; no device memory
}

// ------------------------------------------------------ BATCH_FOR_MMVF
//
// F32/F16/BF16 [MxK] x F32[KxN] with N above MMVF's batch bound. MMVF's limit
// is on the number of dst columns per launch, not on the total, so splitting
// the product column-wise into ceil(N/8) launches makes every one of them
// eligible.
//
// Equivalence: with B = [B0 | B1 | ... ], A^T B = [A^T B0 | A^T B1 | ... ].
// Each column of dst depends only on the corresponding column of src1, so the
// partition changes which launch computes a column, never its value. Exact,
// again, because no column's arithmetic is reassociated.
//
// Whether it is *worth* doing is an entirely separate question and one this
// file does not answer: replacing a single BLAS GEMM with hundreds of thin
// MMVF launches trades one large kernel for a great deal of launch overhead,
// and for large N the trade is very likely bad. That is the point of routing
// it through the tuner rather than through a heuristic -- the loop is timed
// whole (see ggml_hip_transform_launch), so the overhead is in the number that
// decides it, and a bad trade simply loses.

// Defined below; `apply` uses it to land ctx on batch 0 rather than repeating
// the narrowing arithmetic in two places that could drift apart.
bool batch_for_mmvf_select(ggml_hip_transform_ctx * ctx, int64_t index);

bool batch_for_mmvf_can_apply(const ggml_hip_dispatch_signature_v1 & sig) {
    if (sig.flags & GGML_HIP_SIG_HAS_IDS)          return false;
    if (sig.src1_type != GGML_TYPE_F32)            return false;

    const ggml_type src0 = (ggml_type) sig.src0_type;
    const bool is_float = src0 == GGML_TYPE_F32
                       || src0 == GGML_TYPE_F16
                       || src0 == GGML_TYPE_BF16;
    if (!is_float)                                 return false;

    // MMVF asserts ncols % 2 == 0 on the shared dimension.
    if ((sig.ne0[0] % 2) != 0)                     return false;

    // Only the 2-D case. With ne2/ne3 above 1 the columns of dst are not a
    // single contiguous run and the slicing below would address the wrong
    // batch -- silently, and only for shapes that broadcast.
    if (sig.ned[2] != 1 || sig.ned[3] != 1)        return false;

    const uint16_t contig = GGML_HIP_SIG_SRC1_CONTIGUOUS
                          | GGML_HIP_SIG_DST_CONTIGUOUS;
    if ((sig.flags & contig) != contig)            return false;

    return sig.ned[1] > MMVF_MAX_BATCH_SIZE;
}

bool batch_for_mmvf_apply(const ggml_hip_launch_context &  orig_lc,
                          ggml_hip_transform_ctx *         ctx,
                          ggml_hip_dispatch_signature_v1 * out_sig,
                          ggml_hip_launch_context *        out_lc) {
    const int64_t total = orig_lc.dst->ne[1];
    const int64_t width = MMVF_MAX_BATCH_SIZE;

    ctx->src0 = *orig_lc.src0;   // shared by every batch, untouched
    ctx->src1 = *orig_lc.src1;
    ctx->dst  = *orig_lc.dst;

    ctx->total_width = total;
    ctx->batch_width = width;
    ctx->n_batches   = (total + width - 1) / width;
    ctx->src1_base   = (char *) orig_lc.src1->data;
    ctx->dst_base    = (char *) orig_lc.dst->data;

    // Leaves ctx describing batch 0, which is the widest batch and therefore
    // the one whose eligibility binds: if MMVF can serve a full-width batch it
    // can serve the possibly-narrower last one. A width-specific candidate is
    // still safe on that last batch because no family launcher takes width as a
    // forced argument -- each reads it from the dst header it is handed.
    batch_for_mmvf_select(ctx, 0);

    signature_from_ctx(ctx, out_sig);
    launch_context_from_ctx(orig_lc, ctx, out_lc);
    return true;
}

bool batch_for_mmvf_select(ggml_hip_transform_ctx * ctx, int64_t index) {
    if (index < 0 || index >= ctx->n_batches) {
        return false;
    }
    const int64_t first     = index * ctx->batch_width;
    const int64_t remaining = ctx->total_width - first;
    const int64_t width     = remaining < ctx->batch_width ? remaining
                                                           : ctx->batch_width;

    // src1 is [K x N] and dst is [M x N]; batching partitions N, so both move
    // by whole columns of their own stride. src0 is shared by every batch and
    // is not touched.
    ctx->src1.ne[1] = width;
    ctx->dst.ne[1]  = width;
    ctx->src1.data  = ctx->src1_base + first * ctx->src1.nb[1];
    ctx->dst.data   = ctx->dst_base  + first * ctx->dst.nb[1];
    return true;
}

size_t batch_for_mmvf_overhead(const ggml_hip_dispatch_signature_v1 & sig) {
    GGML_UNUSED(sig);
    // No device allocation: the batches are views into the caller's own src1
    // and dst. The cost of this transform is launch count, not memory, and
    // launch count is already inside the measured time.
    return 0;
}

// ---------------------------------------------------------------- registry

const ggml_hip_routing_transformation s_transforms[] = {
    {
        GGML_HIP_TRANSFORM_TRANSPOSE_WEIGHT_FOR_MMVF,
        "transpose_weight_for_mmvf",
        GGML_HIP_TRANSFORM_SOURCE_PREDEFINED,
        /*equivalence_verified =*/ true,   // same products, same order
        GGML_HIP_FAMILY_MMVF,
        transpose_weight_overhead,
        transpose_weight_can_apply,
        transpose_weight_apply,
        /*select_batch =*/ nullptr,
    },
    {
        GGML_HIP_TRANSFORM_BATCH_FOR_MMVF,
        "batch_for_mmvf",
        GGML_HIP_TRANSFORM_SOURCE_PREDEFINED,
        /*equivalence_verified =*/ true,   // matmul distributes over columns
        GGML_HIP_FAMILY_MMVF,
        batch_for_mmvf_overhead,
        batch_for_mmvf_can_apply,
        batch_for_mmvf_apply,
        batch_for_mmvf_select,
    },
};

} // namespace

int ggml_hip_transform_count() {
    return (int) (sizeof(s_transforms) / sizeof(s_transforms[0]));
}

const ggml_hip_routing_transformation * ggml_hip_transform_at(int index) {
    if (index < 0 || index >= ggml_hip_transform_count()) {
        return nullptr;
    }
    return &s_transforms[index];
}

const ggml_hip_routing_transformation * ggml_hip_transform_find(
        ggml_hip_transform_id id) {
    for (int i = 0; i < ggml_hip_transform_count(); ++i) {
        if (s_transforms[i].id == id) {
            return &s_transforms[i];
        }
    }
    return nullptr;
}

const ggml_hip_routing_transformation * ggml_hip_transform_find_by_name(
        const char * name) {
    if (name == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < ggml_hip_transform_count(); ++i) {
        if (strcmp(s_transforms[i].name, name) == 0) {
            return &s_transforms[i];
        }
    }
    return nullptr;
}

void ggml_hip_transform_launch(
        const ggml_hip_routing_transformation * transform,
        const ggml_hip_candidate_descriptor *   candidate,
        const ggml_hip_variant_params &         variant,
        ggml_hip_transform_ctx *                ctx,
        const ggml_hip_launch_context &         lc) {
    GGML_ASSERT(transform != nullptr);
    GGML_ASSERT(candidate != nullptr && candidate->launch != nullptr);

    // As in ggml_hip_dispatch_launch: launch through a copy so the registry
    // table stays immutable and shareable across threads.
    ggml_hip_candidate_descriptor effective = *candidate;
    effective.variant = variant;

    if (transform->select_batch == nullptr) {
        effective.launch(&effective, lc);
        return;
    }
    for (int64_t i = 0; i < ctx->n_batches; ++i) {
        if (!transform->select_batch(ctx, i)) {
            break;
        }
        // `lc` already points at ctx's headers, so selecting the batch is what
        // re-points the launch; no part of lc is rebuilt per batch.
        effective.launch(&effective, lc);
    }
}

bool ggml_hip_transform_signature_is_eligible(
        const ggml_hip_dispatch_signature_v1 & sig) {
    // Standards 11.1: a fused pattern is a distinct semantic operation. Its
    // operands are not free to rearrange, because the fusion's own arguments
    // (bias, gate) are indexed against the layout being rearranged.
    if (sig.fusion != GGML_HIP_FUSION_NONE) {
        return false;
    }
    // MUL_MAT_ID's routing tensor indexes the original operand layout. A
    // rewrite invalidates those indices, and nothing in the ids says so.
    if (sig.flags & GGML_HIP_SIG_HAS_IDS) {
        return false;
    }
    return true;
}

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH && GGML_HIP_ROUTING_TRANSFORM
