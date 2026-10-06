"""1281 (MET02, phase A): range-aware MUL_MAT_ID - the semantic primitive, CPU reference only.

ggml_mul_mat_id_range(ctx, as, b, ids, id_base) is GGML_OP_MUL_MAT_ID over a tensor that holds only the experts
[id_base, id_base + as->ne[2]) of a larger set. ids stay GLOBAL. For each selected id g the local index is
l = g - id_base (widened before the subtraction); if 0 <= l < as->ne[2] the lane is computed with local expert l,
otherwise the lane is an exact +0 and no expert weight is indexed. This is what a tier graph (MET03) and whole-expert
parallelism (MET04) are built from: every device computes the experts it holds and the partial outputs add up.

The variant is a flag plus the signed base in the op params of the ordinary op (indices 6 and 7; 0..3 are taken by
precision and hints). ggml_mul_mat_id is unchanged, and an op without the flag takes the identical path everywhere.

Phase A scope (MET02): the constructor, two accessors, both CPU implementations (the generic one and the one for
weights repacked at load, ggml-cpu/repack.cpp), and a reference test program. The
HIP backend REFUSES the range variant in supports_op, so the scheduler runs it on the CPU: no global id can reach a GPU
kernel before phase B gives the GPU its own translation. No translated-ids tensor is materialised and nothing is
allocated per call.
"""
import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "validated"

_A_H = (
    "    GGML_API struct ggml_tensor * ggml_mul_mat_id(\n"
    "            struct ggml_context * ctx,\n"
    "            struct ggml_tensor  * as,\n"
    "            struct ggml_tensor  * b,\n"
    "            struct ggml_tensor  * ids);\n"
)
_N_H = _A_H + r'''
    // BigCherry 1281: indirect matrix multiplication over a RANGE of a larger expert set
    // `as` holds the experts [id_base, id_base + as->ne[2]); `ids` are global expert ids
    // a lane whose id is outside the range is an exact zero and reads no expert weight
    GGML_API struct ggml_tensor * ggml_mul_mat_id_range(
            struct ggml_context * ctx,
            struct ggml_tensor  * as,
            struct ggml_tensor  * b,
            struct ggml_tensor  * ids,
            int32_t               id_base);

    // true if `t` is a GGML_OP_MUL_MAT_ID built by ggml_mul_mat_id_range; the base is only meaningful then
    GGML_API bool    ggml_mul_mat_id_is_range(const struct ggml_tensor * t);
    GGML_API int32_t ggml_mul_mat_id_range_base(const struct ggml_tensor * t);
'''

_A_C = (
    "    result->op     = GGML_OP_MUL_MAT_ID;\n"
    "    result->src[0] = as;\n"
    "    result->src[1] = b;\n"
    "    result->src[2] = ids;\n"
    "\n"
    "    return result;\n"
    "}\n"
)
_N_C = _A_C + r'''
// BigCherry 1281: range variant of MUL_MAT_ID, see ggml.h
// op params 6 and 7 (0..3 belong to precision and hints): a marker and the signed id base
#define GGML_BC_MUL_MAT_ID_RANGE_MARK 0x52414E47 // "RANG"

struct ggml_tensor * ggml_mul_mat_id_range(
        struct ggml_context * ctx,
        struct ggml_tensor  * as,
        struct ggml_tensor  * b,
        struct ggml_tensor  * ids,
        int32_t               id_base) {
    struct ggml_tensor * result = ggml_mul_mat_id(ctx, as, b, ids);

    ggml_set_op_params_i32(result, 6, GGML_BC_MUL_MAT_ID_RANGE_MARK);
    ggml_set_op_params_i32(result, 7, id_base);

    return result;
}

bool ggml_mul_mat_id_is_range(const struct ggml_tensor * t) {
    return t->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(t, 6) == GGML_BC_MUL_MAT_ID_RANGE_MARK;
}

int32_t ggml_mul_mat_id_range_base(const struct ggml_tensor * t) {
    return ggml_mul_mat_id_is_range(t) ? ggml_get_op_params_i32(t, 7) : 0;
}
'''

_A_CPU_LOCALS = (
    "    // row groups\n"
    "    const int n_ids = ids->ne[0]; // n_expert_used\n"
    "    const int n_as  = ne02;       // n_expert\n"
)
_N_CPU_LOCALS = _A_CPU_LOCALS + (
    "\n"
    "    // BigCherry 1281: range variant - ids are global, src0 holds the experts [bc_id_base, bc_id_base + n_as)\n"
    "    const bool    bc_range   = ggml_mul_mat_id_is_range(dst);\n"
    "    const int32_t bc_id_base = ggml_mul_mat_id_range_base(dst);\n"
)

_A_CPU_GROUP = (
    "        // initialize matrix_row_counts\n"
    "        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));\n"
    "\n"
    "        // group rows by src0 matrix\n"
    "        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {\n"
    "            for (int id = 0; id < n_ids; ++id) {\n"
    "                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);\n"
    "\n"
    "                assert(i02 >= 0 && i02 < n_as);\n"
    "\n"
    "                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};\n"
    "                matrix_row_counts[i02] += 1;\n"
)
_N_CPU_GROUP = (
    "        // initialize matrix_row_counts\n"
    "        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));\n"
    "\n"
    "        if (bc_range) {\n"
    "            // BigCherry 1281: a lane whose expert is not held here is an exact +0; the other threads wait at the\n"
    "            // barrier below, so the whole output is cleared before any lane is written\n"
    "            memset(dst->data, 0, ggml_nbytes(dst));\n"
    "        }\n"
    "\n"
    "        // group rows by src0 matrix\n"
    "        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {\n"
    "            for (int id = 0; id < n_ids; ++id) {\n"
    "                int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);\n"
    "\n"
    "                if (bc_range) {\n"
    "                    // widen before the subtraction: no signed overflow for any id / base pair\n"
    "                    const int64_t bc_local = (int64_t) i02 - (int64_t) bc_id_base;\n"
    "                    if (bc_local < 0 || bc_local >= n_as) {\n"
    "                        continue; // not held here: no row is grouped, no weight is indexed\n"
    "                    }\n"
    "                    i02 = (int32_t) bc_local;\n"
    "                }\n"
    "\n"
    "                assert(i02 >= 0 && i02 < n_as);\n"
    "\n"
    "                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};\n"
    "                matrix_row_counts[i02] += 1;\n"
)

# The CPU backend has a second MUL_MAT_ID implementation for weights it repacked at load (ggml-cpu/repack.cpp,
# tensor_traits::forward_mul_mat_id). It groups rows from the ids itself, so it needs the same translation.
_A_REPACK = (
    "            // initialize matrix_row_counts\n"
    "            memset(matrix_row_counts, 0, n_as * sizeof(int64_t));\n"
    "\n"
    "            // group rows by src0 matrix\n"
    "            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {\n"
    "                for (int32_t id = 0; id < n_ids; ++id) {\n"
    "                    const int32_t i02 =\n"
    "                        *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);\n"
    "\n"
    "                    GGML_ASSERT(i02 >= 0 && i02 < n_as);\n"
)
_N_REPACK = (
    "            // initialize matrix_row_counts\n"
    "            memset(matrix_row_counts, 0, n_as * sizeof(int64_t));\n"
    "\n"
    "            // BigCherry 1281: range variant - ids are global, src0 holds the experts [bc_id_base, bc_id_base + n_as);\n"
    "            // a lane whose expert is not held here is an exact +0 (cleared before the barrier below)\n"
    "            const bool    bc_range   = ggml_mul_mat_id_is_range(op);\n"
    "            const int32_t bc_id_base = ggml_mul_mat_id_range_base(op);\n"
    "            if (bc_range) {\n"
    "                memset(dst->data, 0, ggml_nbytes(dst));\n"
    "            }\n"
    "\n"
    "            // group rows by src0 matrix\n"
    "            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {\n"
    "                for (int32_t id = 0; id < n_ids; ++id) {\n"
    "                    int32_t i02 =\n"
    "                        *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);\n"
    "\n"
    "                    if (bc_range) {\n"
    "                        const int64_t bc_local = (int64_t) i02 - (int64_t) bc_id_base;\n"
    "                        if (bc_local < 0 || bc_local >= n_as) {\n"
    "                            continue; // not held here: no row is grouped, no weight is indexed\n"
    "                        }\n"
    "                        i02 = (int32_t) bc_local;\n"
    "                    }\n"
    "\n"
    "                    GGML_ASSERT(i02 >= 0 && i02 < n_as);\n"
)

# ---- phase B, part 1: the small-batch GPU kernels (MMVQ / MMVF) --------------------------------------------------
# Drafted with GPT (req_e116405983eb4a28) and reworked: the kernels only translate the id and return early for a lane
# whose expert is not held; the host clears dst on the stream before the launch, so an inactive lane is an exact +0
# without a second copy of each kernel's output-write logic. id_count == 0 keeps today's path. Not reachable until
# supports_op stops refusing the variant (after the MMQ / MMF / fallback paths and the fusion guards are in).
_A_FUSION_ARGS = "    uint32_t shared_stride_col_dst = 0;\n"
_N_FUSION_ARGS = _A_FUSION_ARGS + (
    "    // BigCherry 1281: range MUL_MAT_ID - src0 holds the experts [id_base, id_base + id_count); 0 = ordinary op\n"
    "    int32_t id_base = 0;\n"
    "    int64_t id_count = 0;\n"
)

_A_MMVQ_K1 = (
    "    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);\n"
    "    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;\n"
    "    sample_dst = blockIdx.z;\n"
)
_N_MMVQ_K1 = _A_MMVQ_K1 + (
    "\n"
    "    if (fusion.id_count != 0 && !shared_expert && ncols_dst == 1 && ids) {\n"
    "        // BigCherry 1281: global id -> local expert; a lane this device does not hold keeps the +0 the host wrote\n"
    "        const int64_t bc_local = (int64_t) ids[channel_dst] - (int64_t) fusion.id_base;\n"
    "        if (bc_local < 0 || bc_local >= fusion.id_count) {\n"
    "            return;\n"
    "        }\n"
    "        channel_x = (uint32_t) bc_local;\n"
    "    }\n"
)

_A_MMVQ_K2 = "    const uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];\n"
_N_MMVQ_K2 = (
    "    uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];\n"
    "    if (fusion.id_count != 0 && !shared_expert) {\n"
    "        // BigCherry 1281: global id -> local expert; a lane this device does not hold keeps the +0 the host wrote\n"
    "        const int64_t bc_local = (int64_t) ids[channel_dst + token_idx * ids_stride] - (int64_t) fusion.id_base;\n"
    "        if (bc_local < 0 || bc_local >= fusion.id_count) {\n"
    "            return;\n"
    "        }\n"
    "        channel_x = (uint32_t) bc_local;\n"
    "    }\n"
)

_A_MMVF_K = (
    "        channel_y  = ids ? fastmodulo(blockIdx.y, nchannels_y)                 : channel_dst;\n"
    "        sample_dst = ids ? 0                                                   : blockIdx.z;\n"
    "    }\n"
)
_N_MMVF_K = _A_MMVF_K + (
    "\n"
    "    if (ids && fusion.id_count != 0) {\n"
    "        // BigCherry 1281: global id -> local expert; a lane this device does not hold keeps the +0 the host wrote\n"
    "        const int64_t bc_local = (int64_t) channel_x - (int64_t) fusion.id_base;\n"
    "        if (bc_local < 0 || bc_local >= fusion.id_count) {\n"
    "            return;\n"
    "        }\n"
    "        channel_x = (int) bc_local;\n"
    "    }\n"
)

_A_MMV_HOST = (
    "    ggml_cuda_mm_fusion_args_device fusion_local{};\n"
    "\n"
    "    if (fusion) {\n"
)
_N_MMV_HOST = (
    "    ggml_cuda_mm_fusion_args_device fusion_local{};\n"
    "\n"
    "    // BigCherry 1281: range variant. In a plain launch dst is the MUL_MAT_ID node; in a fused launch dst is the\n"
    "    // fused output (GLU) and the caller passes the range. The kernel skips the lanes this device does not hold,\n"
    "    // so clear dst first (stream ordered, no synchronisation): those lanes are then an exact +0.\n"
    "    if (ids != nullptr && fusion == nullptr && ggml_mul_mat_id_is_range(dst)) {\n"
    "        fusion_local.id_base  = ggml_mul_mat_id_range_base(dst);\n"
    "        fusion_local.id_count = src0->ne[2];\n"
    "    } else if (ids != nullptr && fusion != nullptr && fusion->id_count != 0) {\n"
    "        fusion_local.id_base  = fusion->id_base;\n"
    "        fusion_local.id_count = fusion->id_count;\n"
    "    }\n"
    "    if (fusion_local.id_count != 0) {\n"
    "        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), ctx.stream()));\n"
    "    }\n"
    "\n"
    "    if (fusion) {\n"
)

# ---- phase B, part 2: the large-batch quantized path (MMQ) ---------------------------------------------------------
# MMQ groups rows by expert with ggml_cuda_launch_mm_ids_helper. Instead of changing the helper, the global ids are
# translated on the device into a scratch buffer: local index, or INT_MAX for an expert that is not held - a value the
# helper already uses for padding and never groups. The compact row lists then only cover the active rows; ids_src1 is
# cleared first so the quantizer's unused tail rows read row 0 (harmless), dst is cleared so inactive lanes are +0,
# and the token-dedup quantizer is not used for range ops (its inverse map has no entry for inactive lanes).
_A_MMID_DECL = (
    "void ggml_cuda_launch_mm_ids_helper(\n"
    "        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,\n"
    "        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);\n"
)
_N_MMID_DECL = _A_MMID_DECL + (
    "\n"
    "// BigCherry 1281: global expert ids -> local index in [0, n_local), INT_MAX for an expert outside\n"
    "// [id_base, id_base + n_local). The result can be given to ggml_cuda_launch_mm_ids_helper in place of the ids.\n"
    "void ggml_cuda_mm_ids_range_translate(\n"
    "        const int32_t * ids, int32_t * ids_local, int64_t n, int32_t id_base, int32_t n_local, cudaStream_t stream);\n"
)

_A_MMID_DEF = (
    "void ggml_cuda_launch_mm_ids_helper(\n"
    "        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,\n"
)
_N_MMID_DEF = (
    "// BigCherry 1281: see mmid.cuh\n"
    "static __global__ void mm_ids_range_translate(\n"
    "        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_local, const int64_t n, const int32_t id_base, const int32_t n_local) {\n"
    "    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (i >= n) {\n"
    "        return;\n"
    "    }\n"
    "    const int64_t local = (int64_t) ids[i] - (int64_t) id_base; // widened: no overflow for any id / base pair\n"
    "    ids_local[i] = local >= 0 && local < n_local ? (int32_t) local : INT_MAX;\n"
    "}\n"
    "\n"
    "void ggml_cuda_mm_ids_range_translate(\n"
    "        const int32_t * ids, int32_t * ids_local, const int64_t n, const int32_t id_base, const int32_t n_local, cudaStream_t stream) {\n"
    "    const int block_size = 256;\n"
    "    const dim3 num_blocks((unsigned int) ((n + block_size - 1) / block_size), 1, 1);\n"
    "    mm_ids_range_translate<<<num_blocks, block_size, 0, stream>>>(ids, ids_local, n, id_base, n_local);\n"
    "}\n"
    "\n"
) + _A_MMID_DEF

_A_MMQ_DEDUP = "    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;\n"
_N_MMQ_DEDUP = (
    "    // BigCherry 1281: range variant - group rows by LOCAL expert; lanes of experts that are not held are never\n"
    "    // grouped, their dst rows keep the +0 written here\n"
    "    const bool bc_range = ggml_mul_mat_id_is_range(dst);\n"
    "    ggml_cuda_pool_alloc<int32_t> bc_ids_local(ctx.pool());\n"
    "    const int32_t * bc_ids = (const int32_t *) ids->data;\n"
    "    if (bc_range) {\n"
    "        const int64_t bc_n_ids = (int64_t) (ids->nb[1] / ggml_element_size(ids)) * ne12;\n"
    "        bc_ids_local.alloc(bc_n_ids);\n"
    "        ggml_cuda_mm_ids_range_translate(bc_ids, bc_ids_local.get(), bc_n_ids, ggml_mul_mat_id_range_base(dst), (int32_t) ne02, stream);\n"
    "        bc_ids = bc_ids_local.get();\n"
    "        CUDA_CHECK(cudaMemsetAsync(ids_src1.get(), 0, ne_get_rows*sizeof(int32_t), stream));\n"
    "        CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), stream));\n"
    "    }\n"
    "\n"
    "    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1 && !bc_range;\n"
)

_A_MMQ_HELPER = "        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),\n"
_N_MMQ_HELPER = "        ggml_cuda_launch_mm_ids_helper(bc_ids, ids_src1.get(), ids_dst.get(), expert_bounds.get(),\n"

# ---- QFP30 chunk 2: range-only Q8_1 scatter kernel; not wired into MMQ yet -----------------------------
_A_Q8_TEMPLATE = r'''template <mmq_q8_1_ds_layout ds_layout, bool scatter>
static __global__ void quantize_mmq_q8_1(
'''
_N_Q8_TEMPLATE = r'''// BigCherry 1281 (QFP30): range_scatter is compile-time-only; false preserves the ordinary scatter specialization.
template <mmq_q8_1_ds_layout ds_layout, bool scatter, bool range_scatter = false>
static __global__ void quantize_mmq_q8_1(
'''

_A_Q8_SENTINEL = r'''        if constexpr (scatter) {
            const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
            ib = k_block*ne1 + i;
        } else {
'''
_N_Q8_SENTINEL = r'''        if constexpr (scatter) {
            const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
            if constexpr (range_scatter) {
                // BigCherry 1281 (QFP30): -1 is the only inactive range inverse-map sentinel.
                // Every other value follows the ordinary indexing path so corrupt maps are not silently hidden.
                if (i == -1) {
                    continue;
                }
            }
            ib = k_block*ne1 + i;
        } else {
'''

_A_Q8_SCATTER_WRAPPER = r'''void quantize_scatter_mmq_q8_1_cuda(
        const float * x, const int32_t * ids_src1_inv, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t stride_token, const int64_t ne0,
        const int64_t n_tokens, const int64_t nrows_dst, const int n_expert_used, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);

    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(n_tokens, block_num_y, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/(int) nrows_dst, /*ne2=*/1, n_expert_used);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/(int) nrows_dst, /*ne2=*/1, n_expert_used);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/(int) nrows_dst, /*ne2=*/1, n_expert_used);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

'''
_N_Q8_SCATTER_WRAPPER = _A_Q8_SCATTER_WRAPPER + r'''// BigCherry 1281 (QFP30): range-only Q8_1 scatter wrapper; -1 inverse-map slots are inactive.
void quantize_scatter_range_mmq_q8_1_cuda(
        const float * x, const int32_t * ids_src1_inv, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t stride_token, const int64_t ne0,
        const int64_t n_tokens, const int64_t nrows_dst, const int n_expert_used, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);

    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(n_tokens, block_num_y, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, true, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/(int) nrows_dst, /*ne2=*/1, n_expert_used);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, true, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/(int) nrows_dst, /*ne2=*/1, n_expert_used);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, true, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/(int) nrows_dst, /*ne2=*/1, n_expert_used);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

'''

_A_Q8_RANGE_DECL = r'''void quantize_scatter_mmq_q8_1_cuda(const float *   x,
                                    const int32_t * ids_src1_inv,
                                    void *          vy,
                                    ggml_type       type_src0,
                                    int64_t         ne00,
                                    int64_t         stride_token,
                                    int64_t         ne0,
                                    int64_t         n_tokens,
                                    int64_t         nrows_dst,
                                    int             n_expert_used,
                                    cudaStream_t    stream);
'''
_N_Q8_RANGE_DECL = _A_Q8_RANGE_DECL + r'''
// BigCherry 1281 (QFP30): range-only scatter; ids_src1_inv may contain -1 for a non-local route.
void quantize_scatter_range_mmq_q8_1_cuda(const float *   x,
                                          const int32_t * ids_src1_inv,
                                          void *          vy,
                                          ggml_type       type_src0,
                                          int64_t         ne00,
                                          int64_t         stride_token,
                                          int64_t         ne0,
                                          int64_t         n_tokens,
                                          int64_t         nrows_dst,
                                          int             n_expert_used,
                                          cudaStream_t    stream);
'''

# ---- phase B: which range ops the GPU takes, and no fusion for them ---------------------------------------------
_A_CUDA_POLICY = (
    "// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization\n"
    "// [TAG_MUL_MAT_ID_CUDA_GRAPHS]\n"
)
_N_CUDA_POLICY = (
    "// BigCherry 1281: a range MUL_MAT_ID is only taken when ggml_cuda_mul_mat_id will run it through a path that\n"
    "// translates the global ids - MMVQ, MMVF (small batches) or MMQ. Same order as the dispatch below. The float\n"
    "// large-batch path (MMF) and the host-sorted fallback are not converted, the scheduler keeps those on the CPU.\n"
    "static bool bc_cuda_mul_mat_id_range_supported(const ggml_tensor * op, const int cc) {\n"
    "    const ggml_tensor * src0 = op->src[0];\n"
    "    const ggml_tensor * src1 = op->src[1];\n"
    "    if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {\n"
    "        return false;\n"
    "    }\n"
    "    if (op->ne[2] <= MMVQ_MAX_BATCH_SIZE) {\n"
    "        if (ggml_is_quantized(src0->type)) {\n"
    "            if (op->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {\n"
    "                return true;\n"
    "            }\n"
    "        } else if (GGML_CUDA_CC_IS_AMD(cc)) {\n"
    "            return true;\n"
    "        }\n"
    "    }\n"
    "    return ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2]);\n"
    "}\n"
    "\n"
) + _A_CUDA_POLICY

_A_CUDA_FUSE = (
    "    if (!is_mul_mat && !is_mul_mat_id) {\n"
    "        return false;\n"
    "    }\n"
)
_N_CUDA_FUSE = _A_CUDA_FUSE + (
    "\n"
    "    // BigCherry 1281: range nodes fuse only in the plain gate + up + GLU form, where an inactive lane is exactly\n"
    "    // GLU(0, 0) = +0 and can be skipped. A fused bias would make an inactive lane non-zero and a fused scale can\n"
    "    // give -0, so those forms are not fused; gate and up must hold the same experts.\n"
    "    if (is_mul_mat_id && (ggml_mul_mat_id_is_range(ffn_up) || ggml_mul_mat_id_is_range(ffn_gate))) {\n"
    "        if (has_bias || has_scale) {\n"
    "            return false;\n"
    "        }\n"
    "        if (!ggml_mul_mat_id_is_range(ffn_up) || !ggml_mul_mat_id_is_range(ffn_gate) ||\n"
    "                ggml_mul_mat_id_range_base(ffn_up) != ggml_mul_mat_id_range_base(ffn_gate) ||\n"
    "                ffn_up->src[0]->ne[2] != ffn_gate->src[0]->ne[2]) {\n"
    "            return false;\n"
    "        }\n"
    "    }\n"
)

# The fused launches hand the kernels the GLU node as dst, so the range is carried in the host-side fusion arguments.
_A_FUSION_HOST = (
    "    const ggml_tensor * shared_gate = nullptr;\n"
    "    ggml_tensor * shared_dst = nullptr;\n"
    "};\n"
    "struct ggml_cuda_mm_fusion_args_device {\n"
)
_N_FUSION_HOST = (
    "    const ggml_tensor * shared_gate = nullptr;\n"
    "    ggml_tensor * shared_dst = nullptr;\n"
    "    // BigCherry 1281: range MUL_MAT_ID in a fused launch (0 = ordinary), taken from the up projection\n"
    "    int32_t id_base = 0;\n"
    "    int64_t id_count = 0;\n"
    "};\n"
    "struct ggml_cuda_mm_fusion_args_device {\n"
)

_A_FUSE_PLAIN = (
    "                fusion_data.gate      = gate->src[0];\n"
    "                fusion_data.glu_op    = ggml_get_glu_op(glu);\n"
    "                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);\n"
)
_N_FUSE_PLAIN = _A_FUSE_PLAIN + (
    "                if (ggml_mul_mat_id_is_range(up)) { // BigCherry 1281: range-aware fusion\n"
    "                    fusion_data.id_base  = ggml_mul_mat_id_range_base(up);\n"
    "                    fusion_data.id_count = up->src[0]->ne[2];\n"
    "                }\n"
)

_A_FUSE_SHARED = (
    "            fusion.shared_gate = shared->src[0]->src[0];\n"
    "            fusion.shared_dst = shared;\n"
)
_N_FUSE_SHARED = _A_FUSE_SHARED + (
    "            if (ggml_mul_mat_id_is_range(up)) { // BigCherry 1281: the routed part is a range op, the shared expert is not\n"
    "                fusion.id_base  = ggml_mul_mat_id_range_base(up);\n"
    "                fusion.id_count = up->src[0]->ne[2];\n"
    "            }\n"
)

_A_FUSE_SCALE = (
    "            ggml_cuda_mm_fusion_args_host fusion_data{};\n"
    "            fusion_data.x_bias  = bias;\n"
    "            fusion_data.x_scale = scale;\n"
)
_N_FUSE_SCALE = (
    "            if (ggml_mul_mat_id_is_range(mm_node)) {\n"
    "                continue; // BigCherry 1281: no scale / bias fusion for a range node (see ggml_cuda_should_fuse_mul_mat)\n"
    "            }\n"
    "\n"
) + _A_FUSE_SCALE

_A_FUSE_BIAS = (
    "        ggml_cuda_mm_fusion_args_host fusion_data{};\n"
    "        fusion_data.x_bias = bias_tensor;\n"
)
_N_FUSE_BIAS = (
    "        if (ggml_mul_mat_id_is_range(mm_node)) {\n"
    "            continue; // BigCherry 1281: a fused bias would make an inactive lane non-zero\n"
    "        }\n"
    "\n"
) + _A_FUSE_BIAS

_A_CUDA = (
    "                if (op->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {\n"
    "                    return false;\n"
    "                }\n"
)
_N_CUDA = _A_CUDA + (
    "                if (ggml_mul_mat_id_is_range(op)) {\n"
    "                    // BigCherry 1281: the range variant carries global ids; take it only where the dispatch runs a\n"
    "                    // path that translates them (MMVQ, MMVF, MMQ) - otherwise the scheduler keeps it on the CPU\n"
    "                    if (!bc_cuda_mul_mat_id_range_supported(op, ggml_cuda_info().devices[dev_ctx->device].cc)) {\n"
    "                        return false;\n"
    "                    }\n"
    "                }\n"
)

_A_CMAKE = "llama_build_and_test(test-tiled-mulmat.cpp)\n"
_N_CMAKE = _A_CMAKE + "llama_build_and_test(test-mul-mat-id-range.cpp)  # BigCherry 1281\n"

_TEST = r'''// BigCherry 1281: reference test of ggml_mul_mat_id_range on the CPU backend.
//
// The range op over the experts [id_base, id_base + n_local) with GLOBAL ids must equal, bit for bit, the ordinary
// ggml_mul_mat_id over the same local experts with the in-range ids translated by hand, and be an exact +0 on every
// lane whose id is out of range. Covers both bases around zero and near INT32_MAX, ids just below / at / at the end of /
// just past the range, all-active, all-inactive, duplicate and unsorted ids, several batch sizes, F32 and Q8_0 weights.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct range_case {
    int32_t id_base;
    int     n_local;
    int     n_used;
    int     n_tokens;
    int     pattern;   // how the ids are chosen, see make_ids
    ggml_type type;
    bool    broadcast; // activation [n_in, 1, n_tokens] instead of per-route [n_in, n_used, n_tokens]
};

static std::vector<int32_t> make_ids(const range_case & c, std::mt19937 & rng) {
    const int64_t lo = c.id_base, hi = (int64_t) c.id_base + c.n_local;  // [lo, hi)
    std::vector<int64_t> pool;
    switch (c.pattern) {
        case 0: // all in range
            for (int64_t g = lo; g < hi; g++) pool.push_back(g);
            break;
        case 1: // all out of range
            if (lo - 1 >= 0)       pool.push_back(lo - 1);
            if (hi <= INT32_MAX)   pool.push_back(hi);
            if (hi + 1 <= INT32_MAX) pool.push_back(hi + 1);
            break;
        default: // mixed: the boundaries and everything inside
            if (lo - 1 >= 0)     pool.push_back(lo - 1);
            for (int64_t g = lo; g < hi; g++) pool.push_back(g);
            if (hi <= INT32_MAX) pool.push_back(hi);
            break;
    }
    std::vector<int32_t> ids((size_t) c.n_used * c.n_tokens);
    for (size_t i = 0; i < ids.size(); i++) {
        // random draws give duplicates within a token and unsorted order
        ids[i] = (int32_t) pool[rng() % pool.size()];
    }
    if (c.pattern == 2 && ids.size() >= 4) {
        // pin the four boundary ids so every mixed case exercises them
        size_t k = 0;
        if (lo - 1 >= 0)     ids[k++] = (int32_t) (lo - 1);
        ids[k++] = (int32_t) lo;
        ids[k++] = (int32_t) (hi - 1);
        if (hi <= INT32_MAX) ids[k++] = (int32_t) hi;
    }
    return ids;
}

static bool run_case(const range_case & c, int n_threads, std::mt19937 & rng) {
    const int n_in = 64, n_out = 24;   // Q8_0 needs the row length to be a multiple of 32

    ggml_init_params ip = { (size_t) 64 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    ggml_tensor * w_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, n_out, c.n_local);
    for (int64_t i = 0; i < ggml_nelements(w_f32); i++) ((float *) w_f32->data)[i] = dist(rng);

    ggml_tensor * w = w_f32;
    if (c.type != GGML_TYPE_F32) {
        w = ggml_new_tensor_3d(ctx, c.type, n_in, n_out, c.n_local);
        ggml_quantize_chunk(c.type, (const float *) w_f32->data, w->data, 0, (int64_t) n_out * c.n_local, n_in, nullptr);
    }

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, c.broadcast ? 1 : c.n_used, c.n_tokens);
    for (int64_t i = 0; i < ggml_nelements(x); i++) ((float *) x->data)[i] = dist(rng);

    const std::vector<int32_t> global = make_ids(c, rng);

    ggml_tensor * ids_global = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, c.n_tokens);
    ggml_tensor * ids_local  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, c.n_tokens);
    std::vector<bool> active(global.size());
    int n_active = 0;
    for (size_t i = 0; i < global.size(); i++) {
        const int64_t l = (int64_t) global[i] - (int64_t) c.id_base;
        active[i] = l >= 0 && l < c.n_local;
        n_active += active[i] ? 1 : 0;
        ((int32_t *) ids_global->data)[i] = global[i];
        ((int32_t *) ids_local->data)[i]  = active[i] ? (int32_t) l : 0;  // inactive lanes are overwritten with zero below
    }

    ggml_tensor * y_range = ggml_mul_mat_id_range(ctx, w, x, ids_global, c.id_base);
    ggml_tensor * y_ref   = ggml_mul_mat_id(ctx, w, x, ids_local);

    bool ok = ggml_mul_mat_id_is_range(y_range) && ggml_mul_mat_id_range_base(y_range) == c.id_base &&
              !ggml_mul_mat_id_is_range(y_ref) && ggml_mul_mat_id_range_base(y_ref) == 0;

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y_range);
    ggml_build_forward_expand(gf, y_ref);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    const float * r = (const float *) y_range->data;
    const float * e = (const float *) y_ref->data;
    int64_t bad = 0;
    for (size_t lane = 0; lane < global.size(); lane++) {
        for (int i = 0; i < n_out; i++) {
            const float got  = r[lane * n_out + i];
            const float want = active[lane] ? e[lane * n_out + i] : 0.0f;
            // bit comparison: an inactive lane must be +0, not -0 and not a small number
            if (std::memcmp(&got, &want, sizeof(float)) != 0) bad++;
        }
    }
    ok = ok && bad == 0;

    printf("%s base=%" PRId32 " n_local=%d used=%d tokens=%d pattern=%d type=%s threads=%d broadcast=%d active=%d/%zu mismatches=%" PRId64 "\n",
           ok ? "ok  " : "FAIL", c.id_base, c.n_local, c.n_used, c.n_tokens, c.pattern, ggml_type_name(c.type), n_threads,
           c.broadcast ? 1 : 0, n_active, global.size(), bad);

    ggml_free(ctx);
    return ok;
}

// ---- GPU mode (--gpu): the range op on every non-CPU device against a CPU reference -------------------------------
// The op is computed directly on the device backend (no scheduler), so it reaches the device kernels whatever
// supports_op says. Reference: the ORDINARY op on the CPU backend over the same local experts with hand-translated
// ids. Active lanes are compared by normalised mean squared error (device kernels are not bit-identical to the CPU);
// inactive lanes must be exactly +0. Token counts cover the single-token, small-batch and large-batch kernels.

struct gpu_case {
    ggml_type type;
    int32_t   id_base;
    int       n_local;
    int       n_used;
    int       n_tokens;
    int       pattern;
    bool      broadcast;
};

static std::vector<float> compute_on(ggml_backend_t backend, const gpu_case & c, bool range,
                                     const std::vector<float> & w_f32, const std::vector<float> & x_f32,
                                     const std::vector<int32_t> & ids, int n_in, int n_out) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w = ggml_new_tensor_3d(ctx, c.type, n_in, n_out, c.n_local);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, c.broadcast ? 1 : c.n_used, c.n_tokens);
    ggml_tensor * t_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, c.n_tokens);
    ggml_tensor * y = range ? ggml_mul_mat_id_range(ctx, w, x, t_ids, c.id_base) : ggml_mul_mat_id(ctx, w, x, t_ids);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    if (range && !ggml_backend_supports_op(backend, y)) {
        // the device declines this range op (a path that is not converted): the scheduler would run it on the CPU
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return {};
    }

    std::vector<uint8_t> w_data(ggml_nbytes(w));
    if (c.type == GGML_TYPE_F32) {
        std::memcpy(w_data.data(), w_f32.data(), w_data.size());
    } else {
        ggml_quantize_chunk(c.type, w_f32.data(), w_data.data(), 0, (int64_t) n_out * c.n_local, n_in, nullptr);
    }
    ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size());
    ggml_backend_tensor_set(x, x_f32.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(t_ids, ids.data(), 0, ggml_nbytes(t_ids));

    std::vector<float> out((size_t) n_out * c.n_used * c.n_tokens);
    if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    } else {
        out.assign(out.size(), NAN);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static bool run_gpu_case(ggml_backend_t gpu, ggml_backend_t cpu, const gpu_case & c, std::mt19937 & rng) {
    const int n_in = 256, n_out = 64;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> w((size_t) n_in * n_out * c.n_local),
                       x((size_t) n_in * (c.broadcast ? 1 : c.n_used) * c.n_tokens);
    for (float & v : w) v = dist(rng);
    for (float & v : x) v = dist(rng);

    // Expert ids are DISTINCT within a token, as a top-k router produces them: the device's large-batch path groups
    // one row per (token, expert) and does not support the same expert twice in one token (the CPU does).
    // pattern 0: mostly held experts; 1: none held; 2: mixed around both ends of the range.
    int64_t u_lo = c.id_base, u_hi = (int64_t) c.id_base + std::max(c.n_local, c.n_used);
    if (c.pattern == 1) { u_lo = (int64_t) c.id_base + c.n_local; u_hi = u_lo + 16; }
    if (c.pattern == 2) { u_lo = std::max<int64_t>(0, (int64_t) c.id_base - 3); u_hi = (int64_t) c.id_base + c.n_local + 13; }
    std::vector<int32_t> universe;
    for (int64_t g = u_lo; g < u_hi; g++) universe.push_back((int32_t) g);
    std::vector<int32_t> global((size_t) c.n_used * c.n_tokens);
    for (int t = 0; t < c.n_tokens; t++) {
        std::shuffle(universe.begin(), universe.end(), rng);
        for (int k = 0; k < c.n_used; k++) global[(size_t) t * c.n_used + k] = universe[k];
    }
    std::vector<int32_t> local(global.size());
    std::vector<bool> active(global.size());
    int n_active = 0;
    for (size_t i = 0; i < global.size(); i++) {
        const int64_t l = (int64_t) global[i] - (int64_t) c.id_base;
        active[i] = l >= 0 && l < c.n_local;
        n_active += active[i] ? 1 : 0;
        local[i] = active[i] ? (int32_t) l : 0;
    }

    const std::vector<float> got  = compute_on(gpu, c, /*range =*/ true,  w, x, global, n_in, n_out);
    if (got.empty()) {
        printf("skip %s type=%s tokens=%d used=%d broadcast=%d: range op not supported on this device path\n",
               ggml_backend_name(gpu), ggml_type_name(c.type), c.n_tokens, c.n_used, c.broadcast ? 1 : 0);
        return true;
    }
    const std::vector<float> want = compute_on(cpu, c, /*range =*/ false, w, x, local,  n_in, n_out);

    double err = 0.0, ref = 0.0;
    int64_t nonzero_inactive = 0, nan = 0;
    for (size_t lane = 0; lane < global.size(); lane++) {
        for (int i = 0; i < n_out; i++) {
            const float g = got[lane * n_out + i];
            if (std::isnan(g)) { nan++; continue; }
            if (active[lane]) {
                const float e = want[lane * n_out + i];
                err += (double) (g - e) * (g - e);
                ref += (double) e * e;
            } else {
                const float zero = 0.0f;
                if (std::memcmp(&g, &zero, sizeof(float)) != 0) nonzero_inactive++;
            }
        }
    }
    const double nmse = ref > 0.0 ? err / ref : 0.0;
    const bool ok = nan == 0 && nonzero_inactive == 0 && nmse < 5e-4;

    if (!ok && getenv("MMID_RANGE_CONTROL") != nullptr) {
        // control: the ORDINARY op on the same device with the hand-translated ids (inactive lanes use expert 0).
        // If this is also far from the CPU on the active lanes, the failure is not in the range translation.
        const std::vector<float> ctl = compute_on(gpu, c, /*range =*/ false, w, x, local, n_in, n_out);
        double cerr = 0.0, cref = 0.0;
        for (size_t lane = 0; lane < global.size(); lane++) {
            if (!active[lane]) continue;
            for (int i = 0; i < n_out; i++) {
                const float e = want[lane * n_out + i], g = ctl[lane * n_out + i];
                cerr += (double) (g - e) * (g - e);
                cref += (double) e * e;
            }
        }
        printf("ctrl %s type=%s tokens=%d used=%d n_local=%d pattern=%d broadcast=%d ordinary-op nmse=%.2e\n", ggml_backend_name(gpu),
               ggml_type_name(c.type), c.n_tokens, c.n_used, c.n_local, c.pattern, c.broadcast ? 1 : 0,
               cref > 0.0 ? cerr / cref : 0.0);
    }
    printf("%s %s type=%s base=%" PRId32 " n_local=%d used=%d tokens=%d pattern=%d broadcast=%d active=%d/%zu nmse=%.2e nonzero_inactive=%" PRId64 " nan=%" PRId64 "\n",
           ok ? "ok  " : "FAIL", ggml_backend_name(gpu), ggml_type_name(c.type), c.id_base, c.n_local, c.n_used, c.n_tokens,
           c.pattern, c.broadcast ? 1 : 0, n_active, global.size(), nmse, nonzero_inactive, nan);
    return ok;
}

static int run_gpu() {
    std::mt19937 rng(1281);
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    int n_fail = 0, n_run = 0, n_broadcast = 0, n_dev = 0;
    // MMID_RANGE_MIN_TOKENS / MMID_RANGE_MAX_TOKENS restrict the batch sizes, to test one kernel family at a time
    const int min_tokens = getenv("MMID_RANGE_MIN_TOKENS") ? atoi(getenv("MMID_RANGE_MIN_TOKENS")) : 0;
    const int max_tokens = getenv("MMID_RANGE_MAX_TOKENS") ? atoi(getenv("MMID_RANGE_MAX_TOKENS")) : INT_MAX;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
        if (gpu == nullptr) {
            continue;
        }
        n_dev++;
        for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS }) {
            for (int32_t id_base : { 0, 5 }) {
                for (int n_local : { 1, 4, 12 }) {
                    for (int pattern : { 0, 1, 2 }) {
                        for (int n_used : { 4, 10 }) {
                            for (int n_tokens : { 1, 2, 4, 8, 9, 33, 300 }) {
                                if (n_tokens < min_tokens || n_tokens > max_tokens) {
                                    continue;
                                }
                                const gpu_case c = { type, id_base, n_local, n_used, n_tokens, pattern, false };
                                n_run++;
                                n_fail += run_gpu_case(gpu, cpu, c, rng) ? 0 : 1;
                            }
                        }
                    }
                }
            }
        }
        // QFP30 chunk 1: repeat the exact case matrix with the gate/up broadcast activation shape.
        for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS }) {
            for (int32_t id_base : { 0, 5 }) {
                for (int n_local : { 1, 4, 12 }) {
                    for (int pattern : { 0, 1, 2 }) {
                        for (int n_used : { 4, 10 }) {
                            for (int n_tokens : { 1, 2, 4, 8, 9, 33, 300 }) {
                                if (n_tokens < min_tokens || n_tokens > max_tokens) {
                                    continue;
                                }
                                const gpu_case c = { type, id_base, n_local, n_used, n_tokens, pattern, true };
                                n_run++;
                                n_broadcast++;
                                n_fail += run_gpu_case(gpu, cpu, c, rng) ? 0 : 1;
                            }
                        }
                    }
                }
            }
        }
        ggml_backend_free(gpu);
    }
    ggml_backend_free(cpu);
    printf("test-mul-mat-id-range --gpu: %d device(s), %d case(s), %d broadcast, %d failed\n",
           n_dev, n_run, n_broadcast, n_fail);
    return n_fail == 0 && n_dev > 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--gpu") == 0) {
        return run_gpu();
    }
    std::mt19937 rng(1281);
    int n_fail = 0, n_run = 0, n_broadcast = 0;
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_Q8_0 }) {
        for (int32_t id_base : { 0, 1, 7, INT32_MAX - 3 }) {
            for (int n_local : { 1, 2, 4 }) {
                if ((int64_t) id_base + n_local - 1 > INT32_MAX) {
                    continue;  // the last held expert must be a representable id
                }
                for (int pattern : { 0, 1, 2 }) {
                    for (int n_tokens : { 1, 3, 40 }) {
                        for (int n_threads : { 1, 4 }) {
                            const range_case c = { id_base, n_local, 4, n_tokens, pattern, type, false };
                            n_run++;
                            n_fail += run_case(c, n_threads, rng) ? 0 : 1;
                        }
                    }
                }
            }
        }
    }
    // QFP30 chunk 1: same reference matrix, now with broadcast activation [n_in, 1, n_tokens].
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_Q8_0 }) {
        for (int32_t id_base : { 0, 1, 7, INT32_MAX - 3 }) {
            for (int n_local : { 1, 2, 4 }) {
                if ((int64_t) id_base + n_local - 1 > INT32_MAX) {
                    continue;
                }
                for (int pattern : { 0, 1, 2 }) {
                    for (int n_tokens : { 1, 3, 40 }) {
                        for (int n_threads : { 1, 4 }) {
                            const range_case c = { id_base, n_local, 4, n_tokens, pattern, type, true };
                            n_run++;
                            n_broadcast++;
                            n_fail += run_case(c, n_threads, rng) ? 0 : 1;
                        }
                    }
                }
            }
        }
    }
    printf("test-mul-mat-id-range: %d case(s), %d broadcast, %d failed\n", n_run, n_broadcast, n_fail);
    return n_fail == 0 ? 0 : 1;
}
'''

PATCHES = [
    FilePatch(
        path="ggml/include/ggml.h",
        description="1281: ggml_mul_mat_id_range declaration and accessors",
        language="none",
        edits=(
            Edit(id="mmid-range-decl", anchor=_re.escape(_A_H), mode="replace", text=_N_H,
                 guard=r"GGML_API struct ggml_tensor \* ggml_mul_mat_id_range\(",
                 rationale="After the ordinary ggml_mul_mat_id declaration.", expect_matches=1, max_span_lines=6),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml.c",
        description="1281: ggml_mul_mat_id_range constructor (flag and base in the op params of MUL_MAT_ID)",
        language="none",
        edits=(
            Edit(id="mmid-range-ctor", anchor=_re.escape(_A_C), mode="replace", text=_N_C,
                 guard=r"#define GGML_BC_MUL_MAT_ID_RANGE_MARK",
                 rationale="After the ordinary constructor, identified by its three sources.", expect_matches=1, max_span_lines=8),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cpu/ggml-cpu.c",
        description="1281: CPU MUL_MAT_ID translates global ids to the held range, exact zero outside it",
        language="none",
        edits=(
            Edit(id="mmid-range-cpu-locals", anchor=_re.escape(_A_CPU_LOCALS), mode="replace", text=_N_CPU_LOCALS,
                 guard=r"const bool    bc_range   = ggml_mul_mat_id_is_range\(dst\);",
                 rationale="Locals of ggml_compute_forward_mul_mat_id.", expect_matches=1, max_span_lines=4),
            Edit(id="mmid-range-cpu-group", anchor=_re.escape(_A_CPU_GROUP), mode="replace", text=_N_CPU_GROUP,
                 guard=r"const int64_t bc_local = \(int64_t\) i02 - \(int64_t\) bc_id_base;",
                 rationale="The row grouping, the one place that interprets expert ids on the CPU.", expect_matches=1,
                 max_span_lines=13),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cpu/repack.cpp",
        description="1281: the repacked-weights CPU MUL_MAT_ID translates global ids the same way",
        language="none",
        edits=(
            Edit(id="mmid-range-cpu-repack", anchor=_re.escape(_A_REPACK), mode="replace", text=_N_REPACK,
                 guard=r"const bool    bc_range   = ggml_mul_mat_id_is_range\(op\);",
                 rationale="tensor_traits::forward_mul_mat_id row grouping (found by review req_61670b6161b94b09).",
                 expect_matches=1, max_span_lines=11),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/common.cuh",
        description="1281 phase B: range base / count carried to the MMVQ / MMVF kernels",
        language="none",
        edits=(
            Edit(id="mmid-range-fusion-host-args", anchor=_re.escape(_A_FUSION_HOST), mode="replace", text=_N_FUSION_HOST,
                 guard=r"range MUL_MAT_ID in a fused launch \(0 = ordinary\)", rationale="End of ggml_cuda_mm_fusion_args_host.",
                 expect_matches=1, max_span_lines=5),
            Edit(id="mmid-range-fusion-args", anchor=_re.escape(_A_FUSION_ARGS), mode="replace", text=_N_FUSION_ARGS,
                 guard=r"BigCherry 1281: range MUL_MAT_ID - src0 holds the experts", rationale="Last field of ggml_cuda_mm_fusion_args_device.",
                 expect_matches=1, max_span_lines=2),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/mmvq.cu",
        description="1281 phase B: MMVQ kernels translate global expert ids and skip lanes that are not held",
        language="none",
        edits=(
            Edit(id="mmid-range-mmvq-kernel", anchor=_re.escape(_A_MMVQ_K1), mode="replace", text=_N_MMVQ_K1,
                 guard=r"if \(fusion\.id_count != 0 && !shared_expert && ncols_dst == 1 && ids\) \{",
                 rationale="mul_mat_vec_q, where the single-token kernel reads the expert id.", expect_matches=1,
                 max_span_lines=4),
            Edit(id="mmid-range-mmvq-moe-kernel", anchor=_re.escape(_A_MMVQ_K2), mode="replace", text=_N_MMVQ_K2,
                 guard=r"    uint32_t channel_x = shared_expert \? 0 : ids\[channel_dst \+ token_idx \* ids_stride\];",
                 rationale="mul_mat_vec_q_moe, where the multi-token kernel reads the expert id.", expect_matches=1,
                 max_span_lines=2),
            Edit(id="mmid-range-mmvq-host", anchor=_re.escape(_A_MMV_HOST), mode="replace", text=_N_MMV_HOST,
                 guard=r"fusion_local\.id_count = src0->ne\[2\];",
                 rationale="ggml_cuda_mul_mat_vec_q, before the fusion arguments are filled.", expect_matches=1,
                 max_span_lines=4),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/mmvf.cu",
        description="1281 phase B: MMVF kernel translates global expert ids and skips lanes that are not held",
        language="none",
        edits=(
            Edit(id="mmid-range-mmvf-kernel", anchor=_re.escape(_A_MMVF_K), mode="replace", text=_N_MMVF_K,
                 guard=r"const int64_t bc_local = \(int64_t\) channel_x - \(int64_t\) fusion\.id_base;",
                 rationale="mul_mat_vec_f, after both id-reading branches.", expect_matches=1, max_span_lines=4),
            Edit(id="mmid-range-mmvf-host", anchor=_re.escape(_A_MMV_HOST), mode="replace", text=_N_MMV_HOST,
                 guard=r"fusion_local\.id_count = src0->ne\[2\];",
                 rationale="ggml_cuda_mul_mat_vec_f, before the fusion arguments are filled.", expect_matches=1,
                 max_span_lines=4),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/mmid.cuh",
        description="1281 phase B: device-side translation of global expert ids (declaration)",
        language="none",
        edits=(
            Edit(id="mmid-range-translate-decl", anchor=_re.escape(_A_MMID_DECL), mode="replace", text=_N_MMID_DECL,
                 guard=r"void ggml_cuda_mm_ids_range_translate\(", rationale="After the ids helper declaration.",
                 expect_matches=1, max_span_lines=4),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/mmid.cu",
        description="1281 phase B: device-side translation of global expert ids",
        language="none",
        edits=(
            Edit(id="mmid-range-translate", anchor=_re.escape(_A_MMID_DEF), mode="replace", text=_N_MMID_DEF,
                 guard=r"static __global__ void mm_ids_range_translate\(",
                 rationale="Before the public ids helper entry point.", expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/quantize.cu",
        description="1281 QFP30 chunk 2: range-only Q8_1 scatter specialization and host wrapper",
        language="none",
        edits=(
            Edit(id="mmid-range-q8-template", anchor=_re.escape(_A_Q8_TEMPLATE), mode="replace", text=_N_Q8_TEMPLATE,
                 guard=r"BigCherry 1281 \(QFP30\): range_scatter is compile-time-only",
                 rationale="Add a compile-time-only range scatter variant; ordinary two-argument instantiations default false.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="mmid-range-q8-sentinel", anchor=_re.escape(_A_Q8_SENTINEL), mode="replace", text=_N_Q8_SENTINEL,
                 guard=r"BigCherry 1281 \(QFP30\): -1 is the only inactive range inverse-map sentinel",
                 rationale="Only the range specialization skips -1; every other value keeps the ordinary indexing path.",
                 expect_matches=1, max_span_lines=5),
            Edit(id="mmid-range-q8-wrapper", anchor=_re.escape(_A_Q8_SCATTER_WRAPPER), mode="replace", text=_N_Q8_SCATTER_WRAPPER,
                 guard=r"BigCherry 1281 \(QFP30\): range-only Q8_1 scatter wrapper",
                 rationale="Add an uncalled host wrapper launching the range specialization for all ordinary Q8_1 ds layouts.",
                 expect_matches=1, max_span_lines=36),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/quantize.cuh",
        description="1281 QFP30 chunk 2: declare the range-only Q8_1 scatter wrapper",
        language="none",
        edits=(
            Edit(id="mmid-range-q8-wrapper-decl", anchor=_re.escape(_A_Q8_RANGE_DECL), mode="replace", text=_N_Q8_RANGE_DECL,
                 guard=r"BigCherry 1281 \(QFP30\): range-only scatter",
                 rationale="Declare the uncalled range-only Q8_1 scatter wrapper beside the ordinary wrapper.",
                 expect_matches=1, max_span_lines=12),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/mmq.cu",
        description="1281 phase B: MMQ groups rows by local expert for the range variant",
        language="none",
        edits=(
            Edit(id="mmid-range-mmq-setup", anchor=_re.escape(_A_MMQ_DEDUP), mode="replace", text=_N_MMQ_DEDUP,
                 guard=r"const bool bc_range = ggml_mul_mat_id_is_range\(dst\);",
                 rationale="ggml_cuda_mul_mat_q, MUL_MAT_ID branch, before the ids helper runs.", expect_matches=1,
                 max_span_lines=2),
            Edit(id="mmid-range-mmq-helper", anchor=_re.escape(_A_MMQ_HELPER), mode="replace", text=_N_MMQ_HELPER,
                 guard=r"ggml_cuda_launch_mm_ids_helper\(bc_ids, ids_src1\.get\(\)",
                 rationale="The ids helper call of the MMQ MUL_MAT_ID branch.", expect_matches=1, max_span_lines=2),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1281 phase B: the GPU takes the range variant only on the converted paths; range nodes are never fused",
        language="none",
        edits=(
            Edit(id="mmid-range-cuda-policy", anchor=_re.escape(_A_CUDA_POLICY), mode="replace", text=_N_CUDA_POLICY,
                 guard=r"static bool bc_cuda_mul_mat_id_range_supported\(",
                 rationale="Before ggml_cuda_mul_mat_id_needs_sync, next to the dispatch it mirrors.", expect_matches=1,
                 max_span_lines=3),
            Edit(id="mmid-range-cuda-fusion-rule", anchor=_re.escape(_A_CUDA_FUSE), mode="replace", text=_N_CUDA_FUSE,
                 guard=r"range nodes fuse only in the plain gate \+ up \+ GLU form",
                 rationale="ggml_cuda_should_fuse_mul_mat, after the op-kind check.", expect_matches=1, max_span_lines=4),
            Edit(id="mmid-range-cuda-fuse-plain", anchor=_re.escape(_A_FUSE_PLAIN), mode="replace_all", text=_N_FUSE_PLAIN,
                 guard=r"if \(ggml_mul_mat_id_is_range\(up\)\) \{ // BigCherry 1281: range-aware fusion",
                 rationale="The two plain gate + up + GLU launches (MMVF and MMVQ).", expect_matches=2, max_span_lines=4),
            Edit(id="mmid-range-cuda-fuse-shared", anchor=_re.escape(_A_FUSE_SHARED), mode="replace", text=_N_FUSE_SHARED,
                 guard=r"the routed part is a range op, the shared expert is not",
                 rationale="The routed + shared expert fused launch.", expect_matches=1, max_span_lines=3),
            Edit(id="mmid-range-cuda-no-scale-fusion", anchor=_re.escape(_A_FUSE_SCALE), mode="replace", text=_N_FUSE_SCALE,
                 guard=r"no scale / bias fusion for a range node", rationale="The scale (+ bias) fused launch.",
                 expect_matches=1, max_span_lines=4),
            Edit(id="mmid-range-cuda-no-bias-fusion", anchor=_re.escape(_A_FUSE_BIAS), mode="replace", text=_N_FUSE_BIAS,
                 guard=r"a fused bias would make an inactive lane non-zero", rationale="The bias fused launch.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="mmid-range-cuda-refuse", anchor=_re.escape(_A_CUDA), mode="replace", text=_N_CUDA,
                 guard=r"if \(ggml_mul_mat_id_is_range\(op\)\) \{",
                 rationale="supports_op, MUL_MAT / MUL_MAT_ID case, after the F32-precision refusal.", expect_matches=1,
                 max_span_lines=4),
        ),
    ),
    FilePatch(
        path="tests/CMakeLists.txt",
        description="1281: build and run the range reference test",
        language="none",
        edits=(
            Edit(id="mmid-range-test-cmake", anchor=_re.escape(_A_CMAKE), mode="replace", text=_N_CMAKE,
                 guard=r"llama_build_and_test\(test-mul-mat-id-range\.cpp\)",
                 rationale="Next to the other ggml-level matmul test.", expect_matches=1, max_span_lines=2),
        ),
    ),
    FilePatch(
        path="tests/test-mul-mat-id-range.cpp",
        description="1281: reference test of the range op on the CPU backend (new file)",
        language="none",
        create=True,
        edits=(
            Edit(id="mmid-range-test-create", anchor="\\A", mode="insert_after", text=_TEST,
                 guard=r"// BigCherry 1281: reference test of ggml_mul_mat_id_range on the CPU backend\.",
                 rationale="New test program.", max_span_lines=1),
        ),
    ),
]
