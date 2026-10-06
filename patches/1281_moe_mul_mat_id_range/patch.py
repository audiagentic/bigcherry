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
STATE = "untested"

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
    "    if (ids != nullptr && fusion == nullptr && ggml_mul_mat_id_is_range(dst)) {\n"
    "        // BigCherry 1281: range variant. The kernel skips the lanes this device does not hold, so clear dst first\n"
    "        // (stream ordered, no synchronisation): those lanes are then an exact +0.\n"
    "        fusion_local.id_base  = ggml_mul_mat_id_range_base(dst);\n"
    "        fusion_local.id_count = src0->ne[2];\n"
    "        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), ctx.stream()));\n"
    "    }\n"
    "\n"
    "    if (fusion) {\n"
)

_A_CUDA = (
    "                if (op->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {\n"
    "                    return false;\n"
    "                }\n"
)
_N_CUDA = _A_CUDA + (
    "                if (ggml_mul_mat_id_is_range(op)) {\n"
    "                    // BigCherry 1281 phase A: the range variant carries global ids, which only the CPU\n"
    "                    // implementation translates; refuse it so no global id reaches a GPU kernel\n"
    "                    return false;\n"
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
#include "ggml-cpu.h"

#include <cinttypes>
#include <climits>
#include <cstdio>
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

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, c.n_used, c.n_tokens);
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

    printf("%s base=%" PRId32 " n_local=%d used=%d tokens=%d pattern=%d type=%s threads=%d active=%d/%zu mismatches=%" PRId64 "\n",
           ok ? "ok  " : "FAIL", c.id_base, c.n_local, c.n_used, c.n_tokens, c.pattern, ggml_type_name(c.type), n_threads,
           n_active, global.size(), bad);

    ggml_free(ctx);
    return ok;
}

int main() {
    std::mt19937 rng(1281);
    int n_fail = 0, n_run = 0;
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_Q8_0 }) {
        for (int32_t id_base : { 0, 1, 7, INT32_MAX - 3 }) {
            for (int n_local : { 1, 2, 4 }) {
                if ((int64_t) id_base + n_local - 1 > INT32_MAX) {
                    continue;  // the last held expert must be a representable id
                }
                for (int pattern : { 0, 1, 2 }) {
                    for (int n_tokens : { 1, 3, 40 }) {
                        for (int n_threads : { 1, 4 }) {
                            const range_case c = { id_base, n_local, 4, n_tokens, pattern, type };
                            n_run++;
                            n_fail += run_case(c, n_threads, rng) ? 0 : 1;
                        }
                    }
                }
            }
        }
    }
    printf("test-mul-mat-id-range: %d case(s), %d failed\n", n_run, n_fail);
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
            Edit(id="mmid-range-fusion-args", anchor=_re.escape(_A_FUSION_ARGS), mode="replace", text=_N_FUSION_ARGS,
                 guard=r"int64_t id_count = 0;", rationale="Last field of ggml_cuda_mm_fusion_args_device.",
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
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1281 phase A: the HIP/CUDA backend refuses the range variant (CPU reference only)",
        language="none",
        edits=(
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
