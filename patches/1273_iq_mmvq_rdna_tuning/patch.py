"""1273: opt-in RDNA IQ MMVQ launch tuning for IQ4_XS and IQ3_XXS.

Unset env vars preserve b11233 exactly.  Decode-only variants are selected at
ncols_dst == 1 on gfx1100/gfx1201.  BIGCHERRY_IQ_MMVQ_VDR and
BIGCHERRY_IQ_MMVQ_NWARPS independently enable the experimental VDR and block
width choices.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-iq-mmvq"
STATE = "untested"

# --------------------------------------------------------------- vecdotq.cuh

_IQ3_NEXT_OLD = """#define VDR_IQ3_S_Q8_1_MMVQ 2
#define VDR_IQ3_S_Q8_1_MMQ  2
"""

_IQ3_VDR1 = r'''// BigCherry 1273: VDR=1 IQ3_XXS MMVQ variant for RDNA decode A/B.
// One call consumes half of the pristine VDR=2 pair; aux32/sign/scale state is
// shared by both halves, while q8_base selects the corresponding four q8 bytes.
static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1_vdr1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int q3_packed = get_int_b2(bq3->qs, iqs);
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);
    const int q8_base = 4 * (iqs & 1);
    const int sign_base = 14 * (iqs & 1);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 4; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (sign_base + 7*l0/2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, q8_base + l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, q8_base + l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

'''

_IQ4_FUNC_HEAD = """static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
"""

_IQ4_VDR2 = r'''// BigCherry 1273: VDR=2 IQ4_XS MMVQ variant for RDNA decode A/B.
// iqs may now address either half of a pristine VDR=4 group; q8_base keeps the
// two q4 words aligned with the matching q8 lanes.
static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1_vdr2(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq + kbx;
    const int q8_base = iqs & 0x02;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

        const int u0 = get_int_b4(bq8_1[iqs/4].qs, q8_base + j + 0);
        const int u1 = get_int_b4(bq8_1[iqs/4].qs, q8_base + j + 4);

        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }

    const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
    sumi *= ls - 32;

    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs/4].ds);
    return d * sumi;
}

'''

# ------------------------------------------------------------------- mmvq.cu

_INCLUDES_OLD = """#include <cstdint>
#include <type_traits>
"""

_INCLUDES_NEW = """#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
"""

_VDR_FN_HEAD = """static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
"""

_VEC_SELECTOR = r'''template <ggml_type type, int vdr>
static constexpr __device__ vec_dot_q_cuda_t bigcherry_iq_vec_dot_q_cuda() {
    if constexpr (type == GGML_TYPE_IQ3_XXS && vdr == 1) {
        return vec_dot_iq3_xxs_q8_1_vdr1;
    }
    if constexpr (type == GGML_TYPE_IQ4_XS && vdr == 2) {
        return vec_dot_iq4_xs_q8_1_vdr2;
    }
    return get_vec_dot_q_cuda(type);
}

'''

_CALC_NWARPS_HEAD = """static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
"""

_TUNING_SUPPORT = r'''static bool bigcherry_iq_mmvq_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

static const char * bigcherry_iq_mmvq_arch_name(mmvq_parameter_table_id table_id) {
    switch (table_id) {
        case MMVQ_PARAMETERS_RDNA3_0: return "gfx1100";
        case MMVQ_PARAMETERS_RDNA4:   return "gfx1201";
        default:                      return "other";
    }
}

static const char * bigcherry_iq_mmvq_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ4_XS:  return "iq4_xs";
        case GGML_TYPE_IQ3_XXS: return "iq3_xxs";
        default:                return "other";
    }
}

static const char * bigcherry_iq_mmvq_variant_name(bool tune_vdr, bool tune_nwarps) {
    if (tune_vdr && tune_nwarps) {
        return "vdr_nwarps";
    }
    return tune_vdr ? "vdr" : "nwarps";
}

static void bigcherry_iq_mmvq_trace(
        ggml_type type, mmvq_parameter_table_id table_id, bool tune_vdr, bool tune_nwarps) {
    if (getenv("BIGCHERRY_PATCH_TRACE") == nullptr) {
        return;
    }
    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set(std::memory_order_relaxed)) {
        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1273_iq_mmvq path=%s type=%s arch=%s\n",
            bigcherry_iq_mmvq_variant_name(tune_vdr, tune_nwarps),
            bigcherry_iq_mmvq_type_name(type),
            bigcherry_iq_mmvq_arch_name(table_id));
    }
}

'''

_KERNEL_HEAD_OLD = """template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
"""

_KERNEL_HEAD_NEW = """template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false,
          int iq_vdr = 0, int iq_nwarps = 0>
__launch_bounds__((iq_nwarps > 0 ? iq_nwarps : calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters))*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
"""

_KERNEL_CONSTS_OLD = """    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
"""

_KERNEL_CONSTS_NEW = """    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = iq_vdr > 0 ? iq_vdr : get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = iq_nwarps > 0 ? iq_nwarps : calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = bigcherry_iq_vec_dot_q_cuda<type, vdr>();
"""

_SWITCH_FUSION_HEAD_OLD = """template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(
"""

_SWITCH_FUSION_HEAD_NEW = """template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false,
         int iq_vdr = 0, int iq_nwarps = 0>
static void mul_mat_vec_q_switch_fusion(
"""

_FUSED_LAUNCH_OLD = """            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
"""
_FUSED_LAUNCH_NEW = """            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, iq_vdr, iq_nwarps>, launch_params,
"""

_PLAIN_LAUNCH_OLD = """    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters>, launch_params,
"""
_PLAIN_LAUNCH_NEW = """    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, iq_vdr, iq_nwarps>, launch_params,
"""

_CASE1_ANCHOR = """        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;
"""

_CASE1_TUNING = r'''

            if constexpr (type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ3_XXS) {
                const bool tune_vdr = bigcherry_iq_mmvq_env_enabled("BIGCHERRY_IQ_MMVQ_VDR");
                const bool tune_nwarps = bigcherry_iq_mmvq_env_enabled("BIGCHERRY_IQ_MMVQ_NWARPS");
                const bool target_arch = table_id == MMVQ_PARAMETERS_RDNA3_0 || table_id == MMVQ_PARAMETERS_RDNA4;

                if (target_arch && (tune_vdr || tune_nwarps)) {
                    const auto launch_iq = [&](auto vdr_tag, auto nwarps_tag) {
                        constexpr int c_vdr = decltype(vdr_tag)::value;
                        constexpr int c_nwarps = decltype(nwarps_tag)::value;
                        const dim3 block_nums(nrows_x, nchannels_dst, nsamples_dst);
                        const dim3 block_dims(warp_size, c_nwarps, 1);
                        bigcherry_iq_mmvq_trace(type, table_id, tune_vdr, tune_nwarps);
                        mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, c_vdr, c_nwarps>(
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                            stride_sample_x, stride_sample_y, stride_sample_dst, block_nums, block_dims, 0, ids_stride,
                            stream);
                    };

                    if constexpr (type == GGML_TYPE_IQ4_XS) {
                        // Candidate matrix: gfx1100 baseline=VDR4/NW1 -> VDR2/NW1;
                        // gfx1201 baseline=VDR4/NW8 -> VDR2/NW4.
                        if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
                            if (tune_vdr && tune_nwarps) launch_iq(std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{});
                            else if (tune_vdr)           launch_iq(std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{});
                            else                        launch_iq(std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{});
                        } else {
                            if (tune_vdr && tune_nwarps) launch_iq(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{});
                            else if (tune_vdr)           launch_iq(std::integral_constant<int, 2>{}, std::integral_constant<int, 8>{});
                          else                        launch_iq(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
                        }
                    } else {
                        // Candidate matrix: both target arches baseline=VDR2/NW1 -> VDR1/NW2.
                        if (tune_vdr && tune_nwarps) launch_iq(std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{});
                        else if (tune_vdr)           launch_iq(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{});
                        else                        launch_iq(std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{});
                    }
                    return;
                }
            }
'''

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/vecdotq.cuh",
        description="1273: correctness-preserving lower-VDR IQ3_XXS/IQ4_XS vec-dot variants",
        language="none",
        edits=(
            Edit(
                id="iq3-xxs-vdr1",
                anchor=re.escape(_IQ3_NEXT_OLD),
                mode="insert_before",
                text=_IQ3_VDR1,
                guard=r"vec_dot_iq3_xxs_q8_1_vdr1\(",
                rationale="Attach the VDR=1 IQ3_XXS implementation immediately after the pristine VDR=2 function and before the IQ3_S section.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="iq4-xs-vdr2",
                anchor=re.escape(_IQ4_FUNC_HEAD),
                mode="insert_before",
                text=_IQ4_VDR2,
                guard=r"vec_dot_iq4_xs_q8_1_vdr2\(",
                rationale="Place the VDR=2 IQ4_XS implementation adjacent to the pristine IQ4_XS vec-dot so the host selector can compile either path.",
                expect_matches=1,
                max_span_lines=3,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/mmvq.cu",
        description="1273: opt-in gfx1100/gfx1201 IQ MMVQ VDR/nwarps launch variants with trace evidence",
        language="none",
        edits=(
            Edit(
                id="iq-mmvq-includes",
                anchor=re.escape(_INCLUDES_OLD),
                mode="replace",
                text=_INCLUDES_NEW,
                guard=r"#include <atomic>\n#include <cstdint>\n#include <cstdlib>",
                rationale="The opt-in host selector needs getenv and the once-per-process activation marker needs std::atomic_flag.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="iq-mmvq-vdr-selector",
                anchor=re.escape(_VDR_FN_HEAD),
                mode="insert_before",
                text=_VEC_SELECTOR,
                guard=r"bigcherry_iq_vec_dot_q_cuda",
                rationale="Resolve lower-VDR IQ kernels at compile time while falling back to the pristine vec-dot selector for every other type/VDR.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="iq-mmvq-runtime-support",
                anchor=re.escape(_CALC_NWARPS_HEAD),
                mode="insert_before",
                text=_TUNING_SUPPORT,
                guard=r"bigcherry_iq_mmvq_env_enabled",
                rationale="Keep env parsing and activation evidence on the host side after architecture table IDs are defined and before launch geometry is computed.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="iq-mmvq-kernel-template",
                anchor=re.escape(_KERNEL_HEAD_OLD),
                mode="replace",
                text=_KERNEL_HEAD_NEW,
                guard=r"template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false,\n          int iq_vdr = 0, int iq_nwarps = 0>\n__launch_bounds__",
                rationale="Compile explicit VDR/nwarps variants without changing the default template instantiation used by pristine callers.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="iq-mmvq-kernel-constants",
                anchor=re.escape(_KERNEL_CONSTS_OLD),
                mode="replace",
                text=_KERNEL_CONSTS_NEW,
                guard=r"bigcherry_iq_vec_dot_q_cuda<type, vdr>",
                rationale="Make K-loop stride, reduction width and vec-dot entry point follow the selected compile-time variant; zero overrides preserve pristine constants.",
                expect_matches=1,
                max_span_lines=12,
            ),
            Edit(
                id="iq-mmvq-switch-fusion-template",
                anchor=re.escape(_SWITCH_FUSION_HEAD_OLD),
                mode="replace",
                text=_SWITCH_FUSION_HEAD_NEW,
                guard=r"template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false,\n         int iq_vdr = 0, int iq_nwarps = 0>",
                rationale="Thread compile-time IQ overrides through the existing fusion-preserving launch wrapper.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="iq-mmvq-fused-launch",
                anchor=re.escape(_FUSED_LAUNCH_OLD),
                mode="replace",
                text=_FUSED_LAUNCH_NEW,
                guard=r"mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, iq_vdr, iq_nwarps>",
                rationale="Pass IQ overrides to the fused ncols=1 kernel instantiation.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="iq-mmvq-plain-launch",
                anchor=re.escape(_PLAIN_LAUNCH_OLD),
                mode="replace",
                text=_PLAIN_LAUNCH_NEW,
                guard=r"mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, iq_vdr, iq_nwarps>",
                rationale="Pass IQ overrides to the ordinary MMVQ kernel instantiation.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="iq-mmvq-decode-dispatch",
                anchor=re.escape(_CASE1_ANCHOR),
                mode="insert_after",
                text=_CASE1_TUNING,
                guard=r"BIGCHERRY_IQ_MMVQ_VDR",
                rationale="At the real single-token MMVQ host launch site, dispatch only IQ4_XS/IQ3_XXS on gfx1100/gfx1201 into independently gated VDR/nwarps candidates and return before pristine launch logic.",
                expect_matches=1,
                max_span_lines=4,
            ),
        ),
    ),
]
