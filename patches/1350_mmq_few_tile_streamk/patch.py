"""1350: qualify upstream MMQ Stream-K for ordinary few-tile Q8_0 prefill (QFP37).

The b11474 RDNA Q8_0 tables select non-Stream-K kernels even when a projection exposes only a handful of
destination tiles. Flash-Next Gate 0 attributes about 15% of XTX prefill kernel time to the Q8_0 hyper-connection
projection class. This patch compiles a second specialization of the existing upstream Stream-K kernel/fixup and
selects it only for a narrow ordinary-Q8_0 few-tile class on gfx1100/gfx1201.

No Stream-K arithmetic or reduction algorithm is added. MUL_MAT_ID/MoE remains on the validated 1237/1265 compact
grid path. BIGCHERRY_MMQ_FEW_TILE_STREAMK=0 is the default/control during qualification.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_A_INCLUDES = r"""#include <climits>
#include <cstdint>
"""
_N_INCLUDES = r"""#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
"""

_A_KERNEL_TEMPLATE = r"""template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1), ggml_cuda_mmq_get_occupancy(type, J, fallback, prec_src1))
static __global__ void mul_mat_q(
"""
_N_KERNEL_TEMPLATE = r"""template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8, bool bc_force_stream_k = false>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1), ggml_cuda_mmq_get_occupancy(type, J, fallback, prec_src1))
static __global__ void mul_mat_q(
"""

_A_DEVICE_BRANCH = r"""    if constexpr (!ggml_cuda_mmq_get_stream_k(type, J, fallback, prec_src1)) {
"""
_N_DEVICE_BRANCH = r"""    if constexpr (!(bc_force_stream_k || ggml_cuda_mmq_get_stream_k(type, J, fallback, prec_src1))) {
"""

_A_LAUNCH_TEMPLATE = r"""template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
"""
_N_LAUNCH_TEMPLATE = r"""template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8, bool bc_force_stream_k = false>
static void bc_launch_mul_mat_q_impl(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
"""

_A_SMEM_LIMIT = r"""    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J, false, prec_src1>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J,  true, prec_src1>), nbytes_shared);
"""
_N_SMEM_LIMIT = r"""    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J, false, prec_src1, bc_force_stream_k>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J,  true, prec_src1, bc_force_stream_k>), nbytes_shared);
"""

_A_HOST_BRANCH = r"""    if (!config.stream_k) {
"""
_N_HOST_BRANCH = r"""    if (!bc_force_stream_k && !config.stream_k) {
"""

_A_STREAM_LAUNCH = r"""    mul_mat_q<type, J, fallback, prec_src1><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
"""
_N_STREAM_LAUNCH = r"""    mul_mat_q<type, J, fallback, prec_src1, bc_force_stream_k><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
"""

_A_WRAPPER = r"""template <ggml_type type, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
void mul_mat_q_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args,
                        cudaStream_t stream, int forced_J = 0) {
    int J_best = mul_mat_q_compute_J_best<type, fallback, prec_src1>(args);
"""
_N_WRAPPER = r"""static bool bc_mmq_few_tile_streamk_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("BIGCHERRY_MMQ_FEW_TILE_STREAMK");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return enabled;
}

static std::atomic_flag bc_mmq_few_tile_streamk_logged = ATOMIC_FLAG_INIT;

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    // QFP37 V1 is deliberately only the ordinary Q8_0 projection class seen in Flash-Next prefill.
    // Keep every other type/fallback specialization as a compile-time direct call to the native path.
    if constexpr (type == GGML_TYPE_Q8_0 && fallback && prec_src1 == GGML_PREC_Q8) {
        if (bc_mmq_few_tile_streamk_enabled() &&
                args.ids_dst == nullptr &&
                args.expert_bounds == nullptr &&
                args.nchannels_y == 1 &&
                args.nsamples_y == 1) {
            const int id = ggml_cuda_get_device();
            const int cc = ggml_cuda_info().devices[id].cc;
            const int nsm = ggml_cuda_info().devices[id].nsm;
            const ggml_cuda_mmq_config config =
                ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);

            const bool bc_arch_ok =
                cc == GGML_CUDA_CC_RDNA3 ||
                cc == GGML_CUDA_CC_OFFSET_AMD + 0x1201;

            if (bc_arch_ok && nsm > 0 && config.type != GGML_TYPE_COUNT && !config.stream_k) {
                const int nty = (args.nrows_x + config.I - 1) / config.I;
                const int ntx = (args.ncols_max + config.J - 1) / config.J;
                const int ntiles_dst = nty * ntx;
                const size_t bc_fixup_bytes =
                    (size_t) nsm * config.J * config.I * sizeof(float);

                // Physical-shape qualification, not a model dimension match:
                // - <= 8 output-row tiles: genuinely few-tile M dimension;
                // - fewer destination tiles than CUs: ordinary launch underfills the device;
                // - >= 8 K tiles and >= 128 columns: enough prefill work to amortize fixup;
                // - cap temporary storage to 8 MiB.
                const bool bc_candidate =
                    nty > 0 && nty <= 8 &&
                    ntx > 0 && ntiles_dst < nsm &&
                    args.ncols_x >= 8 * config.K_vram &&
                    args.ncols_max >= 128 &&
                    bc_fixup_bytes <= 8u * 1024u * 1024u;

                if (bc_candidate) {
                    if (!bc_mmq_few_tile_streamk_logged.test_and_set(std::memory_order_relaxed)) {
                        GGML_LOG_WARN(
                            "BIGCHERRY_PATCH_HIT patch=1350_mmq_few_tile_streamk "
                            "cc=%d I=%d J=%d tiles=%d cu=%d k=%lld n=%lld\n",
                            cc, config.I, config.J, ntiles_dst, nsm,
                            (long long) args.ncols_x, (long long) args.ncols_max);
                    }
                    bc_launch_mul_mat_q_impl<type, J, fallback, prec_src1, true>(ctx, args, stream);
                    return;
                }
            }
        }
    }

    bc_launch_mul_mat_q_impl<type, J, fallback, prec_src1, false>(ctx, args, stream);
}

"""


PATCHES = (
    FilePatch(
        path="ggml/src/ggml-cuda/mmq.cuh",
        description="1350: compile and narrowly dispatch upstream Stream-K for ordinary few-tile Q8_0 prefill",
        language="none",
        edits=(
            Edit(
                id="qfp37-streamk-includes",
                anchor=re.escape(_A_INCLUDES),
                mode="replace",
                text=_N_INCLUDES,
                guard=r"#include <atomic>\n#include <climits>\n#include <cstdint>\n#include <cstdlib>",
                rationale="MMQ header standard includes for cached env parsing and one-shot activation marker.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="qfp37-streamk-kernel-template",
                anchor=re.escape(_A_KERNEL_TEMPLATE),
                mode="replace",
                text=_N_KERNEL_TEMPLATE,
                guard=r"bool bc_force_stream_k = false>\n__launch_bounds__",
                rationale="Compile a second kernel specialization without changing the selected I/J/config geometry.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="qfp37-streamk-device-branch",
                anchor=re.escape(_A_DEVICE_BRANCH),
                mode="replace",
                text=_N_DEVICE_BRANCH,
                guard=r"bc_force_stream_k \|\| ggml_cuda_mmq_get_stream_k",
                rationale="The device specialization and host launcher must agree on ordinary versus Stream-K decomposition.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="qfp37-streamk-launch-template",
                anchor=re.escape(_A_LAUNCH_TEMPLATE),
                mode="replace",
                text=_N_LAUNCH_TEMPLATE,
                guard=r"static void bc_launch_mul_mat_q_impl",
                rationale="Keep the upstream launch body as an implementation selected by a narrow runtime wrapper, while forward-declaring the wrapper before 0300's forced-J helper calls it.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="qfp37-streamk-smem-limit",
                anchor=re.escape(_A_SMEM_LIMIT),
                mode="replace",
                text=_N_SMEM_LIMIT,
                guard=r"prec_src1, bc_force_stream_k>\), nbytes_shared",
                rationale="Apply the same dynamic shared-memory limit to the forced Stream-K specialization.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="qfp37-streamk-host-branch",
                anchor=re.escape(_A_HOST_BRANCH),
                mode="replace",
                text=_N_HOST_BRANCH,
                guard=r"if \(!bc_force_stream_k && !config.stream_k\)",
                rationale="Forced specialization enters the existing upstream Stream-K launch/fixup path.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="qfp37-streamk-kernel-launch",
                anchor=re.escape(_A_STREAM_LAUNCH),
                mode="replace",
                text=_N_STREAM_LAUNCH,
                guard=r"prec_src1, bc_force_stream_k><<<block_nums_stream_k",
                rationale="Launch the device specialization whose compile-time branch matches the host Stream-K path.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="qfp37-streamk-runtime-wrapper",
                anchor=re.escape(_A_WRAPPER),
                mode="insert_before",
                text=_N_WRAPPER,
                guard=r"BIGCHERRY_PATCH_HIT patch=1350_mmq_few_tile_streamk",
                rationale="Insert the physical-shape/architecture/env qualification before 0300's production mul_mat_q_switch_J.",
                expect_matches=1,
                max_span_lines=5,
            ),
        ),
    ),
)

ENV_DOCS = (
    EnvDoc(
        "BIGCHERRY_MMQ_FEW_TILE_STREAMK",
        "0|1",
        "0 (off)",
        "enable QFP37's ordinary Q8_0 few-tile prefill dispatch to the existing upstream MMQ Stream-K/fixup path "
        "on exact gfx1100/gfx1201 qualification candidates",
    ),
)
