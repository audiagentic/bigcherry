"""PNRO14: gfx1151-only D=256/ncols=32 FlashAttention tile configuration."""

import re as _re

from bigcherry.patcher import Edit, FilePatch

_INCLUDES_OLD = """#include \"common.cuh\"\n#include \"fattn-common.cuh\"\n"""
_INCLUDES_NEW = """#include \"common.cuh\"\n#include \"fattn-common.cuh\"\n\n#include <atomic>\n#include <cstdlib>\n"""

_RDNA35_CONFIG = """static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_amd_rdna3_5(
        const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 4, 64, 64)
    return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
}

"""

_HOST_OLD = """static __host__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA(cc)) {
            return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
        }
        return ggml_cuda_fattn_tile_get_config_amd(DKQ, DV, ncols);
    }
    if (fast_fp16_available(cc)) {
        return ggml_cuda_fattn_tile_get_config_nvidia_fp16(DKQ, DV, ncols);
    }
    return ggml_cuda_fattn_tile_get_config_nvidia_fp32(DKQ, DV, ncols);
}"""

_HOST_NEW = """static __host__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
            if (DKQ == 256 && DV == 256 && ncols == 32 && getenv(\"BIGCHERRY_PATCH_TRACE\") != nullptr) {
                static std::atomic_flag bigcherry_pnro14_logged = ATOMIC_FLAG_INIT;
                if (!bigcherry_pnro14_logged.test_and_set(std::memory_order_relaxed)) {
                    GGML_LOG_WARN(\"BIGCHERRY_PATCH_HIT patch=1270_pnro14_rdna35_fa_tile_d256 path=tile_d256_cols32 contract=PNRO14-RDNA35-FA-TILE-D256\\n\");
                }
            }
            return ggml_cuda_fattn_tile_get_config_amd_rdna3_5(DKQ, DV, ncols);
        }
        if (GGML_CUDA_CC_IS_RDNA(cc)) {
            return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
        }
        return ggml_cuda_fattn_tile_get_config_amd(DKQ, DV, ncols);
    }
    if (fast_fp16_available(cc)) {
        return ggml_cuda_fattn_tile_get_config_nvidia_fp16(DKQ, DV, ncols);
    }
    return ggml_cuda_fattn_tile_get_config_nvidia_fp32(DKQ, DV, ncols);
}"""

_DEVICE_OLD = """static constexpr __device__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols) {
#ifdef GGML_USE_HIP
#ifdef RDNA
    return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
#else
    return ggml_cuda_fattn_tile_get_config_amd(DKQ, DV, ncols);
#endif // RDNA
#else
#ifdef FAST_FP16_AVAILABLE
    return ggml_cuda_fattn_tile_get_config_nvidia_fp16(DKQ, DV, ncols);
#else
    return ggml_cuda_fattn_tile_get_config_nvidia_fp32(DKQ, DV, ncols);
#endif // FAST_FP16_AVAILABLE
#endif // GGML_USE_HIP
}"""

_DEVICE_NEW = """static constexpr __device__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols) {
#ifdef GGML_USE_HIP
#if defined(RDNA3_5)
    return ggml_cuda_fattn_tile_get_config_amd_rdna3_5(DKQ, DV, ncols);
#elif defined(RDNA)
    return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
#else
    return ggml_cuda_fattn_tile_get_config_amd(DKQ, DV, ncols);
#endif // RDNA3_5 / RDNA
#else
#ifdef FAST_FP16_AVAILABLE
    return ggml_cuda_fattn_tile_get_config_nvidia_fp16(DKQ, DV, ncols);
#else
    return ggml_cuda_fattn_tile_get_config_nvidia_fp32(DKQ, DV, ncols);
#endif // FAST_FP16_AVAILABLE
#endif // GGML_USE_HIP
}"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/fattn-tile.cuh",
        description="PNRO14 dedicated gfx1151 D=256/ncols=32 FlashAttention tile row",
        edits=(
            Edit(
                id="pnro14-includes",
                anchor=_re.escape(_INCLUDES_OLD),
                mode="replace",
                text=_INCLUDES_NEW,
                guard=r"#include <atomic>\n#include <cstdlib>",
                rationale="Support the env-gated once-per-process activation marker.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="pnro14-rdna35-table",
                anchor=r"(?=static __host__ uint32_t ggml_cuda_fattn_tile_get_config\(const int DKQ, const int DV, const int ncols, const int cc\) \{)",
                mode="insert_before",
                text=_RDNA35_CONFIG,
                guard=r"ggml_cuda_fattn_tile_get_config_amd_rdna3_5",
                rationale="Add exactly one gfx1151 row; every other shape delegates to the existing RDNA table.",
                expect_matches=1,
                max_span_lines=1,
            ),
            Edit(
                id="pnro14-host-select",
                anchor=_re.escape(_HOST_OLD),
                mode="replace",
                text=_HOST_NEW,
                guard=r"bigcherry_pnro14_logged",
                rationale="Replace the complete host selector so RDNA3.5 routing and exact-shape activation cannot match another helper.",
                expect_matches=1,
                max_span_lines=14,
            ),
            Edit(
                id="pnro14-device-select",
                anchor=_re.escape(_DEVICE_OLD),
                mode="replace",
                text=_DEVICE_NEW,
                guard=r"#if defined\(RDNA3_5\)",
                rationale="Replace the complete device selector so compile-time launch-bounds selection matches the host selector.",
                expect_matches=1,
                max_span_lines=18,
            ),
        ),
    ),
]
