"""PRBE55: opt-in Vulkan AMD/RDNA small-N DMMV route for speculative verification."""

import re

from bigcherry.patcher import Edit, FilePatch

_INCLUDE_OLD = '#include "ggml-vulkan-common.h"\n'
_INCLUDE_NEW = '#include "ggml-vulkan-common.h"\n\n#include <atomic>\n'

_HELPER_ANCHOR = """static bool ggml_vk_should_use_mmvq(const vk_device& device, uint32_t m, uint32_t n, uint32_t k, ggml_type src0_type) {
"""
_HELPER_TEXT = """// PRBE55 qualification-only override: keep the default selector untouched unless
// explicitly enabled, and restrict the experiment to AMD RDNA small-N batches.
static bool bigcherry_prbe55_force_dmmv_smalln(const vk_device & device, uint32_t n) {
    const char * value = getenv("BIGCHERRY_VK_SMALLN_DMMV");
    if (value == nullptr || strcmp(value, "1") != 0 || device->vendor_id != VK_VENDOR_ID_AMD) {
        return false;
    }
    switch (device->architecture) {
        case vk_device_architecture::AMD_RDNA1:
        case vk_device_architecture::AMD_RDNA2:
        case vk_device_architecture::AMD_RDNA3:
            return n >= 2 && n <= 8;
        default:
            return false;
    }
}

""" + _HELPER_ANCHOR

_SELECTOR_OLD = """    const bool f16_f32_kernel = src1->type == GGML_TYPE_F32;
    bool quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);

    vk_pipeline to_fp16_vk_0 = nullptr;
"""
_SELECTOR_NEW = """    const bool f16_f32_kernel = src1->type == GGML_TYPE_F32;
    bool quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);
    if (quantize_y && bigcherry_prbe55_force_dmmv_smalln(ctx->device, (uint32_t) ne11)) {
        quantize_y = false;
        if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
            static std::atomic_flag bigcherry_prbe55_logged = ATOMIC_FLAG_INIT;
            if (!bigcherry_prbe55_logged.test_and_set(std::memory_order_relaxed)) {
                GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1269_prbe55_vk_smalln_dmmv path=vk_smalln_dmmv contract=PRBE55-VK-SMALLN-DMMV n=%llu\\n",
                              (unsigned long long) ne11);
            }
        }
    }

    vk_pipeline to_fp16_vk_0 = nullptr;
"""

_TEST_ANCHOR = """        test_cases.emplace_back(new test_l2_norm_batch(GGML_TYPE_F32, { n, 16, 16, 1 }, 4, 1e-12f, true));
    }


    return test_cases;
}"""
_TEST_TEXT = """        test_cases.emplace_back(new test_l2_norm_batch(GGML_TYPE_F32, { n, 16, 16, 1 }, 4, 1e-12f, true));
    }

    // bigcherry PRBE55: Vulkan small-N MMVQ/DMMV routing boundaries.
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 4096, 1,   4096, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 4096, 2,   4096, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 4096, 3,   4096, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 4096, 4,   4096, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 4096, 5,   4096, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 4096, 6,   4096, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 4096, 7,   4096, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 4096, 8,   4096, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 4096, 128, 4096, {1, 1}, {1, 1}));

    return test_cases;
}"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-vulkan/ggml-vulkan.cpp",
        description="PRBE55 opt-in AMD/RDNA small-N DMMV selection",
        edits=(
            Edit(id="prbe55-atomic-include", anchor=re.escape(_INCLUDE_OLD), mode="replace", text=_INCLUDE_NEW,
                 guard=r"#include <atomic>", rationale="Thread-safe once-per-process activation marker.", expect_matches=1, max_span_lines=1),
            Edit(id="prbe55-helper", anchor=re.escape(_HELPER_ANCHOR), mode="replace", text=_HELPER_TEXT,
                 guard=r"bigcherry_prbe55_force_dmmv_smalln", rationale="Gate the experiment by env, AMD vendor, RDNA architecture, and N=2..8.", expect_matches=1, max_span_lines=1),
            Edit(id="prbe55-selector", anchor=re.escape(_SELECTOR_OLD), mode="replace", text=_SELECTOR_NEW,
                 guard=re.escape("PRBE55-VK-SMALLN-DMMV"), rationale="Force only an already-eligible MMVQ vector call onto the existing DMMV fallback path.", expect_matches=1, max_span_lines=4),
        ),
    ),
    FilePatch(
        path="tests/test-backend-ops.cpp",
        description="PRBE55 small-N MUL_MAT boundary coverage",
        edits=(Edit(id="prbe55-smalln-cases", anchor=re.escape(_TEST_ANCHOR), mode="replace", text=_TEST_TEXT,
                    guard=re.escape("bigcherry PRBE55: Vulkan small-N MMVQ/DMMV routing boundaries"), rationale="Exercise N=1, every 2..8 width, and a non-target batch.", expect_matches=1, max_span_lines=7),),
    ),
]
