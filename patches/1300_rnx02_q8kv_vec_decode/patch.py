"""RNX02: opt-in native Q8_0/D=256 vector FlashAttention dispatch for decode."""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "untested"

_INCLUDES_OLD = '#include "fattn.cuh"\n'
_INCLUDES_NEW = '#include "fattn.cuh"\n\n#include <atomic>\n'

_SELECTOR = """    // BigCherry 1300: keep this experiment default-off. The existing Q8_0/D=256
    // vector instance avoids the competing whole-K/V F16 staging path; no new
    // attention kernel or dequantizer is introduced here.
    const char * rnx02_q8_vec = std::getenv(\"BIGCHERRY_RNX02_Q8_VEC\");
    const bool rnx02_q8_vec_enabled = rnx02_q8_vec != nullptr &&
        rnx02_q8_vec[0] == '1' && rnx02_q8_vec[1] == '\\0';
    if (rnx02_q8_vec_enabled && Q->ne[0] == 256 && Q->ne[1] == 1 &&
            K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 &&
            V->ne[0] == K->ne[0] &&
            ggml_cuda_get_fattn_vec_case(256, K->type, V->type) != nullptr &&
            (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc))) {
        if (std::getenv(\"BIGCHERRY_PATCH_TRACE\") != nullptr) {
            static std::once_flag rnx02_q8_vec_logged;
            std::call_once(rnx02_q8_vec_logged, [] {
                GGML_LOG_WARN(\"BIGCHERRY_PATCH_HIT patch=1300_rnx02_q8kv_vec_decode path=q8_vec_d256\\n\");
            });
        }
        return BEST_FATTN_KERNEL_VEC;
    }

"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/fattn.cu",
        description="RNX02 opt-in Q8_0/D=256 single-token vector attention dispatch.",
        edits=(
            Edit(
                id="rnx02-includes",
                anchor=_re.escape(_INCLUDES_OLD),
                mode="replace",
                text=_INCLUDES_NEW,
                guard=r"#include <atomic>",
                expect_matches=1,
                rationale="Provide the once-per-process trace marker used by the default-off experiment.",
                max_span_lines=2,
            ),
            Edit(
                id="rnx02-q8-vector-selector",
                anchor=_re.escape(
                    "    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;\n"
                ),
                mode="insert_after",
                text=_SELECTOR,
                guard=r"BIGCHERRY_RNX02_Q8_VEC",
                expect_matches=1,
                rationale="Select only the pre-existing Q8 D=256 vector case for the exact opt-in decode predicate before the generic AMD tensor-core choices.",
                max_span_lines=2,
            ),
        ),
    ),
]
