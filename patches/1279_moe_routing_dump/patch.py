"""1279: full-tensor printing for eval-callback (MoE routing profile, QFN02).

common_debug prints at most the first/last three values per dimension, so
ffn_moe_topk (10 expert ids per token) is truncated. With
BIGCHERRY_DEBUG_FULL_TENSORS=1 the print limit becomes the tensor's full
extent. Default output is unchanged.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "diagnostic"
STATE = "untested"

_CALL = "        common_debug_print_tensor(data, t->type, t->ne, t->nb, 3, pimpl->abort_on_nan);\n"
_NEW = (
    "        // BigCherry 1279: BIGCHERRY_DEBUG_FULL_TENSORS=1 prints every value (MoE routing profile).\n"
    "        const int64_t print_n = getenv(\"BIGCHERRY_DEBUG_FULL_TENSORS\") != nullptr ? INT64_MAX / 4 : 3;\n"
    "        common_debug_print_tensor(data, t->type, t->ne, t->nb, print_n, pimpl->abort_on_nan);\n"
)

PATCHES = [
    FilePatch(
        path="common/debug.cpp",
        language="none",
        description="Env-gated full printing in the eval-callback tensor dump.",
        edits=(
            Edit(
                id="debug-full-tensors-include",
                anchor=_re.escape("#include <cmath>\n"),
                text="#include <cstdlib>\n",
                mode="insert_after",
                guard=r"#include <cstdlib>",
                expect_matches=1,
                rationale="getenv for the full-print switch.",
            ),
            Edit(
                id="debug-full-tensors",
                anchor=_re.escape(_CALL),
                text=_NEW,
                mode="replace",
                guard=r"BIGCHERRY_DEBUG_FULL_TENSORS",
                expect_matches=1,
                rationale="ffn_moe_topk has 10 ids per token; the default 3+3 window drops 4.",
            ),
        ),
    ),
]
