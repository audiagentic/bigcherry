"""1277: diagnostic AllReduce size/provider trace for the adaptive dispatcher (0840).

Lab-only diagnostic, never selected by a production recipe. When
BIGCHERRY_AR_SIZE_TRACE=<n> is set, the first <n> adaptive AllReduce calls log
their byte size, element shape and the provider 0840 chose, so prefill and
decode AllReduce size populations can be compared against the switch
threshold. Unset (the default) adds one predictable branch per call.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "gpu-collectives"
STATE = "untested"

_DISPATCH_ANCHOR = (
    "    const bool prefer_internal = have_internal && (reduction_bytes < switch_bytes || !have_rccl);\n"
)
_TRACE = r'''    {
        static const long trace_max = [] {
            const char * v = getenv("BIGCHERRY_AR_SIZE_TRACE");
            return v != nullptr ? strtol(v, nullptr, 10) : 0L;
        }();
        static long trace_count = 0;
        if (trace_count < trace_max) {
            ++trace_count;
            const ggml_tensor * t0 = tensors != nullptr ? tensors[0] : nullptr;
            GGML_LOG_WARN("BIGCHERRY_AR_SIZE bytes=%zu ne0=%lld ne1=%lld type=%s provider=%s switch=%zu\n",
                reduction_bytes,
                t0 != nullptr ? (long long) t0->ne[0] : -1LL,
                t0 != nullptr ? (long long) t0->ne[1] : -1LL,
                t0 != nullptr ? ggml_type_name(t0->type) : "none",
                prefer_internal ? "internal" : (have_rccl ? "rccl" : "internal"),
                switch_bytes);
        }
    }
'''

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        language="none",
        description="Env-gated per-call size/provider trace in the 0840 adaptive AllReduce dispatcher.",
        edits=(
            Edit(
                id="ar-size-trace",
                anchor=_re.escape(_DISPATCH_ANCHOR),
                text=_TRACE,
                mode="insert_after",
                guard=r"BIGCHERRY_AR_SIZE bytes=",
                expect_matches=1,
                rationale="Log the size and the provider decision right where 0840 makes it.",
            ),
        ),
    ),
]
