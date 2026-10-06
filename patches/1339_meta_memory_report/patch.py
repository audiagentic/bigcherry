"""1339: per-device memory report of the tensor-split (meta) backend.

The server logs one size per Meta buffer, which hides what each card holds. With BIGCHERRY_META_MEM=1 the meta
buffer type prints, for every buffer it creates, the real size on each simple device:

  - compute arenas (ggml_backend_meta_buffer_type_alloc_buffer): the scheduler's arena is allocated at the same
    size on every device, whatever share of the graph the device computes;
  - static buffers (alloc_buffer_n: weights, KV, indexer state): one line per device with the size of its slices,
    the tensor count and the first tensor's name, so cache_k / cache_idx_* / weight buffers can be told apart.

Diagnostic only; nothing is allocated differently.
"""
import re as _re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "core"
STATE = "untested"

_A_INC = "#include <cstdint>\n#include <cstring>\n"
_N_INC = "#include <cstdint>\n#include <cstdio>\n#include <cstdlib> // BigCherry 1339: getenv\n#include <cstring>\n"

_A_COMPUTE = "        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));\n    }\n"
_N_COMPUTE = _A_COMPUTE + r"""    if (getenv("BIGCHERRY_META_MEM") != nullptr && atoi(getenv("BIGCHERRY_META_MEM")) != 0) {
        // BigCherry 1339: the arena is the same size on every simple device
        for (size_t i = 0; i < n_simple_bufts; i++) {
            GGML_LOG_INFO("BIGCHERRY_META_MEM compute dev=%zu buft=%s size_mib=%.2f\n", i,
                ggml_backend_buft_name(ggml_backend_meta_buft_simple_buft(buft, i)), ggml_backend_buffer_get_size(bufs[i]) / 1024.0 / 1024.0);
        }
    }
"""

_A_STATIC = "        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));\n"
_N_STATIC = _A_STATIC + r"""        if (getenv("BIGCHERRY_META_MEM") != nullptr && atoi(getenv("BIGCHERRY_META_MEM")) != 0) {
            // BigCherry 1339: this device's slices of the buffer's tensors
            GGML_LOG_INFO("BIGCHERRY_META_MEM static dev=%zu buft=%s size_mib=%.2f tensors=%d first=%s\n", i, ggml_backend_buft_name(simple_buft),
                ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()) / 1024.0 / 1024.0, n_tensors, n_tensors > 0 ? tensors[0]->name : "");
        }
"""

# fusion candidates refused because an output overlaps a live input: the check compares real addresses, so the
# count depends on the arena layout (MSM02: compact per-device arenas against the common arena with its holes)
_A_FUSE_FN = "static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,\n"
_N_FUSE_FN = r"""// BigCherry 1339: how often a fusion was refused for memory overlap, printed at exit under BIGCHERRY_META_MEM
namespace {
struct bc_fusion_overlap_stats_t {
    uint64_t checks = 0, refused = 0;
    ~bc_fusion_overlap_stats_t() {
        if (checks > 0 && getenv("BIGCHERRY_META_MEM") != nullptr && atoi(getenv("BIGCHERRY_META_MEM")) != 0) {
            fprintf(stderr, "BIGCHERRY_META_MEM fusion_overlap checks=%llu refused=%llu\n",
                    (unsigned long long) checks, (unsigned long long) refused);
        }
    }
};
bc_fusion_overlap_stats_t bc_fusion_overlap_stats;
}

""" + _A_FUSE_FN

_A_FUSE_OK = "    bool is_ok = true;\n"
_N_FUSE_OK = "    bool is_ok = true;\n    bc_fusion_overlap_stats.checks++; // BigCherry 1339\n"

_A_FUSE_REFUSE = "                    if (!found) {\n                        is_ok = false;\n"
_N_FUSE_REFUSE = _A_FUSE_REFUSE + "                        bc_fusion_overlap_stats.refused++; // BigCherry 1339\n"

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1339: count fusion candidates refused for memory overlap",
        language="none",
        edits=(
            Edit(id="meta-mem-fusion-stats", anchor=_re.escape(_A_FUSE_FN), mode="replace", text=_N_FUSE_FN,
                 guard=r"struct bc_fusion_overlap_stats_t \{", rationale="Directly before the overlap check.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="meta-mem-fusion-checks", anchor=_re.escape(_A_FUSE_OK), mode="replace", text=_N_FUSE_OK,
                 guard=r"bc_fusion_overlap_stats\.checks\+\+;", rationale="Entry of the overlap check.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="meta-mem-fusion-refused", anchor=_re.escape(_A_FUSE_REFUSE), mode="replace", text=_N_FUSE_REFUSE,
                 guard=r"bc_fusion_overlap_stats\.refused\+\+;", rationale="The one place the check refuses a fusion.",
                 expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend-meta.cpp",
        description="1339: BIGCHERRY_META_MEM=1 prints the per-device size of every meta buffer",
        language="none",
        edits=(
            Edit(id="meta-mem-include", anchor=_re.escape(_A_INC), mode="replace", text=_N_INC,
                 guard=r"#include <cstdlib> // BigCherry 1339: getenv", rationale="Standard includes of the file.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="meta-mem-compute", anchor=_re.escape(_A_COMPUTE), mode="replace", text=_N_COMPUTE,
                 guard=r"BIGCHERRY_META_MEM compute dev=", rationale="After the per-device allocation loop of the arena path.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="meta-mem-static", anchor=_re.escape(_A_STATIC), mode="replace", text=_N_STATIC,
                 guard=r"BIGCHERRY_META_MEM static dev=", rationale="After each device's buffer of the per-tensor path is allocated.",
                 expect_matches=1, max_span_lines=2),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_META_MEM", "0|1", "0",
           "tensor split: log the size of every meta buffer on each device (compute arenas, and static buffers with "
           "their first tensor's name)"),
)
