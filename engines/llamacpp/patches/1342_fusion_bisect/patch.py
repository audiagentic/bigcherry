"""1342: bisect and count GPU kernel fusions (FKE01).

The GPU backend has one switch for every fusion (GGML_CUDA_DISABLE_FUSION). To find which fused launch is not
bit-identical to the node sequence it replaces, and what each family is worth, this diagnostic adds:

  - BIGCHERRY_FUSION_SKIP_OPS=OP[,OP...]: no fusion is attempted at a node whose op is listed (ggml_op_name, e.g.
    MUL_MAT_ID, MUL_MAT, RMS_NORM, UNARY, SCALE, ROPE, SSM_CONV, ADD); every other fusion stays on. A fusion is
    identified by the op of its first node, which separates the families well enough to bisect.
  - under BIGCHERRY_META_MEM=1, at exit, one line per first-node op: fusions taken and nodes elided.

Nothing changes without the variables.
"""
import re as _re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "core"
STATE = "validated"

_A_TRY = (
    "static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {\n"
    "\n"
    '    static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));\n'
    "    if (disable_fusion) {\n"
    "        return 0;\n"
    "    }\n"
)
_N_TRY = _A_TRY + r"""
    {
        // BigCherry 1342: bisect - no fusion that starts at one of the listed ops
        static const std::string bc_skip_ops = getenv("BIGCHERRY_FUSION_SKIP_OPS") != nullptr ?
            std::string(",") + getenv("BIGCHERRY_FUSION_SKIP_OPS") + "," : std::string();
        if (!bc_skip_ops.empty() &&
                bc_skip_ops.find(std::string(",") + ggml_op_name(cgraph->nodes[i]->op) + ",") != std::string::npos) {
            return 0;
        }
    }
"""

_A_STATS = "// try and fuse nodes and return the number of nodes to skip\n"
_N_STATS = r"""// BigCherry 1342: fusions taken per first-node op, printed at exit under BIGCHERRY_META_MEM
namespace {
struct bc_fusion_taken_stats_t {
    std::map<std::string, std::pair<uint64_t, uint64_t>> by_op; // op -> (fusions, nodes elided)
    ~bc_fusion_taken_stats_t() {
        if (getenv("BIGCHERRY_META_MEM") == nullptr || atoi(getenv("BIGCHERRY_META_MEM")) == 0) {
            return;
        }
        for (const auto & kv : by_op) {
            fprintf(stderr, "BIGCHERRY_META_MEM fusion_taken op=%s fusions=%llu nodes_elided=%llu\n", kv.first.c_str(),
                    (unsigned long long) kv.second.first, (unsigned long long) kv.second.second);
        }
    }
};
bc_fusion_taken_stats_t bc_fusion_taken_stats;
}

""" + _A_STATS

_A_CALL = (
    "                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);\n"
    "\n"
    "                if (nodes_to_skip != 0) {\n"
)
_N_CALL = _A_CALL + r"""                    {
                        // BigCherry 1342
                        static const bool bc_count = getenv("BIGCHERRY_META_MEM") != nullptr && atoi(getenv("BIGCHERRY_META_MEM")) != 0;
                        if (bc_count) {
                            auto & bc_e = bc_fusion_taken_stats.by_op[ggml_op_name(cgraph->nodes[i]->op)];
                            bc_e.first++;
                            bc_e.second += (uint64_t) nodes_to_skip;
                        }
                    }
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1342: fusion bisect by first-node op and per-family counters",
        language="none",
        edits=(
            Edit(id="fusion-bisect-stats", anchor=_re.escape(_A_STATS), mode="replace", text=_N_STATS,
                 guard=r"struct bc_fusion_taken_stats_t \{", rationale="Directly before ggml_cuda_try_fuse.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="fusion-bisect-skip", anchor=_re.escape(_A_TRY), mode="replace", text=_N_TRY,
                 guard=r"BigCherry 1342: bisect - no fusion that starts at one of the listed ops",
                 rationale="Entry of ggml_cuda_try_fuse, after the global switch and before e117148a4's runtime MMVQ/address-overlap admission checks.",
                 expect_matches=1, max_span_lines=7),
            Edit(id="fusion-bisect-count", anchor=_re.escape(_A_CALL), mode="replace", text=_N_CALL,
                 guard=r"bc_fusion_taken_stats\.by_op\[", rationale="The one call site in graph compute, where a fusion was taken.",
                 expect_matches=1, max_span_lines=4),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_FUSION_SKIP_OPS", "OP[,OP...]", "(unset)",
           "GPU backend: attempt no fusion at a node whose op (ggml_op_name) is listed; to bisect which fused launch "
           "differs from its unfused sequence and to measure each family"),
)
