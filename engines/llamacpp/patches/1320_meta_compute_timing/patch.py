"""1320 (QFP16): meta-backend graph_compute host-time breakdown.

1319 put the target's per-round submit cost (5 ms median) inside graph_compute. Under -sm tensor the meta backend
splits each graph at every AllReduce and, per subgraph and per device, calls ggml_backend_graph_compute_async (one HIP
graph launch each) and then the AllReduce. With BIGCHERRY_SUBMIT_TIMING=1 every meta graph_compute logs
`BIGCHERRY_META_TIMING n_nodes n_subgraphs rebuild rebuild_us launch_us allreduce_us total_us`:
  rebuild_us   - per-device subgraph (re)construction (skipped when the cgraph uid is unchanged)
  launch_us    - sum of the per-subgraph per-device ggml_backend_graph_compute_async host times
  allreduce_us - sum of the per-subgraph AllReduce enqueue host times
Diagnostic only.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_A_START = ("    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.\n"
            "    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);\n")
_N_START = _A_START + r"""    // bigcherry 1320: meta graph_compute host-time breakdown (BIGCHERRY_SUBMIT_TIMING=1)
    static const bool bc_mt_on = getenv("BIGCHERRY_SUBMIT_TIMING") != nullptr && atoi(getenv("BIGCHERRY_SUBMIT_TIMING")) != 0;
    const int64_t bc_mt_t0 = bc_mt_on ? ggml_time_us() : 0;
    int64_t bc_mt_t_exec = 0, bc_mt_launch = 0, bc_mt_ar = 0;
"""

_A_EXEC = ("    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {\n"
           "        for (size_t j = 0; j < n_backends; j++) {\n"
           "            auto & bcj = backend_ctx->backend_configs[j];\n"
           "            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);\n")
_N_EXEC = ("    if (bc_mt_on) {  // bigcherry 1320: rebuild done, execution starts\n"
           "        bc_mt_t_exec = ggml_time_us();\n"
           "    }\n"
           "    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {\n"
           "        const int64_t bc_mt_l0 = bc_mt_on ? ggml_time_us() : 0;\n"
           "        for (size_t j = 0; j < n_backends; j++) {\n"
           "            auto & bcj = backend_ctx->backend_configs[j];\n"
           "            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);\n")

# 0830 (experiment sets) inserts a declaration after this line and extends the fallback block, so neither anchor
# below includes those lines.
_A_AR = "        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {\n"
_N_AR = ("        const int64_t bc_mt_l1 = bc_mt_on ? ggml_time_us() : 0;  // bigcherry 1320\n"
         "        bc_mt_launch += bc_mt_l1 - bc_mt_l0;\n"
         "        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {\n")

_A_END = ("                if (status != GGML_STATUS_SUCCESS) {\n"
          "                    return status;\n"
          "                }\n"
          "            }\n"
          "        }\n"
          "    }\n"
          "    return GGML_STATUS_SUCCESS;\n"
          "}\n")
_N_END = ("                if (status != GGML_STATUS_SUCCESS) {\n"
          "                    return status;\n"
          "                }\n"
          "            }\n"
          "        }\n"
          "        if (bc_mt_on) {  // bigcherry 1320\n"
          "            bc_mt_ar += ggml_time_us() - bc_mt_l1;\n"
          "        }\n"
          "    }\n"
          "    if (bc_mt_on) {\n"
          "        const int64_t bc_mt_t1 = ggml_time_us();\n"
          "        GGML_LOG_WARN(\"BIGCHERRY_META_TIMING n_nodes=%d n_subgraphs=%zu rebuild=%d rebuild_us=%lld launch_us=%lld allreduce_us=%lld total_us=%lld\\n\",\n"
          "            cgraph->n_nodes, backend_ctx->n_subgraphs, needs_rebuild ? 1 : 0, (long long) (bc_mt_t_exec - bc_mt_t0),\n"
          "            (long long) bc_mt_launch, (long long) bc_mt_ar, (long long) (bc_mt_t1 - bc_mt_t0));\n"
          "    }\n"
          "    return GGML_STATUS_SUCCESS;\n"
          "}\n")

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-backend-meta.cpp",
        description="1320: BIGCHERRY_SUBMIT_TIMING meta-backend graph_compute breakdown",
        language="none",
        edits=(
            Edit(id="meta-timing-include", anchor=re.escape("#include <algorithm>\n"), mode="insert_after",
                 text="#include <cstdlib>  // bigcherry 1320: getenv/atoi\n", guard=r"#include <cstdlib>  // bigcherry 1320",
                 rationale="Standard include block of ggml-backend-meta.cpp.", expect_matches=1, max_span_lines=2),
            Edit(id="meta-timing-start", anchor=re.escape(_A_START), mode="replace", text=_N_START,
                 guard=r"bigcherry 1320: meta graph_compute host-time breakdown", rationale="graph_compute entry.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="meta-timing-exec", anchor=re.escape(_A_EXEC), mode="replace", text=_N_EXEC,
                 guard=r"bigcherry 1320: rebuild done, execution starts", rationale="Per-subgraph launch loop.",
                 expect_matches=1, max_span_lines=5),
            Edit(id="meta-timing-ar", anchor=re.escape(_A_AR), mode="replace", text=_N_AR,
                 guard=r"bc_mt_launch \+= bc_mt_l1 - bc_mt_l0;", rationale="Between launches and the AllReduce.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="meta-timing-end", anchor=re.escape(_A_END), mode="replace", text=_N_END,
                 guard=r"BIGCHERRY_META_TIMING n_nodes=", rationale="End of graph_compute.",
                 expect_matches=1, max_span_lines=10),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_META_TIMING', '0|1', '0',
           'diagnostic: meta backend per-device compute timing'),
)
