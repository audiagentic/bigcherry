"""1319 (QFP16): split llama_context::process_ubatch host time into graph build/reuse, set_inputs and graph_compute.

1317 measured llama_process(ctx_tgt) host time at 5-7 ms per verify round (15-17% of an MTP step) with graph reuse
working. With BIGCHERRY_SUBMIT_TIMING=1 every process_ubatch call logs
`BIGCHERRY_SUBMIT_TIMING ctx n_tokens reused graph_us inputs_us compute_us` (compute_us = graph_compute host time:
scheduler split/copy setup + per-backend async submit incl. HIP graph launch). submit_us (1317) minus the sum of these
is the work outside process_ubatch (batch split, memory/KV slot prep). Diagnostic only.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_A_START = "    auto * res = get_gf_res_prev();\n    auto * gf  = res->get_gf();\n"
_N_START = """    // bigcherry 1319: host-time split of process_ubatch (BIGCHERRY_SUBMIT_TIMING=1)
    static const bool bc_st_on = getenv("BIGCHERRY_SUBMIT_TIMING") != nullptr && atoi(getenv("BIGCHERRY_SUBMIT_TIMING")) != 0;
    const int64_t bc_st_t0 = bc_st_on ? ggml_time_us() : 0;
    const int bc_st_reused0 = n_reused;

    auto * res = get_gf_res_prev();
    auto * gf  = res->get_gf();
"""

_A_GRAPH = "        gf_res_prev_active = res;\n    }\n\n    // set the input data for the input tensors\n"
_N_GRAPH = """        gf_res_prev_active = res;
    }

    const int64_t bc_st_t1 = bc_st_on ? ggml_time_us() : 0;  // bigcherry 1319: graph build or reuse done

    // set the input data for the input tensors
"""

_A_COMPUTE = ("    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);\n"
              "    if (status != GGML_STATUS_SUCCESS) {\n")
_N_COMPUTE = """    const int64_t bc_st_t2 = bc_st_on ? ggml_time_us() : 0;  // bigcherry 1319: inputs set
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (bc_st_on) {
        const int64_t bc_st_t3 = ggml_time_us();
        LLAMA_LOG_WARN("BIGCHERRY_SUBMIT_TIMING ctx=%p n_tokens=%u reused=%d graph_us=%lld inputs_us=%lld compute_us=%lld\\n",
            (void *) this, ubatch.n_tokens, n_reused != bc_st_reused0 ? 1 : 0, (long long) (bc_st_t1 - bc_st_t0),
            (long long) (bc_st_t2 - bc_st_t1), (long long) (bc_st_t3 - bc_st_t2));
    }
    if (status != GGML_STATUS_SUCCESS) {
"""

PATCHES = [
    FilePatch(
        path="src/llama-context.cpp",
        description="1319: BIGCHERRY_SUBMIT_TIMING host-time split of process_ubatch",
        language="none",
        edits=(
            Edit(id="submit-timing-include", anchor=re.escape("#include <cinttypes>\n"), mode="insert_after",
                 text="#include <cstdlib>  // bigcherry 1319: getenv/atoi\n", guard=r"#include <cstdlib>  // bigcherry 1319",
                 rationale="First standard include of llama-context.cpp.", expect_matches=1, max_span_lines=2),
            Edit(id="submit-timing-start", anchor=re.escape(_A_START), mode="replace", text=_N_START,
                 guard=r"bigcherry 1319: host-time split of process_ubatch", rationale="process_ubatch entry.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="submit-timing-graph", anchor=re.escape(_A_GRAPH), mode="replace", text=_N_GRAPH,
                 guard=r"bigcherry 1319: graph build or reuse done", rationale="After the graph reuse/build branch.",
                 expect_matches=1, max_span_lines=5),
            Edit(id="submit-timing-compute", anchor=re.escape(_A_COMPUTE), mode="replace", text=_N_COMPUTE,
                 guard=r"BIGCHERRY_SUBMIT_TIMING ctx=", rationale="Around graph_compute (async submit).",
                 expect_matches=1, max_span_lines=3),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_SUBMIT_TIMING', '0|1', '0',
           'diagnostic: graph build/alloc/submit time per ubatch'),
)
