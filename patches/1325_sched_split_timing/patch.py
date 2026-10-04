"""1325 (QFP16): per-split host timing in ggml_backend_sched_compute_splits.

1319 measured the target's graph_compute host time at ~5.0 ms per verify round (~10K); 1320 attributed only ~2.2 ms to
the meta backend (97 subgraphs: 1.7 ms launches + 0.5 ms AllReduce enqueue). The rest is in the scheduler: other splits
(CPU token_embd / per_layer_token_embd get_rows) and the input copies/synchronization between splits. With
BIGCHERRY_SUBMIT_TIMING=1 each split logs `BIGCHERRY_SCHED_SPLIT i n_splits backend n_inputs input_us compute_us`:
  input_us   - input handling for the split (event/backend synchronize, cross-backend copies, MoE id copies)
  compute_us - ggml_backend_graph_compute_async host time (synchronous for the CPU backend)
Diagnostic only.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_A_LOOP = ("    int prev_backend_id = -1;\n"
           "\n"
           "    for (int split_id = 0; split_id < sched->n_splits; split_id++) {\n"
           "        struct ggml_backend_sched_split * split = &splits[split_id];\n")
_N_LOOP = ("    int prev_backend_id = -1;\n"
           "\n"
           "    // bigcherry 1325: per-split host timing (BIGCHERRY_SUBMIT_TIMING=1)\n"
           "    static const bool bc_ss_on = getenv(\"BIGCHERRY_SUBMIT_TIMING\") != nullptr && atoi(getenv(\"BIGCHERRY_SUBMIT_TIMING\")) != 0;\n"
           "\n"
           "    for (int split_id = 0; split_id < sched->n_splits; split_id++) {\n"
           "        struct ggml_backend_sched_split * split = &splits[split_id];\n"
           "        const int64_t bc_ss_t0 = bc_ss_on ? ggml_time_us() : 0;\n")

_A_COMPUTE = ("        if (!sched->callback_eval) {\n"
              "            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);\n"
              "            if (ec != GGML_STATUS_SUCCESS) {\n"
              "                return ec;\n"
              "            }\n")
_N_COMPUTE = ("        if (!sched->callback_eval) {\n"
              "            const int64_t bc_ss_t1 = bc_ss_on ? ggml_time_us() : 0;  // bigcherry 1325\n"
              "            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);\n"
              "            if (bc_ss_on) {\n"
              "                GGML_LOG_WARN(\"BIGCHERRY_SCHED_SPLIT i=%d n_splits=%d backend=%s n_inputs=%d n_nodes=%d input_us=%lld compute_us=%lld\\n\",\n"
              "                    split_id, sched->n_splits, ggml_backend_name(split_backend), split->n_inputs, split->graph.n_nodes,\n"
              "                    (long long) (bc_ss_t1 - bc_ss_t0), (long long) (ggml_time_us() - bc_ss_t1));\n"
              "            }\n"
              "            if (ec != GGML_STATUS_SUCCESS) {\n"
              "                return ec;\n"
              "            }\n")

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-backend.cpp",
        description="1325: BIGCHERRY_SUBMIT_TIMING per-split scheduler host timing",
        language="none",
        edits=(
            Edit(id="sched-timing-loop", anchor=re.escape(_A_LOOP), mode="replace", text=_N_LOOP,
                 guard=r"bigcherry 1325: per-split host timing", rationale="Split loop entry of compute_splits.",
                 expect_matches=1, max_span_lines=5),
            Edit(id="sched-timing-compute", anchor=re.escape(_A_COMPUTE), mode="replace", text=_N_COMPUTE,
                 guard=r"BIGCHERRY_SCHED_SPLIT i=", rationale="Per-split async compute submit.",
                 expect_matches=1, max_span_lines=6),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_SCHED_SPLIT', '0|1', '0',
           'diagnostic: scheduler per-split timing (with BIGCHERRY_SUBMIT_TIMING)'),
)
