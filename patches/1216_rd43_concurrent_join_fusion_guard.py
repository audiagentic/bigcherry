"""RD43 (AMD-STREAM-005): keep a concurrent-region join node out of
op-fusion, so op-fusion never crosses the join and leaves an aux stream
unjoined at cudaStreamEndCapture.

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            amd-ecosystem-llama-cpp
  repo:              https://github.com/AMD-Ecosystem/llama.cpp
  locator:           PR #71 (fork tracks per-PR, not per-branch -- see the
                     source's own registry notes)
  merge commit:      0f0db6292a90beb3a0f753baaa766c1454799a50
  title:             "ggml-cuda: keep the concurrent-region join node out
                     of op-fusion"
  reviewed:          2026-08-20 (RD28-RD53 validation pass); merged, not
                     ancestral to ggml-org/llama.cpp mainline (fork-only
                     work) and not ancestral to our b10502 pin.

What it does (correctness fix, prerequisite of RD42's concurrency being
enabled beyond the isolated single-experiment lane):
  The MoE shared-expert overlap (RD42, patch 1215) forks the shared expert
  onto an aux stream and joins it back at `join = ggml_add(ffn_moe_out,
  ffn_shexp)`. During CUDA/HIP-graph capture in decode, op-fusion can absorb
  that join `add` as the bias-add of the preceding shared-expert `mul_mat`
  (an add-family fusion); the executor then advances its node index past
  the join via `i += nodes_to_skip`, so the join handler never runs. Two
  consequences follow: the forked aux stream is never rejoined into the
  main stream, and `is_concurrent_event_active` is never cleared (only the
  first layer ever forks). At `cudaStreamEndCapture` the aux stream still
  holds unjoined work, so capture aborts with "capturing stream has
  unjoined work".

  Fix: while a concurrent region is active, cap the fusion horizon at the
  join node's index by temporarily lowering `cgraph->n_nodes` around
  `ggml_cuda_try_fuse` -- `ggml_can_fuse`'s only use of `n_nodes` is a
  bounds check, so any candidate fusion reaching index >= join is rejected
  and the join stays standalone. `use_counts` are independent of
  `n_nodes`, so fusions that stay inside the region (topk-moe, the routed
  branch) are unaffected. A `GGML_ASSERT` after the call fails loudly if a
  future fusion pattern ever bypasses the bounds-check cap and still
  crosses the join, instead of surfacing as a cryptic capture-time abort.
  Gated to the concurrent path only: prefill and the non-graph-opt path
  are unaffected.

Porting notes:
  - The anchor region (the node-processing loop inside
    ggml_cuda_graph_evaluate_and_capture, from
    `stream_ctx.concurrent_events.clear();` through the
    `ggml_cuda_try_fuse` call) is BYTE-IDENTICAL between the fork's PR
    base and our pinned b10502 tree, and untouched by patch 1215
    (RD39-RD42, which edits the join-node stream-selection branches above
    this point and the graph_optimize function below it, not this loop
    body) -- verified 2026-08-20 by direct comparison against
    vendor/llama.cpp. Apply this patch after 1215.

Hardware status (this patch's own acceptance, from the plan item):
  Needs a real decode run on a MoE-with-shared-expert model (e.g.
  GLM-4.7-Flash) with GGML_CUDA_GRAPH_OPT=1 to prove no capture abort and
  output parity against the =0 baseline. UNVALIDATED on BigCherry hardware
  as of porting -- STATE stays "untested".

Isolation and promotion (first-sweep policy, matching existing RD items):
  - GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the
    production 'framework' and 'validated-enhancements' patch-sets.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

PROVENANCE = {
    "source-id": "amd-ecosystem-llama-cpp",
    "plan-item": "RD43",
    "fork-commit": "0f0db6292a90beb3a0f753baaa766c1454799a50",
    "fork-commit-title": "ggml-cuda: keep the concurrent-region join node out of op-fusion",
    "snapshot-head": "58ab0a5f2ce3f426d657d55647846b03fbc1a20b",
    "snapshot-base": "58ab0a5f2ce3f426d657d55647846b03fbc1a20b",
    "adaptations": [
        "Target region byte-identical between the fork's PR #71 base and "
        "our b10502 pin, untouched by patch 1215.",
        "Added a GGML_ASSERT immediately after the exec_node_idx lookup "
        "(GPT review, 2026-08-20): the fork's own implementation left "
        "join_idx at -1 silently if the lookup ever missed, which would "
        "make the fusion cap below a silent no-op -- the exact hazard "
        "this patch exists to close. Fails loud at the lookup instead of "
        "only downstream if a fusion happens to reach far enough to "
        "cross the join.",
    ],
}

# --- hunk 1: build a node->execution-index map once per graph, right
# before the node-processing loop (used to find the join node's index).
_CLEAR_OLD = """            } else {
                stream_ctx.concurrent_events.clear();
            }

            for (int i = 0; i < cgraph->n_nodes; i++) {
"""

_CLEAR_NEW = """            } else {
                stream_ctx.concurrent_events.clear();
            }

            // Node -> index in the final (post-restore) execution order. Used to cap the op-fusion
            // horizon at a concurrent region's join node (see the try_fuse call below).
            std::unordered_map<const ggml_tensor *, int> exec_node_idx;
            if (should_launch_concurrent_events) {
                exec_node_idx.reserve(cgraph->n_nodes);
                for (int j = 0; j < cgraph->n_nodes; ++j) {
                    exec_node_idx[cgraph->nodes[j]] = j;
                }
            }

            for (int i = 0; i < cgraph->n_nodes; i++) {
"""

# --- hunk 2: cap the fusion horizon at the join node while a concurrent
# region is active, then fail loudly if a fusion ever crosses it anyway.
_TRY_FUSE_OLD = """                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
"""

_TRY_FUSE_NEW = """                int join_idx = -1;
                if (is_concurrent_event_active) {
                    auto it = exec_node_idx.find(concurrent_event->join_node);
                    // A concurrent event's own join_node must always be a
                    // real node in this same graph -- it was set from a node
                    // in this exact cgraph elsewhere in this function. If the
                    // lookup ever misses, that invariant is broken and the
                    // fusion-cap below would silently do nothing (join_idx
                    // stays -1), the exact hazard this patch exists to
                    // close. Fail loud immediately rather than deferring to
                    // the downstream assert, which only fires if a fusion
                    // actually reaches far enough to cross the join.
                    GGML_ASSERT(it != exec_node_idx.end() &&
                                "concurrent event join_node missing from this graph's node index");
                    join_idx = it->second;
                }

                const int saved_n_nodes = cgraph->n_nodes;
                if (join_idx > i) {
                    cgraph->n_nodes = join_idx;
                }
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
                cgraph->n_nodes = saved_n_nodes;

                GGML_ASSERT(!(join_idx > i && join_idx <= i + nodes_to_skip) &&
                            "op-fusion crossed a concurrent-region join node");
"""


PATCH = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="Cap op-fusion at a concurrent region's join node so it "
                "never absorbs it (amd-ecosystem PR #71 / RD43)",
    edits=(
        Edit(
            id="rd43-exec-node-idx-map",
            anchor=re.escape(_CLEAR_OLD),
            rationale="build the node->execution-index map used to find "
                      "the join node's cap index",
            mode="replace",
            text=_CLEAR_NEW,
            guard=r"Node -> index in the final \(post-restore\) execution order",
        ),
        Edit(
            id="rd43-cap-fusion-at-join",
            anchor=re.escape(_TRY_FUSE_OLD),
            rationale="temporarily lower cgraph->n_nodes around the "
                      "try_fuse call so fusion cannot reach past the join "
                      "node, plus a loud assert if it ever does anyway",
            mode="replace",
            text=_TRY_FUSE_NEW,
            guard=r"op-fusion crossed a concurrent-region join node",
        ),
    ),
)

PATCHES = [PATCH]
