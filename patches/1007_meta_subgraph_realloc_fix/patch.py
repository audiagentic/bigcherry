"""1007: meta backend re-creates every retained subgraph cgraph after resetting its graph context.

ggml_backend_meta_graph_compute keeps one cgraph per (device, subgraph) in a ggml context. When a graph has more
nodes than any before it (or more subgraphs), the context is reset - which frees ALL of those cgraphs - and they
are re-created, but only for the current graph's n_subgraphs and with the current graph's node capacity. The
bookkeeping (max_subgraphs, max_nnodes) keeps the high-water marks, so a later graph with more subgraphs than the
one that triggered the reset, but no more than max_subgraphs, takes no reallocation and writes through the stale
cgraph pointers of the freed context: SIGSEGV at `cgraph_ij->n_nodes = ...`.

Seen with 1332 (QFP22, build b-chunk7, no MTP): reserve (7139 / 6814 nodes), the chunked prefill graph (7197 nodes,
raises max_nnodes -> reset, fewer subgraphs re-created), then a dense 4-token batch (6810 nodes, 97 subgraphs) ->
crash on subgraph 96. It needs only three graph shapes in that order, so it is not specific to 1332.

Fix: after the reset re-create max_subgraphs cgraphs with max_nnodes capacity - exactly what the context was sized
for (mem_per_device_graphs_main = max_subgraphs * overhead(max_nnodes)).
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "untested"

_OLD = """                for (size_t i = 0; i < n_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), cgraph->n_nodes, /*grads =*/ false);
                }
"""
_NEW = """                // BigCherry 1007: the reset above freed every subgraph cgraph, re-create all the retained ones at the
                // retained capacity (a later graph with more subgraphs or nodes than this one reuses them)
                for (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ false);
                }
"""

PATCH = FilePatch(
    path="ggml/src/ggml-backend-meta.cpp",
    language="none",
    description="meta backend: re-create max_subgraphs cgraphs (max_nnodes capacity) after a graph context reset.",
    edits=(
        Edit(
            id="meta-recreate-all-subgraphs",
            anchor=re.escape(_OLD),
            text=_NEW,
            mode="replace",
            guard=r"BigCherry 1007: the reset above freed every subgraph cgraph",
            expect_matches=1,
            max_span_lines=4,
            rationale="The only cgraph_main allocation, directly after backend_ctx->ctx.reset in graph_compute.",
        ),
    ),
)

PATCHES = [PATCH]
