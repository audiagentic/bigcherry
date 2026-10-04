"""1302: survive CUDA/HIP graph instantiation OOM by evicting the context's other cached graphs.

ggml_backend_cuda_context keeps one ggml_cuda_graph (captured graph + executable instance) per graph key and
only evicts entries unused for 10 s. Near the context limit the executable instances of other shapes hold the
last device memory and the next instantiation aborts via CUDA_CHECK. Both instantiate sites now go through a
helper that, on cudaErrorMemoryAllocation, destroys every other cached graph, synchronises and retries once.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_UPDATE_HEAD = "static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {\n"

_HELPER = """\
// bigcherry 1302: instantiate a captured graph; on out-of-memory evict every other cached graph of this
// context (each holds an executable instance in device memory), synchronise and retry once.
static void bigcherry_cuda_graph_instantiate(ggml_backend_cuda_context * cuda_ctx, ggml_cuda_graph * graph) {
    cudaError_t err = cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0);
    if (err == cudaErrorMemoryAllocation) {
        (void) cudaGetLastError();
        graph->instance = nullptr;
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
        size_t evicted = 0;
        for (auto it = cuda_ctx->cuda_graphs.begin(); it != cuda_ctx->cuda_graphs.end(); ) {
            if (it->second.get() != graph) {
                it = cuda_ctx->cuda_graphs.erase(it);
                evicted++;
            } else {
                ++it;
            }
        }
        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1302_graph_oom_evict device=%d evicted=%zu\\n", cuda_ctx->device, evicted);
        err = cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0);
    }
    CUDA_CHECK(err);
}

"""

_INSTANTIATE = "CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));"

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1302: graph instantiation OOM evicts the context's other cached graphs and retries",
        language="none",
        edits=(
            Edit(
                id="graph-oom-helper",
                anchor=re.escape(_UPDATE_HEAD),
                mode="insert_before",
                text=_HELPER,
                guard=r"static void bigcherry_cuda_graph_instantiate\(",
                rationale="Define the helper just ahead of the first instantiate site (the executable-update path).",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="graph-oom-sites",
                anchor=re.escape(_INSTANTIATE),
                mode="replace_all",
                text="bigcherry_cuda_graph_instantiate(cuda_ctx, graph);",
                guard=r"bigcherry_cuda_graph_instantiate\(cuda_ctx, graph\);",
                rationale="Both upstream instantiate sites (re-instantiate after a failed exec update, first instantiate before launch).",
                expect_matches=2,
                max_span_lines=1,
            ),
        ),
    ),
]
